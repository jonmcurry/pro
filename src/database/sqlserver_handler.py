# src/database/sqlserver_handler.py
"""
SQL Server Database Handler with Bulk Operations
"""
import logging
import pyodbc
from typing import Dict, List, Any, Optional
import threading
import time
from contextlib import contextmanager


class SQLServerHandler:
    """Optimized SQL Server handler for validation results storage."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_string = config['connection_string']
        self.pool_size = config.get('pool_size', 10)
        
        # Connection management
        self._connections = []
        self._connection_lock = threading.Lock()
        
        # Initialize connection pool
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize connection pool."""
        try:
            for _ in range(self.pool_size):
                conn = pyodbc.connect(self.connection_string)
                conn.autocommit = False
                self._connections.append(conn)
            
            self.logger.info(f"SQL Server connection pool initialized with {self.pool_size} connections")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQL Server connections: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = None
        try:
            with self._connection_lock:
                if self._connections:
                    conn = self._connections.pop()
                else:
                    # Create new connection if pool is empty
                    conn = pyodbc.connect(self.connection_string)
                    conn.autocommit = False
            
            yield conn
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                with self._connection_lock:
                    self._connections.append(conn)
    
    def store_validation_results_bulk(self, validation_results: List[Dict[str, Any]]) -> bool:
        """Store validation results using bulk insert."""
        if not validation_results:
            return True
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare bulk insert data
                insert_data = []
                for result in validation_results:
                    insert_data.append((
                        result.get('claim_id'),
                        result.get('validation_status', 'COMPLETED'),
                        str(result.get('predicted_filters', [])),
                        str(result.get('validation_results', [])),
                        result.get('processing_time', 0),
                        result.get('error_message', '')
                    ))
                
                # Bulk insert
                insert_query = """
                INSERT INTO dbo.ValidationResults 
                (claim_id, validation_status, predicted_filters, validation_details, 
                 processing_time, error_message, created_date)
                VALUES (?, ?, ?, ?, ?, ?, GETDATE())
                """
                
                cursor.executemany(insert_query, insert_data)
                conn.commit()
                
                self.logger.debug(f"Bulk inserted {len(validation_results)} validation results")
                return True
                
        except Exception as e:
            self.logger.error(f"Bulk insert error: {str(e)}")
            return False
    
    def cleanup_old_results(self, days_to_keep: int = 90) -> int:
        """Cleanup old validation results."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Archive old results first
                archive_query = """
                INSERT INTO dbo.ValidationResultsArchive
                SELECT * FROM dbo.ValidationResults
                WHERE created_date < DATEADD(day, -%s, GETDATE())
                """
                cursor.execute(archive_query, (days_to_keep,))
                
                # Delete old results
                delete_query = """
                DELETE FROM dbo.ValidationResults
                WHERE created_date < DATEADD(day, -%s, GETDATE())
                """
                cursor.execute(delete_query, (days_to_keep,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            return 0
    
    def get_validation_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get validation statistics for the specified period."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats_query = """
                SELECT 
                    COUNT(*) as total_validations,
                    SUM(CASE WHEN validation_status = 'COMPLETED' THEN 1 ELSE 0 END) as successful_validations,
                    SUM(CASE WHEN validation_status = 'ERROR' THEN 1 ELSE 0 END) as failed_validations,
                    AVG(processing_time) as avg_processing_time,
                    MIN(created_date) as earliest_validation,
                    MAX(created_date) as latest_validation
                FROM dbo.ValidationResults
                WHERE created_date >= DATEADD(day, -%s, GETDATE())
                """
                
                cursor.execute(stats_query, (days,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'total_validations': row[0] or 0,
                        'successful_validations': row[1] or 0,
                        'failed_validations': row[2] or 0,
                        'avg_processing_time': float(row[3] or 0),
                        'earliest_validation': row[4],
                        'latest_validation': row[5],
                        'success_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0
                    }
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Statistics query error: {str(e)}")
            return {}
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            self.logger.error(f"SQL Server connection test failed: {str(e)}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._connection_lock:
            return {
                'pool_size': self.pool_size,
                'available_connections': len(self._connections),
                'active_connections': self.pool_size - len(self._connections)
            }
    
    def close(self):
        """Close all connections."""
        with self._connection_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except:
                    pass
            self._connections.clear()
            self.logger.info("SQL Server connections closed")