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
        
        # Ensure required tables exist
        self.ensure_tables_exist()
    
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
                    self.logger.info("SQL Server connection pool empty or exhausted, creating new ad-hoc connection.")
                    conn = pyodbc.connect(self.connection_string)
                    # conn.autocommit = False # pyodbc connections default to autocommit=False
            
            # Ensure autocommit is False. pyodbc connections default to autocommit=False.
            # This is a safeguard in case it was changed.
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
                cursor.fast_executemany = True # Enable for better performance
                
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
                
                # First, check if archive table exists and create if needed
                self._ensure_archive_table_exists(cursor)
                
                # Get count of records to be archived
                count_query = """
                SELECT COUNT(*) FROM dbo.ValidationResults
                WHERE created_date < DATEADD(day, -?, GETDATE())
                """
                cursor.execute(count_query, (days_to_keep,))
                records_to_archive = cursor.fetchone()[0]
                
                if records_to_archive > 0:
                    # Archive old results with explicit column mapping
                    archive_query = """
                    INSERT INTO dbo.ValidationResultsArchive 
                    (claim_id, validation_status, predicted_filters, validation_details, 
                     processing_time, error_message, created_date, archived_date)
                    SELECT 
                        claim_id, validation_status, predicted_filters, validation_details,
                        processing_time, error_message, created_date, GETDATE()
                    FROM dbo.ValidationResults
                    WHERE created_date < DATEADD(day, -?, GETDATE())
                    """
                    cursor.execute(archive_query, (days_to_keep,))
                    
                    # Delete old results
                    delete_query = """
                    DELETE FROM dbo.ValidationResults
                    WHERE created_date < DATEADD(day, -?, GETDATE())
                    """
                    cursor.execute(delete_query, (days_to_keep,))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    self.logger.info(f"Archived and deleted {deleted_count} old validation results")
                    return deleted_count
                else:
                    self.logger.debug("No old records found to archive")
                    return 0
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            return 0
    
    def _ensure_archive_table_exists(self, cursor):
        """Ensure the archive table exists with correct structure."""
        try:
            # Check if archive table exists
            check_table_query = """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'ValidationResultsArchive'
            """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0
            
            if not table_exists:
                # Create archive table with same structure as main table plus archived_date
                create_archive_table = """
                CREATE TABLE dbo.ValidationResultsArchive (
                    result_id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    claim_id NVARCHAR(50),
                    validation_status NVARCHAR(20) DEFAULT 'COMPLETED',
                    predicted_filters NVARCHAR(MAX),
                    validation_details NVARCHAR(MAX),
                    processing_time DECIMAL(10,3) DEFAULT 0,
                    error_message NVARCHAR(MAX),
                    created_date DATETIME2 DEFAULT GETDATE(),
                    archived_date DATETIME2 DEFAULT GETDATE()
                )
                """
                cursor.execute(create_archive_table)
                
                # Add index for performance
                create_index = """
                CREATE INDEX IX_ValidationResultsArchive_ArchivedDate 
                ON dbo.ValidationResultsArchive(archived_date)
                """
                cursor.execute(create_index)
                
                self.logger.info("Created ValidationResultsArchive table")
                
        except Exception as e:
            self.logger.warning(f"Error ensuring archive table exists: {str(e)}")
    
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
                WHERE created_date >= DATEADD(day, -?, GETDATE())
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
    
    def get_recent_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent validation errors for troubleshooting."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                error_query = """
                SELECT TOP (?) 
                    claim_id,
                    error_message,
                    created_date,
                    processing_time
                FROM dbo.ValidationResults
                WHERE validation_status = 'ERROR'
                    AND error_message IS NOT NULL
                    AND error_message != ''
                ORDER BY created_date DESC
                """
                
                cursor.execute(error_query, (limit,))
                rows = cursor.fetchall()
                
                return [
                    {
                        'claim_id': row[0],
                        'error_message': row[1],
                        'created_date': row[2],
                        'processing_time': row[3]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Error fetching recent errors: {str(e)}")
            return []
    
    def get_validation_trends(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily validation trends."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                trend_query = """
                SELECT 
                    CAST(created_date AS DATE) as validation_date,
                    COUNT(*) as total_count,
                    SUM(CASE WHEN validation_status = 'COMPLETED' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN validation_status = 'ERROR' THEN 1 ELSE 0 END) as error_count,
                    AVG(processing_time) as avg_processing_time
                FROM dbo.ValidationResults
                WHERE created_date >= DATEADD(day, -?, GETDATE())
                GROUP BY CAST(created_date AS DATE)
                ORDER BY validation_date DESC
                """
                
                cursor.execute(trend_query, (days,))
                rows = cursor.fetchall()
                
                return [
                    {
                        'date': row[0],
                        'total_validations': row[1],
                        'successful_validations': row[2],
                        'failed_validations': row[3],
                        'avg_processing_time': float(row[4] or 0),
                        'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Error fetching validation trends: {str(e)}")
            return []
    
    def ensure_tables_exist(self):
        """Ensure required tables exist in the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if main ValidationResults table exists
                check_main_table = """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'ValidationResults'
                """
                cursor.execute(check_main_table)
                main_table_exists = cursor.fetchone()[0] > 0
                
                if not main_table_exists:
                    # Create main ValidationResults table
                    create_main_table = """
                    CREATE TABLE dbo.ValidationResults (
                        result_id BIGINT IDENTITY(1,1) PRIMARY KEY,
                        claim_id NVARCHAR(50) NOT NULL,
                        validation_status NVARCHAR(20) DEFAULT 'COMPLETED',
                        predicted_filters NVARCHAR(MAX),
                        validation_details NVARCHAR(MAX),
                        processing_time DECIMAL(10,3) DEFAULT 0,
                        error_message NVARCHAR(MAX),
                        created_date DATETIME2 DEFAULT GETDATE()
                    )
                    """
                    cursor.execute(create_main_table)
                    
                    # Add indexes
                    create_indexes = [
                        "CREATE INDEX IX_ValidationResults_ClaimId ON dbo.ValidationResults(claim_id)",
                        "CREATE INDEX IX_ValidationResults_CreatedDate ON dbo.ValidationResults(created_date)",
                        "CREATE INDEX IX_ValidationResults_Status ON dbo.ValidationResults(validation_status)"
                    ]
                    
                    for index_sql in create_indexes:
                        cursor.execute(index_sql)
                    
                    self.logger.info("Created ValidationResults table with indexes")
                
                # Ensure archive table exists
                self._ensure_archive_table_exists(cursor)
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error ensuring tables exist: {str(e)}")
            return False
    
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
            closed_count = 0
            errors_closing = 0
            for conn in self._connections:
                try:
                    conn.close()
                    closed_count += 1
                except pyodbc.Error as db_err:
                    self.logger.warning(f"Error closing a SQL Server connection: {db_err}")
                    errors_closing += 1
                except Exception as e:
                    self.logger.error(f"Unexpected error closing a SQL Server connection: {e}", exc_info=True)
                    errors_closing += 1

            self._connections.clear()
            self.logger.info(f"SQL Server connection pool closed. Connections processed: {closed_count + errors_closing}, "
                               f"Successfully closed: {closed_count}, Errors on close: {errors_closing}")