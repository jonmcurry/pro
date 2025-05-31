# src/database/postgresql_handler.py
"""
PostgreSQL Database Handler with Connection Pooling
"""
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional
import threading
import time


class PostgreSQLHandler:
    """Optimized PostgreSQL handler with connection pooling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._connection_pool = None
        self._pool_lock = threading.Lock()
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize connection pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.get('min_connections', 2),
                maxconn=self.config.get('max_connections', 10),
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                cursor_factory=RealDictCursor
            )
            self.logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise
    
    def get_connection(self):
        """Get connection from pool."""
        with self._pool_lock:
            return self._connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool."""
        with self._pool_lock:
            self._connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute query and return results."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                
                # Check if query returns results
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    return []
                    
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Query execution error: {str(e)}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_enriched_claims(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        """Get enriched claims with all related data in single query."""
        query = """
        SELECT 
            c.claim_id,
            c.patient_age,
            c.provider_type,
            c.place_of_service,
            c.total_charge_amount,
            c.service_date,
            c.claim_data,
            c.patient_id,
            c.provider_id,
            c.processing_status,
            -- Aggregate diagnoses
            COALESCE(
                array_agg(
                    DISTINCT jsonb_build_object(
                        'code', d.diagnosis_code,
                        'sequence', d.diagnosis_sequence,
                        'is_principal', d.is_principal,
                        'type', d.diagnosis_type,
                        'description', d.description
                    )
                ) FILTER (WHERE d.diagnosis_code IS NOT NULL),
                ARRAY[]::jsonb[]
            ) as diagnoses,
            -- Aggregate procedures
            COALESCE(
                array_agg(
                    DISTINCT jsonb_build_object(
                        'code', p.procedure_code,
                        'sequence', p.procedure_sequence,
                        'charge_amount', p.charge_amount,
                        'type', p.procedure_type,
                        'description', p.description,
                        'service_date', p.service_date,
                        'diagnosis_pointers', p.diagnosis_pointers
                    )
                ) FILTER (WHERE p.procedure_code IS NOT NULL),
                ARRAY[]::jsonb[]
            ) as procedures
        FROM edi.claims c
        LEFT JOIN edi.diagnoses d ON c.claim_id = d.claim_id
        LEFT JOIN edi.procedures p ON c.claim_id = p.claim_id
        WHERE c.processing_status IN ('PENDING', 'SENT', 'RETRY')
        GROUP BY c.claim_id, c.patient_age, c.provider_type, c.place_of_service,
                 c.total_charge_amount, c.service_date, c.claim_data, c.patient_id, 
                 c.provider_id, c.processing_status
        ORDER BY c.service_date DESC
        LIMIT %s OFFSET %s
        """
        return self.execute_query(query, (limit, offset))
    
    def get_total_unprocessed_claims(self) -> int:
        """Get count of unprocessed claims."""
        query = """
        SELECT COUNT(*) as total
        FROM edi.claims 
        WHERE processing_status IN ('PENDING', 'SENT', 'RETRY')
        """
        result = self.execute_query(query)
        return result[0]['total'] if result else 0
    
    def get_active_filters(self) -> List[Dict[str, Any]]:
        """Get active validation filters."""
        query = """
        SELECT filter_id, filter_name, rule_definition, description, rule_type
        FROM edi.filters
        WHERE active = true
        ORDER BY filter_id
        """
        return self.execute_query(query)
    
    def get_training_data_batch(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        """Get training data batch for ML model."""
        query = """
        SELECT 
            c.claim_id,
            c.patient_age,
            c.provider_type,
            c.place_of_service,
            c.total_charge_amount,
            c.patient_id,
            c.provider_id,
            COALESCE(
                array_agg(d.diagnosis_code) FILTER (WHERE d.diagnosis_code IS NOT NULL),
                ARRAY[]::text[]
            ) as diagnoses,
            COALESCE(
                array_agg(p.procedure_code) FILTER (WHERE p.procedure_code IS NOT NULL),
                ARRAY[]::text[]
            ) as procedures,
            -- Get applied filters
            COALESCE(
                array_agg(vr.filter_id) FILTER (WHERE vr.filter_id IS NOT NULL),
                ARRAY[]::int[]
            ) as applied_filters
        FROM edi.claims c
        LEFT JOIN edi.diagnoses d ON c.claim_id = d.claim_id
        LEFT JOIN edi.procedures p ON c.claim_id = p.claim_id
        LEFT JOIN edi.validation_results vr ON c.claim_id = vr.claim_id AND vr.passed = true
        WHERE c.processing_status = 'COMPLETED'
        GROUP BY c.claim_id, c.patient_age, c.provider_type, c.place_of_service, 
                 c.total_charge_amount, c.patient_id, c.provider_id
        LIMIT %s OFFSET %s
        """
        return self.execute_query(query, (limit, offset))
    
    def mark_claims_processed(self, claim_ids: List[str]):
        """Mark claims as processed in bulk using efficient UPDATE."""
        if not claim_ids:
            self.logger.warning("No claim IDs provided for processing update")
            return 0
            
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Use efficient bulk update with ANY() operator
                query = """
                UPDATE edi.claims 
                SET processing_status = 'COMPLETED', 
                    processed_date = CURRENT_TIMESTAMP
                WHERE claim_id = ANY(%s)
                  AND processing_status != 'COMPLETED'
                """
                
                cursor.execute(query, (claim_ids,))
                updated_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Successfully updated {updated_count} claims to COMPLETED status")
                return updated_count
                
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Bulk claims update error: {str(e)}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def mark_claims_processed_bulk(self, claim_ids: List[str]):
        """Alias for mark_claims_processed for backward compatibility."""
        return self.mark_claims_processed(claim_ids)
    
    def update_claim_status_batch(self, claim_status_updates: List[Dict[str, str]]):
        """Update multiple claims with different statuses in one operation."""
        if not claim_status_updates:
            return 0
            
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Create temporary table approach for complex batch updates
                cursor.execute("""
                CREATE TEMP TABLE temp_claim_updates (
                    claim_id VARCHAR(50),
                    new_status VARCHAR(20),
                    error_message TEXT
                ) ON COMMIT DROP
                """)
                
                # Insert update data
                update_data = [
                    (update['claim_id'], update['status'], update.get('error_message', ''))
                    for update in claim_status_updates
                ]
                
                cursor.executemany(
                    "INSERT INTO temp_claim_updates VALUES (%s, %s, %s)",
                    update_data
                )
                
                # Perform bulk update using JOIN
                cursor.execute("""
                UPDATE edi.claims 
                SET processing_status = temp_claim_updates.new_status,
                    processed_date = CURRENT_TIMESTAMP
                FROM temp_claim_updates
                WHERE edi.claims.claim_id = temp_claim_updates.claim_id
                """)
                
                updated_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Batch updated {updated_count} claims with various statuses")
                return updated_count
                
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Batch status update error: {str(e)}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        query = """
        SELECT 
            processing_status,
            COUNT(*) as count,
            MIN(service_date) as earliest_service,
            MAX(service_date) as latest_service,
            AVG(total_charge_amount) as avg_charge
        FROM edi.claims
        GROUP BY processing_status
        ORDER BY processing_status
        """
        
        results = self.execute_query(query)
        
        # Convert to dictionary format
        stats = {}
        total_claims = 0
        
        for row in results:
            status = row['processing_status']
            count = row['count']
            stats[status] = {
                'count': count,
                'earliest_service': row['earliest_service'],
                'latest_service': row['latest_service'],
                'avg_charge': float(row['avg_charge']) if row['avg_charge'] else 0
            }
            total_claims += count
        
        stats['total_claims'] = total_claims
        stats['unprocessed_count'] = sum(
            stats.get(status, {}).get('count', 0) 
            for status in ['PENDING', 'SENT', 'RETRY']
        )
        
        return stats
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            result = self.execute_query("SELECT 1 as test")
            return len(result) > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self._connection_pool:
            return {}
            
        return {
            'total_connections': self._connection_pool.maxconn,
            'available_connections': len(self._connection_pool._pool),
            'active_connections': self._connection_pool.maxconn - len(self._connection_pool._pool)
        }
    
    def clear_cache(self):
        """Clear internal cache."""
        with self._cache_lock:
            self._cache.clear()
    
    def close(self):
        """Close all connections."""
        if self._connection_pool:
            self._connection_pool.closeall()
            self.logger.info("PostgreSQL connections closed")