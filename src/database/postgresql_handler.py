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
        WHERE c.processing_status != 'COMPLETED'
        GROUP BY c.claim_id, c.patient_age, c.provider_type, c.place_of_service,
                 c.total_charge_amount, c.service_date, c.claim_data, c.patient_id, c.provider_id
        ORDER BY c.service_date DESC
        LIMIT %s OFFSET %s
        """
        return self.execute_query(query, (limit, offset))
    
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
        """Mark claims as processed in bulk."""
        if not claim_ids:
            return
            
        query = """
        UPDATE edi.claims 
        SET processing_status = 'COMPLETED', processed_date = CURRENT_TIMESTAMP
        WHERE claim_id = ANY(%s)
        """
        self.execute_query(query, (claim_ids,))
    
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