### src/parser.py
"""
Optimized Claim Parser - Fixes N+1 query problem and adds caching
"""
import logging
from typing import Dict, List, Any, Tuple, Iterator, Optional
import json
import time
from database.postgresql_handler import PostgreSQLHandler


class ClaimParser:
    """Optimized claim parsing with efficient batch processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_handler = None
        
        # Caching
        self._claim_count_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
        
    def initialize_database(self, db_handler: PostgreSQLHandler):
        """Initialize database connection."""
        self.db_handler = db_handler
        
    def get_total_claim_count(self) -> int:
        """Get total number of unprocessed claims with caching."""
        try:
            # Check cache
            current_time = time.time()
            if (self._claim_count_cache is not None and 
                self._cache_timestamp is not None and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._claim_count_cache
            
            # Query database
            query = """
            SELECT COUNT(*) as total_claims 
            FROM edi.claims 
            WHERE processing_status != 'COMPLETED'
            """
            result = self.db_handler.execute_query(query)
            count = result[0]['total_claims'] if result else 0
            
            # Update cache
            self._claim_count_cache = count
            self._cache_timestamp = current_time
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error getting claim count: {str(e)}")
            return 0
    
    def get_claim_chunks(self, chunk_size: int) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """
        OPTIMIZED: Generate claim chunks with single-query enrichment.
        Fixes the N+1 query problem by using JOINs instead of individual queries.
        """
        try:
            # Check for existing processed chunks to resume processing
            last_chunk_id = self._get_last_processed_chunk()
            
            chunk_id = last_chunk_id + 1
            offset = last_chunk_id * chunk_size
            
            while True:
                # OPTIMIZATION: Use enriched query that fetches everything in one go
                claims = self._fetch_enriched_claim_batch(offset, chunk_size)
                
                if not claims:
                    break
                    
                # Process JSON claim data
                processed_claims = self._process_claim_data(claims)
                
                yield chunk_id, processed_claims
                
                chunk_id += 1
                offset += chunk_size
                
        except Exception as e:
            self.logger.error(f"Error generating claim chunks: {str(e)}")
            raise
    
    def _fetch_enriched_claim_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Fetch claims with all related data in single query.
        This replaces the previous N+1 query pattern.
        """
        try:
            return self.db_handler.get_enriched_claims(limit, offset)
            
        except Exception as e:
            self.logger.error(f"Error fetching enriched claim batch: {str(e)}")
            return []
    
    def _process_claim_data(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Process claim data efficiently with batch operations.
        """
        try:
            processed_claims = []
            
            for claim in claims:
                # Parse JSON claim data if present
                if claim.get('claim_data'):
                    try:
                        parsed_data = json.loads(claim['claim_data'])
                        claim.update(parsed_data)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON in claim {claim['claim_id']}")
                
                # Convert JSON arrays back to lists for easier processing
                claim['diagnoses'] = claim.get('diagnoses', [])
                claim['procedures'] = claim.get('procedures', [])
                
                # Add derived fields for ML processing
                claim['diagnosis_count'] = len(claim['diagnoses'])
                claim['procedure_count'] = len(claim['procedures'])
                claim['primary_diagnosis'] = self._get_primary_diagnosis(claim['diagnoses'])
                claim['total_procedure_amount'] = self._calculate_procedure_total(claim['procedures'])
                
                processed_claims.append(claim)
            
            return processed_claims
            
        except Exception as e:
            self.logger.error(f"Error processing claim data: {str(e)}")
            return claims
    
    def _get_primary_diagnosis(self, diagnoses: List[Dict[str, Any]]) -> Optional[str]:
        """Get primary diagnosis code from diagnosis list."""
        try:
            # Look for principal diagnosis first
            for diag in diagnoses:
                if diag.get('is_principal', False):
                    return diag.get('code')
            
            # If no principal, get first in sequence
            if diagnoses:
                sorted_diags = sorted(diagnoses, key=lambda x: x.get('sequence', 999))
                return sorted_diags[0].get('code')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting primary diagnosis: {str(e)}")
            return None
    
    def _calculate_procedure_total(self, procedures: List[Dict[str, Any]]) -> float:
        """Calculate total procedure amount."""
        try:
            total = 0.0
            for proc in procedures:
                amount = proc.get('charge_amount', 0)
                if amount:
                    total += float(amount)
            return total
            
        except Exception as e:
            self.logger.error(f"Error calculating procedure total: {str(e)}")
            return 0.0
    
    def _get_last_processed_chunk(self) -> int:
        """Get the last successfully processed chunk ID for resume capability."""
        try:
            query = """
            SELECT COALESCE(MAX(chunk_id), 0) as last_chunk
            FROM edi.processed_chunks
            WHERE status = 'COMPLETED'
            """
            result = self.db_handler.execute_query(query)
            return result[0]['last_chunk'] if result else 0
            
        except Exception as e:
            self.logger.error(f"Error getting last processed chunk: {str(e)}")
            return 0
    
    def mark_chunk_processed(self, chunk_id: int, claims_count: int = 0, duration: float = 0):
        """
        OPTIMIZED: Mark a chunk as successfully processed with statistics.
        """
        try:
            query = """
            INSERT INTO edi.processed_chunks (
                chunk_id, status, processed_date, claims_count, processing_duration_seconds
            )
            VALUES (%s, 'COMPLETED', CURRENT_TIMESTAMP, %s, %s)
            ON CONFLICT (chunk_id) 
            DO UPDATE SET 
                status = 'COMPLETED', 
                processed_date = CURRENT_TIMESTAMP,
                claims_count = %s,
                processing_duration_seconds = %s
            """
            self.db_handler.execute_query(query, (chunk_id, claims_count, duration, claims_count, duration))
            
        except Exception as e:
            self.logger.error(f"Error marking chunk {chunk_id} as processed: {str(e)}")
    
    def mark_claims_processed_bulk(self, claim_ids: List[str]):
        """
        OPTIMIZED: Bulk update claim processing status.
        """
        try:
            if claim_ids:
                self.db_handler.mark_claims_processed(claim_ids)
                
        except Exception as e:
            self.logger.error(f"Error marking claims as processed: {str(e)}")
    
    def add_to_retry_queue(self, chunk_id: int, error_message: str):
        """Add failed chunk to retry queue."""
        try:
            query = """
            INSERT INTO edi.retry_queue (chunk_id, error_message, retry_count, created_date)
            VALUES (%s, %s, 1, CURRENT_TIMESTAMP)
            ON CONFLICT (chunk_id)
            DO UPDATE SET 
                retry_count = edi.retry_queue.retry_count + 1,
                error_message = %s,
                last_retry_date = CURRENT_TIMESTAMP
            """
            self.db_handler.execute_query(query, (chunk_id, error_message, error_message))
            
        except Exception as e:
            self.logger.error(f"Error adding chunk {chunk_id} to retry queue: {str(e)}")
    
    def get_retry_chunks(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Get chunks that need to be retried."""
        try:
            query = """
            SELECT chunk_id, error_message, retry_count
            FROM edi.retry_queue
            WHERE retry_count < %s AND resolved = false
            ORDER BY created_date
            """
            return self.db_handler.execute_query(query, (max_retries,))
            
        except Exception as e:
            self.logger.error(f"Error getting retry chunks: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear parser cache."""
        self._claim_count_cache = None
        self._cache_timestamp = None
        if self.db_handler:
            self.db_handler.clear_cache()