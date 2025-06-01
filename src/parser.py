### src/parser.py
"""
FIXED: Parser for PENDING claims only - Corrected resume logic
"""
import logging
from typing import Dict, List, Any, Tuple, Iterator, Optional
import json
import time
from database.postgresql_handler import PostgreSQLHandler


class ClaimParser:
    """Optimized claim parsing for PENDING claims only."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_handler = None
        
        # Caching
        self._claim_count_cache = None
        self._cache_timestamp = None
        self._cache_ttl = config.get('parser_cache_ttl_seconds', 300)  # Default 5 minutes
        
    def initialize_database(self, db_handler: PostgreSQLHandler):
        """Initialize database connection."""
        self.db_handler = db_handler
        
    def get_total_claim_count(self) -> int:
        """Get total number of PENDING claims with caching."""
        try:
            # Check cache
            current_time = time.time()
            if (self._claim_count_cache is not None and 
                self._cache_timestamp is not None and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._claim_count_cache
            
            # Query database for PENDING claims only
            query = """
            SELECT COUNT(*) as total_claims 
            FROM edi.claims 
            WHERE processing_status = 'PENDING'
            """
            result = self.db_handler.execute_query(query)
            count = result[0]['total_claims'] if result else 0
            
            # Update cache
            self._claim_count_cache = count
            self._cache_timestamp = current_time
            
            self.logger.info(f"Total PENDING claims: {count}")
            return count
            
        except Exception as e:
            self.logger.error(f"Error getting claim count: {str(e)}")
            return 0
    
    def get_claim_chunks(self, chunk_size: int) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """
        FIXED: Generate chunks for PENDING claims only, without resume gaps.
        """
        try:
            # Get total PENDING claims
            total_pending_claims = self.get_total_claim_count()
            expected_chunks = (total_pending_claims + chunk_size - 1) // chunk_size if total_pending_claims > 0 else 0
            
            self.logger.info(f"Starting chunk generation for PENDING claims only")
            self.logger.info(f"Total PENDING claims: {total_pending_claims}")
            self.logger.info(f"Expected chunks: {expected_chunks}")
            self.logger.info(f"Chunk size: {chunk_size}")
            
            # FIXED: Always start from offset 0 for PENDING claims
            # Don't use resume logic that can create gaps
            chunk_id = 1
            offset = 0 # Initialize offset for logging purposes
            # The offset for fetching PENDING claims should always be 0,
            # as the set of PENDING claims shrinks with each processed batch.
            chunks_generated = 0
            
            while True:
                self.logger.info(f"Fetching chunk {chunk_id} with offset {offset}, limit {chunk_size}")
                
                # Fetch PENDING claims only
                claims = self._fetch_pending_claims_batch(0, chunk_size) # Always fetch with offset 0
                
                self.logger.info(f"Chunk {chunk_id}: Retrieved {len(claims)} PENDING claims from database")
                
                if not claims:
                    self.logger.info(f"No more PENDING claims found at offset {offset}. Total chunks generated: {chunks_generated}")
                    break
                    
                # Process JSON claim data
                processed_claims = self._process_claim_data(claims)
                
                if not processed_claims:
                    self.logger.warning(f"Chunk {chunk_id}: No processed claims after data processing")
                    # Still increment to avoid infinite loop
                    chunk_id += 1 # Increment chunk_id for logging/tracking
                    continue
                
                chunks_generated += 1
                self.logger.info(f"Yielding chunk {chunk_id} with {len(processed_claims)} processed PENDING claims")
                yield chunk_id, processed_claims
                
                # Check if we've reached the end based on chunk size
                if len(claims) < chunk_size:
                    self.logger.info(f"Received {len(claims)} claims (< {chunk_size}), reaching end of PENDING claims")
                    break
                    
                chunk_id += 1

            self.logger.info(f"PENDING claims chunk generation complete.")
            self.logger.info(f"Generated {chunks_generated} chunks for {total_pending_claims} PENDING claims")
            
            if chunks_generated < expected_chunks:
                self.logger.warning(f"Generated fewer chunks than expected: {chunks_generated}/{expected_chunks}")
                
        except Exception as e:
            self.logger.error(f"Error generating claim chunks: {str(e)}", exc_info=True)
            raise
    
    def _fetch_pending_claims_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """FIXED: Fetch PENDING claims only with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Test connection before query
                if not self.db_handler.test_connection():
                    self.logger.warning(f"Database connection test failed on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise Exception("Database connection failed after all retries")
                
                return self.db_handler.get_pending_claims(limit, offset)
                
            except Exception as e:
                self.logger.error(f"Error fetching PENDING claims batch (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return []
    
    def _process_claim_data(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Process claim data efficiently with batch operations.
        """
        try:
            processed_claims = []
            
            for claim in claims:
                # Verify this is still a PENDING claim
                if claim.get('processing_status') != 'PENDING':
                    self.logger.debug(f"Skipping non-PENDING claim {claim.get('claim_id')}: status {claim.get('processing_status')}")
                    continue
                
                # Parse JSON claim data if present
                if claim.get('claim_data'):
                    try:
                        # Check if claim_data is already a dict or if it's a JSON string
                        if isinstance(claim['claim_data'], str):
                            parsed_data = json.loads(claim['claim_data'])
                            claim.update(parsed_data)
                        elif isinstance(claim['claim_data'], dict):
                            # Already parsed, just update
                            claim.update(claim['claim_data'])
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.warning(f"Invalid JSON in claim {claim.get('claim_id', 'unknown')}: {str(e)}")
                
                # Convert JSON arrays back to lists for easier processing
                # Ensure these are lists even if they come as other types
                claim['diagnoses'] = claim.get('diagnoses', []) if isinstance(claim.get('diagnoses'), list) else []
                claim['procedures'] = claim.get('procedures', []) if isinstance(claim.get('procedures'), list) else []
                
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
        OPTIMIZED: Bulk update claim processing status from PENDING to COMPLETED.
        """
        try:
            if claim_ids:
                updated_count = self.db_handler.mark_claims_processed(claim_ids)
                self.logger.info(f"Marked {updated_count} claims as COMPLETED (was PENDING)")
                return updated_count
            return 0
                
        except Exception as e:
            self.logger.error(f"Error marking claims as processed: {str(e)}")
            return 0
    
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
    
    def diagnose_processing_state(self):
        """DIAGNOSTIC: Check current processing state for PENDING claims."""
        try:
            # Check claim status distribution
            status_query = """
            SELECT processing_status, COUNT(*) as count
            FROM edi.claims 
            GROUP BY processing_status
            ORDER BY processing_status
            """
            status_results = self.db_handler.execute_query(status_query)
            
            self.logger.info("Current claim status distribution:")
            pending_count = 0
            for row in status_results:
                count = row['count']
                status = row['processing_status']
                self.logger.info(f"  {status}: {count}")
                if status == 'PENDING':
                    pending_count = count
            
            # Check processed chunks
            chunk_query = """
            SELECT status, COUNT(*) as count, 
                   COALESCE(SUM(claims_count), 0) as total_claims_in_chunks
            FROM edi.processed_chunks 
            GROUP BY status
            """
            chunk_results = self.db_handler.execute_query(chunk_query)
            
            self.logger.info("Processed chunks status:")
            for row in chunk_results:
                self.logger.info(f"  {row['status']}: {row['count']} chunks, {row['total_claims_in_chunks']} claims")
            
            # Calculate expected chunks for PENDING claims
            if pending_count > 0:
                chunk_size = self.config.get('chunk_size', 500)
                expected_chunks = (pending_count + chunk_size - 1) // chunk_size
                self.logger.info(f"Expected chunks for {pending_count} PENDING claims: {expected_chunks}")
                
        except Exception as e:
            self.logger.error(f"Error in diagnostics: {str(e)}")
    
    def reset_for_reprocessing(self):
        """Reset processed chunks to reprocess PENDING claims."""
        try:
            # Clear processed chunks table
            delete_chunks_query = "DELETE FROM edi.processed_chunks"
            self.db_handler.execute_query(delete_chunks_query)
            
            # Clear retry queue
            delete_retry_query = "DELETE FROM edi.retry_queue"
            self.db_handler.execute_query(delete_retry_query)
            
            # Clear cache
            self.clear_cache()
            
            self.logger.info("Reset completed: cleared processed_chunks and retry_queue tables")
            
        except Exception as e:
            self.logger.error(f"Error resetting for reprocessing: {str(e)}")
            raise