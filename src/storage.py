### src/storage.py
"""
OPTIMIZED Result Storage - Enhanced bulk operations and performance
"""
import logging
from typing import Dict, List, Any, Optional
import time
from contextlib import contextmanager
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from database.sqlserver_handler import SQLServerHandler


@dataclass
class StorageMetrics:
    """Storage operation metrics."""
    total_results: int = 0
    successful_inserts: int = 0
    failed_inserts: int = 0
    processing_time: float = 0.0
    average_batch_time: float = 0.0


class ResultStorage:
    """OPTIMIZED result storage with enhanced bulk operations and async processing."""
    
    def __init__(self, db_handler: SQLServerHandler, config: Dict[str, Any]):
        self.db_handler = db_handler
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Enhanced bulk insert settings
        self.batch_size = config.get('batch_size', 1000)
        self.max_retries = config.get('max_retries', 3)
        self.max_workers = config.get('storage_workers', 2)
        
        # Threading
        self._storage_lock = threading.Lock()
        
        # Metrics
        self.metrics = StorageMetrics()
        
        # Buffer for async operations
        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._buffer_size_limit = self.batch_size * 2
        
    def store_results(self, validation_results: List[Dict[str, Any]]) -> bool:
        """
        OPTIMIZED: Store validation results with enhanced bulk operations.
        """
        if not validation_results:
            return True
            
        start_time = time.time()
        
        try:
            # Use the optimized bulk storage method from SQLServerHandler
            success = self.db_handler.store_validation_results_bulk(validation_results)
            
            # Update metrics
            self.metrics.total_results += len(validation_results)
            if success:
                self.metrics.successful_inserts += len(validation_results)
            else:
                self.metrics.failed_inserts += len(validation_results)
            
            self.metrics.processing_time += time.time() - start_time
            
            self.logger.debug(f"Stored {len(validation_results)} validation results")
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing results: {str(e)}")
            self.metrics.failed_inserts += len(validation_results)
            return False
    
    def store_results_async(self, validation_results: List[Dict[str, Any]]) -> bool:
        """
        OPTIMIZED: Async storage with buffering for high-throughput scenarios.
        """
        if not validation_results:
            return True
        
        with self._buffer_lock:
            self._buffer.extend(validation_results)
            
            # If buffer is full, flush it
            if len(self._buffer) >= self._buffer_size_limit:
                return self._flush_buffer()
        
        return True
    
    def _flush_buffer(self) -> bool:
        """Flush buffered results to database."""
        if not self._buffer:
            return True
            
        try:
            with self._buffer_lock:
                buffer_copy = self._buffer.copy()
                self._buffer.clear()
            
            return self.store_results(buffer_copy)
            
        except Exception as e:
            self.logger.error(f"Error flushing buffer: {str(e)}")
            return False
    
    def store_results_parallel(self, validation_results: List[Dict[str, Any]]) -> bool:
        """
        OPTIMIZED: Parallel storage for very large result sets.
        """
        if not validation_results:
            return True
        
        if len(validation_results) < self.batch_size:
            return self.store_results(validation_results)
        
        try:
            # Split into chunks for parallel processing
            chunks = [
                validation_results[i:i + self.batch_size]
                for i in range(0, len(validation_results), self.batch_size)
            ]
            
            success_count = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.store_results, chunk) for chunk in chunks]
                
                for future in futures:
                    if future.result():
                        success_count += 1
            
            return success_count == len(chunks)
            
        except Exception as e:
            self.logger.error(f"Error in parallel storage: {str(e)}")
            return False
    
    def store_with_retry(self, validation_results: List[Dict[str, Any]]) -> bool:
        """
        OPTIMIZED: Store with exponential backoff retry logic.
        """
        for attempt in range(self.max_retries):
            try:
                if self.store_results(validation_results):
                    return True
                    
            except Exception as e:
                self.logger.warning(f"Storage attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 1  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    self.logger.error("All storage retry attempts failed")
                    return False
        
        return False
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage performance statistics."""
        avg_batch_time = 0
        if self.metrics.successful_inserts > 0:
            avg_batch_time = self.metrics.processing_time / (self.metrics.successful_inserts / self.batch_size)
        
        return {
            'total_results_processed': self.metrics.total_results,
            'successful_inserts': self.metrics.successful_inserts,
            'failed_inserts': self.metrics.failed_inserts,
            'success_rate': (self.metrics.successful_inserts / self.metrics.total_results * 100) if self.metrics.total_results > 0 else 0,
            'total_processing_time': self.metrics.processing_time,
            'average_batch_time': avg_batch_time,
            'throughput_per_second': self.metrics.successful_inserts / self.metrics.processing_time if self.metrics.processing_time > 0 else 0
        }
    
    def cleanup_old_results(self, days_to_keep: int = 90) -> bool:
        """
        OPTIMIZED: Cleanup old results with archiving.
        """
        try:
            deleted_count = self.db_handler.cleanup_old_results(days_to_keep)
            self.logger.info(f"Cleaned up {deleted_count} old validation results")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return False
    
    def get_validation_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get validation summary statistics."""
        try:
            return self.db_handler.get_validation_statistics(days)
            
        except Exception as e:
            self.logger.error(f"Error getting validation summary: {str(e)}")
            return {}
    
    def flush_all_buffers(self) -> bool:
        """Flush all buffered data before shutdown."""
        try:
            return self._flush_buffer()
        except Exception as e:
            self.logger.error(f"Error flushing buffers: {str(e)}")
            return False
    
    def reset_metrics(self):
        """Reset storage metrics."""
        self.metrics = StorageMetrics()


class ResultStorageManager:
    """
    OPTIMIZED: High-level storage manager with multiple storage strategies.
    """
    
    def __init__(self, db_handler: SQLServerHandler, config: Dict[str, Any]):
        self.storage = ResultStorage(db_handler, config)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage strategy
        self.strategy = config.get('storage_strategy', 'bulk')  # bulk, async, parallel
        
    def store_validation_results(self, validation_results: List[Dict[str, Any]]) -> bool:
        """Store results using configured strategy."""
        if not validation_results:
            return True
        
        strategy_map = {
            'bulk': self.storage.store_results,
            'async': self.storage.store_results_async,
            'parallel': self.storage.store_results_parallel,
            'retry': self.storage.store_with_retry
        }
        
        storage_method = strategy_map.get(self.strategy, self.storage.store_results)
        
        try:
            return storage_method(validation_results)
        except Exception as e:
            self.logger.error(f"Storage failed with strategy {self.strategy}: {str(e)}")
            # Fallback to basic storage
            return self.storage.store_results(validation_results)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        storage_stats = self.storage.get_storage_statistics()
        validation_summary = self.storage.get_validation_summary()
        
        return {
            'storage_performance': storage_stats,
            'validation_summary': validation_summary,
            'configuration': {
                'strategy': self.strategy,
                'batch_size': self.storage.batch_size,
                'max_workers': self.storage.max_workers
            }
        }
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """
        OPTIMIZED: Auto-tune storage configuration based on performance metrics.
        """
        stats = self.storage.get_storage_statistics()
        recommendations = {}
        
        # Analyze throughput
        current_throughput = stats.get('throughput_per_second', 0)
        
        if current_throughput < 100:  # Low throughput
            recommendations['batch_size'] = min(self.storage.batch_size * 2, 2000)
            recommendations['strategy'] = 'parallel'
        elif current_throughput > 1000:  # High throughput
            recommendations['strategy'] = 'async'
        
        # Analyze error rate
        error_rate = (stats.get('failed_inserts', 0) / max(stats.get('total_results_processed', 1), 1)) * 100
        
        if error_rate > 5:  # High error rate
            recommendations['strategy'] = 'retry'
            recommendations['max_retries'] = 5
        
        return recommendations
    
    def shutdown(self) -> bool:
        """Graceful shutdown with buffer flushing."""
        try:
            self.logger.info("Shutting down storage manager...")
            success = self.storage.flush_all_buffers()
            
            # Log final statistics
            final_stats = self.storage.get_storage_statistics()
            self.logger.info(f"Final storage statistics: {final_stats}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during storage shutdown: {str(e)}")
            return False