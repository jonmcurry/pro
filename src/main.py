### src/main.py
"""
OPTIMIZED Main EDI Processing System - Enhanced performance and batch processing
"""
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import psutil
import gc
import asyncio
from contextlib import asynccontextmanager

from parser import ClaimParser
from validator import ClaimValidator
from storage import ResultStorageManager
from monitoring.metrics import MetricsCollector
from monitoring.notifications import EmailNotifier
from utils.resource_monitor import ResourceOptimizer
from database.postgresql_handler import PostgreSQLHandler
from database.sqlserver_handler import SQLServerHandler
from prometheus_client import start_http_server


class OptimizedProcessingOrchestrator:
    """OPTIMIZED: Enhanced processing orchestrator with batch operations and performance monitoring."""
    
    def __init__(self, parser, validator, storage, metrics, config):
        self.parser = parser
        self.validator = validator
        self.storage = storage
        self.metrics = metrics
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Enhanced processing parameters
        self.chunk_size = config.get('chunk_size', 500)
        self.max_workers = config.get('max_workers', 4)
        self.memory_limit = config.get('memory_limit_percent', 70)
        self.batch_validation_size = config.get('batch_validation_size', 100)
        
        # Performance optimization
        self.enable_batch_processing = config.get('enable_batch_processing', True)
        self.adaptive_chunk_sizing = config.get('adaptive_chunk_sizing', True)
        
        # Threading and async support
        self.processing_semaphore = threading.Semaphore(self.max_workers)
        
    def process_claims(self):
        """OPTIMIZED: Process all claims with enhanced parallel and batch processing."""
        start_time = time.time()
        total_processed = 0
        total_errors = 0
        
        try:
            # Get total claim count for progress tracking
            total_claims = self.parser.get_total_claim_count()
            self.logger.info(f"Processing {total_claims} total claims with optimizations enabled")
            
            if total_claims == 0:
                self.logger.warning("No claims found to process")
                return
            
            # Adaptive chunk sizing based on system resources
            if self.adaptive_chunk_sizing:
                self.chunk_size = self._calculate_adaptive_chunk_size()
            
            # Process in optimized chunks
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                chunk_futures_map = {}
                
                for chunk_id, chunk_data in self.parser.get_claim_chunks(self.chunk_size):
                    # Check resource usage before submitting new work
                    if self._check_resource_limits():
                        future = executor.submit(self._process_chunk_optimized, chunk_id, chunk_data)
                        futures.append(future)
                        chunk_futures_map[future] = chunk_id
                    else:
                        self.logger.warning("Resource limits reached, waiting for completion")
                        # Wait for some tasks to complete before submitting new ones
                        self._wait_for_partial_completion(futures, chunk_futures_map)
                        
                        future = executor.submit(self._process_chunk_optimized, chunk_id, chunk_data)
                        futures.append(future)
                        chunk_futures_map[future] = chunk_id
                
                # Process completed futures as they finish
                for future in as_completed(futures):
                    try:
                        chunk_result = future.result()
                        chunk_id = chunk_futures_map[future]
                        
                        total_processed += chunk_result['processed_count']
                        total_errors += chunk_result.get('error_count', 0)
                        
                        # Update metrics
                        self.metrics.increment_claims_processed(chunk_result['processed_count'])
                        if chunk_result.get('error_count', 0) > 0:
                            self.metrics.increment_error_count()
                        
                        # Mark chunk as processed with statistics
                        self.parser.mark_chunk_processed(
                            chunk_id, 
                            chunk_result['processed_count'],
                            chunk_result['duration']
                        )
                        
                        # Log progress
                        progress = (total_processed / total_claims) * 100 if total_claims > 0 else 0
                        self.logger.info(
                            f"Progress: {progress:.1f}% ({total_processed}/{total_claims}) "
                            f"Errors: {total_errors}"
                        )
                        
                    except Exception as e:
                        chunk_id = chunk_futures_map.get(future, 'unknown')
                        self.logger.error(f"Chunk {chunk_id} processing error: {str(e)}", exc_info=True)
                        self.metrics.increment_error_count()
                        total_errors += 1
            
            # Final statistics and cleanup
            duration = time.time() - start_time
            rate = total_processed / (duration / 3600) if duration > 0 else 0
            
            self.logger.info(f"Processing completed: {total_processed} claims in {duration:.2f} seconds")
            self.logger.info(f"Processing rate: {rate:.0f} claims/hour")
            self.logger.info(f"Total errors: {total_errors}")
            
            # Update final metrics
            self.metrics.set_processing_duration(duration)
            self.metrics.set_processing_rate(rate)
            
            # Flush any remaining buffered data
            self.storage.shutdown()
            
        except Exception as e:
            self.logger.error(f"Processing orchestration error: {str(e)}", exc_info=True)
            raise
    
    def _process_chunk_optimized(self, chunk_id: int, chunk_data: List[Dict]) -> Dict[str, Any]:
        """OPTIMIZED: Process a single chunk with batch validation and enhanced error handling."""
        start_time = time.time()
        processed_count = 0
        error_count = 0
        
        try:
            with self.processing_semaphore:
                self.logger.debug(f"Processing chunk {chunk_id} with {len(chunk_data)} claims")
                
                if self.enable_batch_processing and len(chunk_data) >= self.batch_validation_size:
                    # Use batch validation for better performance
                    validation_results = self._validate_claims_batch(chunk_data)
                    processed_count = len(chunk_data)
                else:
                    # Fall back to individual validation
                    validation_results = []
                    for claim in chunk_data:
                        try:
                            result = self.validator.validate_claim(claim)
                            validation_results.append(result)
                            processed_count += 1
                            
                        except Exception as e:
                            claim_id = claim.get('claim_id', 'unknown')
                            self.logger.error(f"Claim validation error for {claim_id}: {str(e)}")
                            error_count += 1
                            
                            # Create error result
                            error_result = {
                                'claim_id': claim_id,
                                'validation_status': 'ERROR',
                                'error_message': str(e),
                                'processing_time': 0
                            }
                            validation_results.append(error_result)
                
                # Store results efficiently
                if validation_results:
                    storage_success = self.storage.store_validation_results(validation_results)
                    if not storage_success:
                        self.logger.error(f"Failed to store results for chunk {chunk_id}")
                        error_count += len(validation_results)
                
                # Update claim processing status in bulk
                claim_ids = [claim.get('claim_id') for claim in chunk_data if claim.get('claim_id')]
                if claim_ids:
                    self.parser.mark_claims_processed_bulk(claim_ids)
                
                duration = time.time() - start_time
                self.logger.debug(f"Chunk {chunk_id} completed in {duration:.2f} seconds")
                
                return {
                    'chunk_id': chunk_id,
                    'processed_count': processed_count,
                    'error_count': error_count,
                    'duration': duration,
                    'claims_per_second': processed_count / duration if duration > 0 else 0
                }
                
        except Exception as e:
            self.logger.error(f"Chunk {chunk_id} processing error: {str(e)}", exc_info=True)
            # Add to retry queue
            self.parser.add_to_retry_queue(chunk_id, str(e))
            raise
    
    def _validate_claims_batch(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use batch validation for improved performance."""
        try:
            return self.validator.validate_claims_batch(claims)
        except Exception as e:
            self.logger.error(f"Batch validation failed, falling back to individual: {str(e)}")
            # Fallback to individual validation
            results = []
            for claim in claims:
                try:
                    result = self.validator.validate_claim(claim)
                    results.append(result)
                except Exception as claim_error:
                    results.append({
                        'claim_id': claim.get('claim_id'),
                        'validation_status': 'ERROR',
                        'error_message': str(claim_error),
                        'processing_time': 0
                    })
            return results
    
    def _calculate_adaptive_chunk_size(self) -> int:
        """Calculate adaptive chunk size based on system performance."""
        try:
            # Base chunk size on available memory and CPU
            memory_percent = psutil.virtual_memory().percent
            cpu_count = psutil.cpu_count()
            
            base_size = self.chunk_size
            
            # Adjust based on memory usage
            if memory_percent < 50:
                base_size = int(base_size * 1.5)
            elif memory_percent > 80:
                base_size = int(base_size * 0.7)
            
            # Adjust based on CPU cores
            cpu_factor = max(1, cpu_count / 4)
            adaptive_size = int(base_size * cpu_factor)
            
            # Bounds checking
            adaptive_size = max(100, min(adaptive_size, 2000))
            
            self.logger.info(f"Adaptive chunk size calculated: {adaptive_size}")
            return adaptive_size
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive chunk size: {str(e)}")
            return self.chunk_size
    
    def _wait_for_partial_completion(self, futures, chunk_futures_map):
        """Wait for partial completion to free up resources."""
        completed_count = 0
        target_completion = len(futures) // 2
        
        for future in as_completed(futures):
            if completed_count >= target_completion:
                break
            completed_count += 1
            
            # Process the completed future
            try:
                chunk_result = future.result()
                chunk_id = chunk_futures_map[future]
                self.logger.debug(f"Chunk {chunk_id} completed during resource wait")
            except Exception as e:
                self.logger.error(f"Error during partial completion wait: {str(e)}")
    
    def _check_resource_limits(self) -> bool:
        """Enhanced resource limit checking."""
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if memory_percent > self.memory_limit:
            self.logger.warning(f"Memory usage high: {memory_percent}%")
            gc.collect()  # Force garbage collection
            return False
        
        if cpu_percent > 90:  # High CPU usage
            self.logger.warning(f"CPU usage high: {cpu_percent}%")
            time.sleep(0.1)  # Brief pause
            return False
            
        return True


class EDIProcessingSystem:
    """OPTIMIZED: Main system controller with enhanced initialization and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with enhanced configuration
        self.metrics = MetricsCollector()
        self.email_notifier = EmailNotifier(config.get('email', {}))
        self.resource_optimizer = ResourceOptimizer(config.get('processing', {}))
        
        # Initialize databases with connection pooling
        self.postgres_handler = PostgreSQLHandler(config['database']['postgresql'])
        self.sqlserver_handler = SQLServerHandler(config['database']['sqlserver'])
        
        # Initialize processing components with optimizations
        self.parser = ClaimParser(config.get('parsing', {}))
        self.parser.initialize_database(self.postgres_handler)
        
        self.validator = ClaimValidator(
            self.postgres_handler,
            config.get('validation', {})
        )
        
        # Use enhanced storage manager
        self.storage = ResultStorageManager(
            self.sqlserver_handler,
            config.get('storage', {})
        )
        
        # Enhanced processing orchestrator
        self.orchestrator = OptimizedProcessingOrchestrator(
            self.parser,
            self.validator,
            self.storage,
            self.metrics,
            config.get('processing', {})
        )
        
    def run(self):
        """OPTIMIZED: Main processing loop with enhanced monitoring and error handling."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting optimized EDI Claims Processing System")
            
            # Enhanced system initialization
            if not self._initialize_system():
                raise Exception("System initialization failed")
            
            # Start enhanced monitoring services
            monitoring_service = EnhancedMonitoringService(
                self.config.get('monitoring', {}),
                self.metrics,
                self.email_notifier,
                self.resource_optimizer
            )
            monitoring_service.start()
            
            try:
                # Pre-processing optimizations
                self._optimize_system_for_processing()
                
                # Run main processing with performance tracking
                processing_start = time.time()
                self.orchestrator.process_claims()
                processing_duration = time.time() - processing_start
                
                # Post-processing cleanup and reporting
                self._post_processing_cleanup()
                
                # Generate performance report
                performance_report = self._generate_performance_report(processing_duration)
                self.logger.info(f"Performance Report: {performance_report}")
                
                # Send completion notification
                self.email_notifier.send_completion_notification(performance_report)
                
            finally:
                # Stop monitoring
                monitoring_service.stop()
                
        except Exception as e:
            total_duration = time.time() - start_time
            self.logger.error(f"System error after {total_duration:.2f}s: {str(e)}", exc_info=True)
            self.email_notifier.send_error_notification(
                "System Error",
                f"EDI Processing System encountered an error: {str(e)}"
            )
            raise
        finally:
            # Ensure cleanup
            self._cleanup_resources()
    
    def _initialize_system(self) -> bool:
        """Enhanced system initialization with comprehensive testing."""
        try:
            self.logger.info("Performing enhanced system initialization...")
            
            # Test database connections with retry
            if not self._test_database_connections_with_retry():
                return False
            
            # Initialize resource monitoring
            self.resource_optimizer.start_monitoring()
            
            # Warm up caches
            self._warm_up_caches()
            
            # Validate ML models
            self._validate_ml_components()
            
            self.logger.info("Enhanced system initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization error: {str(e)}")
            return False
    
    def _test_database_connections_with_retry(self, max_retries: int = 3) -> bool:
        """Test database connections with retry logic."""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Testing database connections (attempt {attempt + 1}/{max_retries})")
                
                # Test PostgreSQL
                if not self.postgres_handler.test_connection():
                    self.logger.error("PostgreSQL connection test failed")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False
                
                # Test SQL Server
                if not self.sqlserver_handler.test_connection():
                    self.logger.error("SQL Server connection test failed")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False
                
                self.logger.info("Database connections verified successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Database connection test error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False
        
        return False
    
    def _warm_up_caches(self):
        """Warm up system caches for better performance."""
        try:
            self.logger.info("Warming up system caches...")
            
            # Warm up parser cache
            self.parser.get_total_claim_count()
            
            # Warm up validator caches
            if hasattr(self.validator, 'reload_components'):
                self.validator.reload_components()
            
            self.logger.info("Cache warm-up completed")
            
        except Exception as e:
            self.logger.warning(f"Cache warm-up failed: {str(e)}")
    
    def _validate_ml_components(self):
        """Validate ML model availability and performance."""
        try:
            self.logger.info("Validating ML components...")
            
            # Check if ML model is available
            if hasattr(self.validator, 'model_cache'):
                model_available = self.validator.model_cache.ml_model is not None
                self.logger.info(f"ML model available: {model_available}")
            
        except Exception as e:
            self.logger.warning(f"ML component validation failed: {str(e)}")
    
    def _optimize_system_for_processing(self):
        """Pre-processing system optimizations."""
        try:
            self.logger.info("Applying pre-processing optimizations...")
            
            # Clear caches if memory is low
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                gc.collect()
                if hasattr(self.postgres_handler, 'clear_cache'):
                    self.postgres_handler.clear_cache()
            
            # Set process priority
            try:
                psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS') else -5)
            except:
                pass  # Not critical if this fails
            
        except Exception as e:
            self.logger.warning(f"Pre-processing optimization failed: {str(e)}")
    
    def _post_processing_cleanup(self):
        """Post-processing cleanup and maintenance."""
        try:
            self.logger.info("Performing post-processing cleanup...")
            
            # Force garbage collection
            gc.collect()
            
            # Cleanup old validation results if configured
            cleanup_days = self.config.get('storage', {}).get('cleanup_days', 90)
            if cleanup_days > 0:
                self.storage.storage.cleanup_old_results(cleanup_days)
            
        except Exception as e:
            self.logger.warning(f"Post-processing cleanup failed: {str(e)}")
    
    def _generate_performance_report(self, processing_duration: float) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Get storage statistics
            storage_stats = self.storage.get_performance_report()
            
            # Get database connection stats
            postgres_stats = self.postgres_handler.get_connection_stats()
            sqlserver_stats = self.sqlserver_handler.get_connection_stats()
            
            # Get system stats
            system_stats = self.resource_optimizer.get_system_stats()
            
            return {
                'processing_duration': processing_duration,
                'total_claims_processed': storage_stats.get('storage_performance', {}).get('successful_inserts', 0),
                'processing_rate': storage_stats.get('storage_performance', {}).get('throughput_per_second', 0),
                'error_rate': storage_stats.get('storage_performance', {}).get('success_rate', 0),
                'storage_performance': storage_stats,
                'database_stats': {
                    'postgresql': postgres_stats,
                    'sqlserver': sqlserver_stats
                },
                'system_performance': system_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}
    
    def _cleanup_resources(self):
        """Cleanup system resources."""
        try:
            self.logger.info("Cleaning up system resources...")
            
            # Stop resource monitoring
            if hasattr(self.resource_optimizer, 'stop_monitoring'):
                self.resource_optimizer.stop_monitoring()
            
            # Close database connections
            if hasattr(self.postgres_handler, 'close'):
                self.postgres_handler.close()
            
            if hasattr(self.sqlserver_handler, 'close'):
                self.sqlserver_handler.close()
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {str(e)}")


class EnhancedMonitoringService:
    """Enhanced monitoring service with comprehensive metrics collection."""
    
    def __init__(self, config: Dict[str, Any], metrics: MetricsCollector, 
                 email_notifier: EmailNotifier, resource_optimizer: ResourceOptimizer):
        self.config = config
        self.metrics = metrics
        self.email_notifier = email_notifier
        self.resource_optimizer = resource_optimizer
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.monitor_thread = None
        
        self.prometheus_port = config.get('prometheus_port', 8000)
        self.monitoring_interval = config.get('monitoring_interval', 30)
        
    def start(self):
        """Start enhanced monitoring services."""
        try:
            # Start Prometheus metrics server
            start_http_server(self.prometheus_port)
            self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            
            # Start enhanced resource monitoring
            self.running = True
            self.monitor_thread = threading.Thread(target=self._enhanced_monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # Set up resource event callbacks
            self.resource_optimizer.set_memory_warning_callback(self._handle_memory_warning)
            self.resource_optimizer.set_cpu_warning_callback(self._handle_cpu_warning)
            
        except Exception as e:
            self.logger.error(f"Failed to start enhanced monitoring: {str(e)}")
            
    def stop(self):
        """Stop monitoring services."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
            
    def _enhanced_monitoring_loop(self):
        """Enhanced monitoring loop with comprehensive metrics."""
        while self.running:
            try:
                # Collect system metrics
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                disk_percent = psutil.disk_usage('/').percent
                
                # Update Prometheus metrics
                self.metrics.set_memory_usage(memory_percent)
                self.metrics.set_cpu_usage(cpu_percent)
                
                # Check for alerts
                if memory_percent > 90:
                    self.email_notifier.send_error_notification(
                        "High Memory Usage",
                        f"System memory usage is at {memory_percent}%",
                        "CRITICAL"
                    )
                
                if cpu_percent > 95:
                    self.email_notifier.send_error_notification(
                        "High CPU Usage", 
                        f"System CPU usage is at {cpu_percent}%",
                        "CRITICAL"
                    )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Enhanced monitoring error: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _handle_memory_warning(self, memory_percent: float):
        """Handle memory pressure events."""
        self.logger.warning(f"Memory warning triggered: {memory_percent}%")
        
        # Force garbage collection
        gc.collect()
        
        # Send notification if critical
        if memory_percent > 95:
            self.email_notifier.send_error_notification(
                "Critical Memory Usage",
                f"System memory usage has reached critical level: {memory_percent}%",
                "CRITICAL"
            )
    
    def _handle_cpu_warning(self, cpu_percent: float):
        """Handle high CPU usage events."""
        self.logger.warning(f"CPU warning triggered: {cpu_percent}%")
        
        if cpu_percent > 98:
            self.email_notifier.send_error_notification(
                "Critical CPU Usage",
                f"System CPU usage has reached critical level: {cpu_percent}%",
                "CRITICAL"
            )