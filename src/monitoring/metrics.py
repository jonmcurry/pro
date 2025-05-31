"""
Fixed MetricsCollector with singleton pattern and proper registry management
"""
import logging
import threading
from typing import Optional, Dict, Any
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
from prometheus_client.metrics import MetricWrapperBase


class MetricsCollector:
    """
    Singleton MetricsCollector to prevent duplicate metric registration.
    """
    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
            
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._metrics_registry = None
        
        # Initialize metrics safely
        self._initialize_metrics()
        self._initialized = True
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics with duplicate detection."""
        try:
            # Create a custom registry or use the default one
            # Check if we should use a custom registry
            self._metrics_registry = REGISTRY
            
            # Clear any existing metrics with our names first
            self._clear_existing_metrics()
            
            # Initialize counters
            self.claims_processed_total = Counter(
                'edi_claims_processed_total',
                'Total number of EDI claims processed',
                registry=self._metrics_registry
            )
            
            self.claims_validation_errors_total = Counter(
                'edi_claims_validation_errors_total',
                'Total number of claim validation errors',
                registry=self._metrics_registry
            )
            
            self.claims_storage_errors_total = Counter(
                'edi_claims_storage_errors_total', 
                'Total number of claim storage errors',
                registry=self._metrics_registry
            )
            
            self._total_processing_duration_gauge = Gauge(
                'edi_total_processing_duration_seconds',
                'Total processing duration for the current session',
                registry=self._metrics_registry
            )
            
            # Initialize histograms for timing
            self.claim_processing_duration = Histogram(
                'edi_claim_processing_duration_seconds',
                'Time spent processing individual claims',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
                registry=self._metrics_registry
            )
            
            self.validation_duration = Histogram(
                'edi_validation_duration_seconds',
                'Time spent validating individual claims',
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self._metrics_registry
            )
            
            # Initialize gauges for current state
            self.active_workers = Gauge(
                'edi_active_workers',
                'Number of active processing workers',
                registry=self._metrics_registry
            )
            
            self.memory_usage_percent = Gauge(
                'edi_memory_usage_percent',
                'Current memory usage percentage',
                registry=self._metrics_registry
            )
            
            self.cpu_usage_percent = Gauge(
                'edi_cpu_usage_percent',
                'Current CPU usage percentage', 
                registry=self._metrics_registry
            )
            
            self.processing_rate = Gauge(
                'edi_processing_rate_per_hour',
                'Current processing rate in claims per hour',
                registry=self._metrics_registry
            )
            
            # Batch processing metrics
            self.batch_size = Histogram(
                'edi_batch_size',
                'Size of processing batches',
                buckets=[10, 50, 100, 250, 500, 1000, 2000],
                registry=self._metrics_registry
            )
            
            self.batch_processing_duration = Histogram(
                'edi_batch_processing_duration_seconds',
                'Time spent processing batches',
                buckets=[1, 5, 10, 30, 60, 120, 300],
                registry=self._metrics_registry
            )
            
            self.logger.info("Metrics initialized successfully")
            
        except ValueError as e:
            if "Duplicated timeseries" in str(e):
                self.logger.error(f"Duplicate metrics detected: {str(e)}")
                # Try to use existing metrics instead of creating new ones
                self._use_existing_metrics()
            else:
                raise
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics: {str(e)}")
            raise
    
    def _clear_existing_metrics(self):
        """Clear any existing metrics with our names from the registry."""
        try:
            # Get all metric names we're about to create
            our_metrics = [
                'edi_claims_processed_total',
                'edi_claims_validation_errors_total', 
                'edi_claims_storage_errors_total',
                'edi_claim_processing_duration_seconds',
                'edi_validation_duration_seconds',
                'edi_active_workers',
                'edi_memory_usage_percent',
                'edi_cpu_usage_percent',
                'edi_processing_rate_per_hour',
                'edi_batch_size',
                'edi_batch_processing_duration_seconds',
                'edi_total_processing_duration_seconds' # Added new metric here
            ]
            
            # Remove conflicting metrics from registry
            collectors_to_remove = []
            for collector in list(self._metrics_registry._collector_to_names.keys()):
                metric_names = self._metrics_registry._collector_to_names.get(collector, set())
                if any(name in our_metrics for name in metric_names):
                    collectors_to_remove.append(collector)
            
            for collector in collectors_to_remove:
                try:
                    self._metrics_registry.unregister(collector)
                    self.logger.debug(f"Removed existing collector: {collector}")
                except KeyError:
                    # Already removed
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Error clearing existing metrics: {str(e)}")
    
    def _use_existing_metrics(self):
        """Use existing metrics if they're already registered."""
        self.logger.warning(
            "Attempting to use existing metrics due to registry conflicts. "
            "This is a fallback and may indicate issues with metric cleanup or multiple initializations."
        )
        
        # Try to find existing metrics in the registry
        for collector in self._metrics_registry._collector_to_names.keys():
            if hasattr(collector, '_name'):
                if 'claims_processed' in collector._name:
                    self.claims_processed_total = collector
                elif 'validation_errors' in collector._name:
                    self.claims_validation_errors_total = collector
                elif 'storage_errors' in collector._name:
                    self.claims_storage_errors_total = collector
                # Note: This approach is fragile as `collector` is the raw collector object,
                # not necessarily the Counter/Gauge/Histogram wrapper.
                # This method might not correctly re-assign the high-level metric attributes
                # (self.claims_processed_total, etc.) to functional Prometheus metric objects.
                # Proper cleanup in _clear_existing_metrics and ensuring true singleton behavior
                # is preferred.

        self.logger.warning("_use_existing_metrics might not fully restore metric functionality.")
    
    # Metric update methods
    def increment_claims_processed(self, count: int = 1):
        """Increment the claims processed counter."""
        try:
            self.claims_processed_total.inc(count)
        except Exception as e:
            self.logger.error(f"Error incrementing claims processed: {str(e)}")
    
    def increment_validation_errors(self, count: int = 1):
        """Increment the validation errors counter."""
        try:
            self.claims_validation_errors_total.inc(count)
        except Exception as e:
            self.logger.error(f"Error incrementing validation errors: {str(e)}")
    
    def increment_storage_errors(self, count: int = 1):
        """Increment the storage errors counter."""
        try:
            self.claims_storage_errors_total.inc(count)
        except Exception as e:
            self.logger.error(f"Error incrementing storage errors: {str(e)}")
    
    def record_processing_duration(self, duration: float):
        """Record claim processing duration."""
        try:
            self.claim_processing_duration.observe(duration)
        except Exception as e:
            self.logger.error(f"Error recording processing duration: {str(e)}")
    
    def record_validation_duration(self, duration: float):
        """Record validation duration."""
        try:
            self.validation_duration.observe(duration)
        except Exception as e:
            self.logger.error(f"Error recording validation duration: {str(e)}")
    
    def set_active_workers(self, count: int):
        """Set the number of active workers."""
        try:
            self.active_workers.set(count)
        except Exception as e:
            self.logger.error(f"Error setting active workers: {str(e)}")
    
    def set_memory_usage(self, percent: float):
        """Set current memory usage percentage."""
        try:
            self.memory_usage_percent.set(percent)
        except Exception as e:
            self.logger.error(f"Error setting memory usage: {str(e)}")
    
    def set_cpu_usage(self, percent: float):
        """Set current CPU usage percentage."""
        try:
            self.cpu_usage_percent.set(percent)
        except Exception as e:
            self.logger.error(f"Error setting CPU usage: {str(e)}")
    
    def set_processing_rate(self, rate: float):
        """Set current processing rate."""
        try:
            self.processing_rate.set(rate)
        except Exception as e:
            self.logger.error(f"Error setting processing rate: {str(e)}")
    
    def record_batch_size(self, size: int):
        """Record batch size."""
        try:
            self.batch_size.observe(size)
        except Exception as e:
            self.logger.error(f"Error recording batch size: {str(e)}")
    
    def record_batch_duration(self, duration: float):
        """Record batch processing duration."""
        try:
            self.batch_processing_duration.observe(duration)
        except Exception as e:
            self.logger.error(f"Error recording batch duration: {str(e)}")
    
    # Legacy method compatibility
    def increment_error_count(self, count: int = 1):
        """Legacy method for incrementing error count."""
        self.increment_validation_errors(count)
    
    def set_processing_duration(self, duration: float):
        """Legacy method for setting processing duration."""
        # This was used for total duration, now we'll use it as a gauge
        # Assumes _total_processing_duration_gauge is initialized in _initialize_metrics
        try:
            self._total_processing_duration_gauge.set(duration)
        except Exception as e:
            self.logger.error(f"Error setting processing duration: {str(e)}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            # Accessing ._value is an internal detail of prometheus-client
            # and might break in future versions. Use with caution.
            return {
                'claims_processed': self.claims_processed_total._value,
                'validation_errors': self.claims_validation_errors_total._value,
                'storage_errors': self.claims_storage_errors_total._value,
                'active_workers': self.active_workers._value,
                'memory_usage_percent': self.memory_usage_percent._value,
                'cpu_usage_percent': self.cpu_usage_percent._value,
                'processing_rate': self.processing_rate._value,
                'total_processing_duration_seconds': self._total_processing_duration_gauge._value
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {str(e)}")
            return {}
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance - useful for testing."""
        with cls._lock:
            if cls._instance is not None:
                # Clean up existing metrics
                if hasattr(cls._instance, '_metrics_registry'):
                    try:
                        # Clear our metrics from registry
                        cls._instance._clear_existing_metrics()
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Error during cleanup: {e}")
                
                cls._instance = None