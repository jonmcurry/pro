### src/monitoring/metrics.py
"""
Prometheus Metrics Collection
"""
import logging
import time
from prometheus_client import Counter, Histogram, Gauge, Info


class MetricsCollector:
    """Centralized metrics collection for Prometheus."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.claims_processed_total = Counter(
            'claims_processed_total',
            'Total number of claims processed'
        )
        
        self.errors_total = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        self.validation_results_total = Counter(
            'validation_results_total',
            'Total validation results',
            ['status']
        )
        
        # Histograms
        self.processing_duration = Histogram(
            'processing_duration_seconds',
            'Time spent processing claims',
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600]
        )
        
        self.claim_validation_duration = Histogram(
            'claim_validation_duration_seconds',
            'Time spent validating individual claims',
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]
        )
        
        # Gauges
        self.active_workers = Gauge(
            'active_workers',
            'Number of active worker threads'
        )
        
        self.memory_usage_percent = Gauge(
            'memory_usage_percent',
            'Memory usage percentage'
        )
        
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage' 
        )
        
        self.database_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database_type']
        )
        
        self.processing_rate = Gauge(
            'processing_rate_claims_per_hour',
            'Claims processing rate'
        )
        
        # Info metrics
        self.system_info = Info(
            'system_info',
            'System information'
        )
    
    def increment_claims_processed(self, count: int = 1):
        """Increment claims processed counter."""
        self.claims_processed_total.inc(count)
    
    def increment_error_count(self, error_type: str = 'general'):
        """Increment error counter."""
        self.errors_total.labels(error_type=error_type).inc()
    
    def increment_validation_result(self, status: str):
        """Increment validation result counter."""
        self.validation_results_total.labels(status=status).inc()
    
    def set_processing_duration(self, duration: float):
        """Record processing duration."""
        self.processing_duration.observe(duration)
    
    def set_claim_validation_duration(self, duration: float):
        """Record claim validation duration."""
        self.claim_validation_duration.observe(duration)
    
    def set_active_workers(self, count: int):
        """Set active workers gauge."""
        self.active_workers.set(count)
    
    def set_memory_usage(self, percent: float):
        """Set memory usage gauge."""
        self.memory_usage_percent.set(percent)
    
    def set_cpu_usage(self, percent: float):
        """Set CPU usage gauge."""
        self.cpu_usage_percent.set(percent)
    
    def set_database_connections(self, count: int, db_type: str):
        """Set database connections gauge."""
        self.database_connections_active.labels(database_type=db_type).set(count)
    
    def set_processing_rate(self, rate: float):
        """Set processing rate gauge."""
        self.processing_rate.set(rate)
    
    def set_system_info(self, info_dict: dict):
        """Set system information."""
        self.system_info.info(info_dict)


# Global metrics instance
system_metrics = MetricsCollector()