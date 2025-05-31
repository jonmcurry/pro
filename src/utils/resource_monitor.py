### src/utils/resource_monitor.py
"""
Resource Monitoring and Optimization
"""
import logging
import psutil
import threading
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
import gc


class ResourceOptimizer:
    """System resource monitoring and optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Resource thresholds
        self.memory_limit_percent = config.get('memory_limit_percent', 70)
        self.cpu_limit_percent = config.get('cpu_limit_percent', 80)
        
        # Monitoring
        self.monitoring_interval = config.get('monitoring_interval', 30)
        self.monitor_thread = None
        self.monitoring_active = False
        
        # Callbacks for resource events
        self.memory_warning_callback: Optional[Callable] = None
        self.cpu_warning_callback: Optional[Callable] = None

        # For periodic logging
        self._last_status_log_time: float = 0
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.memory_limit_percent:
                    self._handle_memory_pressure(memory_percent)
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > self.cpu_limit_percent:
                    self._handle_cpu_pressure(cpu_percent)
                
                # Log resource usage periodically
                current_time = time.time()
                if current_time - self._last_status_log_time >= 300:  # Log every 5 minutes (300 seconds)
                    self._log_resource_status(memory_percent, cpu_percent)
                    self._last_status_log_time = current_time
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _handle_memory_pressure(self, memory_percent: float):
        """Handle high memory usage."""
        self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
        
        # Force garbage collection
        collected = gc.collect()
        self.logger.info(f"Garbage collection freed {collected} objects")
        
        # Call callback if registered
        if self.memory_warning_callback:
            try:
                self.memory_warning_callback(memory_percent)
            except Exception as e:
                self.logger.error(f"Memory warning callback error: {str(e)}")
    
    def _handle_cpu_pressure(self, cpu_percent: float):
        """Handle high CPU usage."""
        self.logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
        
        # Call callback if registered
        if self.cpu_warning_callback:
            try:
                self.cpu_warning_callback(cpu_percent)
            except Exception as e:
                self.logger.error(f"CPU warning callback error: {str(e)}")
    
    def _log_resource_status(self, memory_percent: float, cpu_percent: float):
        """Log current resource status."""
        disk_usage = psutil.disk_usage('/').percent
        
        self.logger.info(
            f"Resource Status - Memory: {memory_percent:.1f}%, "
            f"CPU: {cpu_percent:.1f}%, Disk: {disk_usage:.1f}%"
        )
    
    def get_optimal_chunk_size(self, base_chunk_size: int) -> int:
        """Calculate optimal chunk size based on available memory."""
        try:
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            
            # Adjust chunk size based on available memory
            if available_gb > 8:
                return min(base_chunk_size * 2, 1000)
            elif available_gb < 4:
                return max(base_chunk_size // 2, 100)
            else:
                return base_chunk_size
                
        except Exception as e:
            self.logger.error(f"Error calculating optimal chunk size: {str(e)}")
            return base_chunk_size
    
    def get_optimal_worker_count(self, max_workers: int) -> int:
        """Calculate optimal worker count based on system resources."""
        try:
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            
            # Base calculation on CPU cores
            optimal_workers = max(1, cpu_count - 1)
            
            # Adjust based on memory availability
            memory_gb = memory_info.total / (1024**3)
            if memory_gb < 16:
                optimal_workers = max(1, optimal_workers // 2)
            
            # Respect maximum
            return min(optimal_workers, max_workers)
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal worker count: {str(e)}")
            return max_workers
    
    def set_memory_warning_callback(self, callback: Callable[[float], None]):
        """Set callback for memory pressure events."""
        self.memory_warning_callback = callback
    
    def set_cpu_warning_callback(self, callback: Callable[[float], None]):
        """Set callback for CPU pressure events."""
        self.cpu_warning_callback = callback
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            boot_timestamp = psutil.boot_time()
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_percent': psutil.cpu_percent(),
                'cpu_count': psutil.cpu_count(),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'boot_timestamp': boot_timestamp,
                'boot_time_utc': datetime.fromtimestamp(boot_timestamp, tz=timezone.utc).isoformat(),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            self.logger.error(f"Error getting system stats: {str(e)}")
            return {}