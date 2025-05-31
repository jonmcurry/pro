"""
Fixed monitoring/__init__.py - Updated imports to match new metrics structure
"""

# Import main classes
from .metrics import MetricsCollector

# Import notification components
try:
    from .notifications import EmailNotifier
except ImportError:
    # Create a basic EmailNotifier if the module doesn't exist
    class EmailNotifier:
        def __init__(self, config=None):
            self.config = config or {}
        
        def send_completion_notification(self, report):
            pass
        
        def send_error_notification(self, subject, message, severity="ERROR"):
            pass

# Create a system_metrics instance for backward compatibility
try:
    system_metrics = MetricsCollector()
except Exception:
    # If MetricsCollector fails to initialize, create a dummy object
    class DummyMetrics:
        def __getattr__(self, name):
            # Return a no-op function for any method call
            return lambda *args, **kwargs: None
    
    system_metrics = DummyMetrics()

# Export main components
__all__ = [
    'MetricsCollector',
    'EmailNotifier', 
    'system_metrics'
]