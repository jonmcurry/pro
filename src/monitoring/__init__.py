# === src/monitoring/__init__.py ===
"""
Monitoring and metrics package
"""
from .metrics import MetricsCollector, system_metrics
from .notifications import EmailNotifier

__all__ = ['MetricsCollector', 'system_metrics', 'EmailNotifier']