### src/utils/logging_config.py
"""
Logging Configuration
"""
import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Any, Dict


class PHISafeFormatter(logging.Formatter):
    """Custom formatter that sanitizes PHI from log messages."""
    
    def __init__(self):
        super().__init__()
        
        # Patterns to sanitize (basic examples)
        self.phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{16}\b',  # Credit card pattern
            r'\bDOB[:\s]*\d{1,2}/\d{1,2}/\d{4}\b',  # Date of birth
        ]
    
    def format(self, record):
        """Format log record with PHI sanitization."""
        # Create JSON log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': self._sanitize_message(record.getMessage()),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread',
                          'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry)
    
    def _sanitize_message(self, message: str) -> str:
        """Remove or mask PHI from log messages."""
        import re
        
        sanitized = message
        
        # Replace SSN patterns
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', sanitized)
        
        # Replace potential card numbers
        sanitized = re.sub(r'\b\d{16}\b', 'XXXX-XXXX-XXXX-XXXX', sanitized)
        
        # Replace DOB patterns
        sanitized = re.sub(r'\bDOB[:\s]*\d{1,2}/\d{1,2}/\d{4}\b', 'DOB: XX/XX/XXXX', sanitized)
        
        return sanitized


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup structured logging with PHI safety."""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Default log file
    if not log_file:
        log_file = os.path.join(log_dir, "edi_processing.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(PHISafeFormatter())
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors and above
    error_file = log_file.replace('.log', '_errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(PHISafeFormatter())
    root_logger.addHandler(error_handler)
    
    logging.info("Logging system initialized")