### src/utils/logging_config.py
"""
Logging Configuration
"""
import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Pattern, Set
import re


class PHISafeFormatter(logging.Formatter):
    """Custom formatter that sanitizes PHI from log messages."""
    
    # Pre-compile regex patterns for efficiency
    _PHI_PATTERNS: List[Tuple[Pattern[str], str]] = [
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'XXX-XX-XXXX'),  # SSN pattern
        (re.compile(r'\b\d{16}\b'), 'XXXX-XXXX-XXXX-XXXX'),  # Basic credit card pattern
        (re.compile(r'\bDOB[:\s]*\d{1,2}/\d{1,2}/\d{4}\b', re.IGNORECASE), 'DOB: XX/XX/XXXX'),  # Date of birth
        # Add more specific or broader patterns as needed
    ]

    # Standard LogRecord attributes that are handled explicitly or are not considered "extra"
    _STANDARD_RECORD_ATTRS: Set[str] = {
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'message', 'module',
        'msecs', 'msg', 'name', 'pathname', 'process', 'processName',
        'relativeCreated', 'stack_info', 'thread', 'threadName',
        # Attributes explicitly added to log_entry
        'timestamp', 'level', 'logger' 
        # 'message' is derived from getMessage(), 'module', 'function', 'line' are explicit
    }

    def __init__(self):
        super().__init__()

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
            # Only add attributes not already handled or standard
            if key not in self._STANDARD_RECORD_ATTRS and not hasattr(logging.LogRecord, key):
                log_entry[key] = value
        
        return json.dumps(log_entry)
    
    def _sanitize_message(self, message: str) -> str:
        """Remove or mask PHI from log messages."""
        sanitized_message = message
        for pattern, replacement in self._PHI_PATTERNS:
            sanitized_message = pattern.sub(replacement, sanitized_message)
        return sanitized_message


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup structured logging with PHI safety."""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        # Fallback or critical error handling if log directory cannot be created
        logging.error(f"Could not create log directory {log_dir}: {e}", exc_info=True)
        # Depending on requirements, you might raise e or use a default path
    
    # Default log file
    log_file_path = log_file if log_file else os.path.join(log_dir, "edi_processing.log")
    
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
        log_file_path,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(PHISafeFormatter())
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors and above
    base_log_name, log_ext = os.path.splitext(log_file_path)
    error_file = f"{base_log_name}_errors{log_ext if log_ext else '.log'}"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(PHISafeFormatter())
    root_logger.addHandler(error_handler)
    
    logging.info("Logging system initialized")