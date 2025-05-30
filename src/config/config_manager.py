### src/config/config_manager.py
"""
Configuration Manager with Encryption Support
"""
import logging
import yaml
import os
from typing import Dict, Any
from ..utils.encryption import decrypt_config


class ConfigurationManager:
    """Manages system configuration with encryption support."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self._config = None
        
    def get_config(self) -> Dict[str, Any]:
        """Load and decrypt configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Decrypt sensitive values if encryption is configured
            if 'encryption_key' in raw_config:
                config = decrypt_config(raw_config)
                # Remove encryption key from memory for security
                if 'encryption_key' in config:
                    del config['encryption_key']
            else:
                config = raw_config
            
            # Validate required configuration sections
            self._validate_config(config)
            
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate required configuration sections."""
        required_sections = ['database', 'processing']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate database configuration
        db_config = config['database']
        if 'postgresql' not in db_config or 'sqlserver' not in db_config:
            raise ValueError("Both PostgreSQL and SQL Server configurations are required")
        
        # Validate processing configuration
        proc_config = config['processing']
        required_proc_keys = ['chunk_size', 'max_workers']
        for key in required_proc_keys:
            if key not in proc_config:
                self.logger.warning(f"Missing processing configuration: {key}, using default")