### src/config/config_manager.py
"""
Configuration Manager with Encryption Support
"""
import logging
import yaml
import os
from typing import Dict, Any
from utils.encryption import decrypt_config



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
            
            self.logger.info(f"Raw configuration loaded from {self.config_path}")
            
            # Check if encryption is actually needed
            has_encryption_key = 'encryption_key' in raw_config
            has_encrypted_fields = self._has_encrypted_fields(raw_config)
            
            if has_encryption_key and has_encrypted_fields:
                self.logger.info("Found encryption key and encrypted fields, attempting decryption")
                try:
                    config = decrypt_config(raw_config)
                    # Remove encryption key from memory for security
                    if 'encryption_key' in config:
                        del config['encryption_key']
                    self.logger.info("Configuration decrypted successfully")
                except Exception as e:
                    self.logger.error(f"Decryption failed: {str(e)}")
                    self.logger.warning("Using raw configuration without decryption")
                    config = raw_config
            elif has_encryption_key and not has_encrypted_fields:
                self.logger.info("Found encryption key but no encrypted fields, skipping decryption")
                config = raw_config
                # Still remove the unused encryption key
                if 'encryption_key' in config:
                    del config['encryption_key']
            elif not has_encryption_key and has_encrypted_fields:
                self.logger.warning("Found encrypted fields but no encryption key - encrypted values will not work")
                config = raw_config
            else:
                self.logger.info("No encryption configured, using plain configuration")
                config = raw_config
            
            # Validate required configuration sections
            self._validate_config(config)
            
            self.logger.info("Configuration loaded and validated successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _has_encrypted_fields(self, config: Dict[str, Any]) -> bool:
        """Check if configuration contains any encrypted fields (ending with '_encrypted')."""
        def check_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.endswith('_encrypted'):
                        return True
                    if isinstance(value, (dict, list)):
                        if check_recursive(value):
                            return True
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)) and check_recursive(item):
                        return True
            return False
        
        result = check_recursive(config)
        self.logger.debug(f"Encrypted fields check result: {result}")
        return result
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate required configuration sections."""
        required_sections = ['database']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate database configuration
        db_config = config['database']
        if 'postgresql' not in db_config:
            raise ValueError("PostgreSQL configuration is required")
        
        # Check PostgreSQL connection parameters
        pg_config = db_config['postgresql']
        required_pg_keys = ['host', 'port', 'database', 'user', 'password']
        missing_keys = []
        
        for key in required_pg_keys:
            if key not in pg_config:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required PostgreSQL configuration keys: {missing_keys}")
        
        # Validate processing configuration if it exists
        if 'processing' in config:
            proc_config = config['processing']
            recommended_proc_keys = ['chunk_size', 'max_workers']
            for key in recommended_proc_keys:
                if key not in proc_config:
                    self.logger.warning(f"Missing processing configuration: {key}, using default")
        else:
            self.logger.info("No processing configuration found, will use defaults")
    
    def reload_config(self):
        """Force reload configuration from file."""
        self._config = None
        return self.get_config()
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration section."""
        config = self.get_config()
        return config.get('database', {})
    
    def get_postgresql_config(self) -> Dict[str, Any]:
        """Get PostgreSQL configuration."""
        db_config = self.get_database_config()
        return db_config.get('postgresql', {})