### src/utils/encryption.py
"""
Configuration Encryption Utilities
"""
import logging
from typing import Dict, Any
from cryptography.fernet import Fernet
import base64
import os


class ConfigEncryption:
    """Utility class for encrypting and decrypting configuration values."""
    
    def __init__(self, encryption_key: str = None):
        self.logger = logging.getLogger(__name__)
        
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            # Try to get from environment variable
            env_key = os.getenv('EDI_ENCRYPTION_KEY')
            if env_key:
                self.fernet = Fernet(env_key.encode())
            else:
                self.fernet = None
                self.logger.warning("No encryption key provided")
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        if not self.fernet:
            return value
            
        try:
            encrypted_bytes = self.fernet.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        if not self.fernet:
            return encrypted_value
            
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            return encrypted_value
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()


def decrypt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Decrypt encrypted values in configuration dictionary."""
    encryption_key = config.get('encryption_key')
    if not encryption_key:
        return config
    
    encryptor = ConfigEncryption(encryption_key)
    decrypted_config = {}
    
    def decrypt_recursive(obj, path=""):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this is an encrypted field
                if key.endswith('_encrypted'):
                    # Decrypt and use original key name
                    original_key = key.replace('_encrypted', '')
                    result[original_key] = encryptor.decrypt_value(value)
                else:
                    result[key] = decrypt_recursive(value, current_path)
            return result
        elif isinstance(obj, list):
            return [decrypt_recursive(item, path) for item in obj]
        else:
            return obj
    
    return decrypt_recursive(config)