### src/utils/encryption.py
"""
Configuration Encryption Utilities
"""
import logging
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
import base64
import os


class ConfigEncryption:
    """Utility class for encrypting and decrypting configuration values."""
    
    def __init__(self, encryption_key: str = None):
        self.logger = logging.getLogger(__name__)
        
        if encryption_key:
            try:
                self.fernet = Fernet(encryption_key.encode())
            except (ValueError, TypeError) as e:
                self.fernet = None
                self.logger.error(f"Invalid encryption key provided directly: {e}. Encryption/decryption disabled.")
        else:
            # Try to get from environment variable
            env_key = os.getenv('EDI_ENCRYPTION_KEY')
            if env_key:
                try:
                    self.fernet = Fernet(env_key.encode())
                except (ValueError, TypeError) as e:
                    self.fernet = None
                    self.logger.error(f"Invalid encryption key from EDI_ENCRYPTION_KEY: {e}. Encryption/decryption disabled.")
            else:
                self.fernet = None
        
        if not self.fernet:
            self.logger.warning(
                "No encryption key provided or key was invalid. "
                "Encryption/decryption features will be effectively disabled."
            )

    @property
    def active(self) -> bool:
        """Checks if the Fernet instance is initialized and ready for use."""
        return self.fernet is not None
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        if not self.active:
            self.logger.debug("Encryption skipped: Fernet not active (no valid key).")
            return value
            
        try:
            # Fernet.encrypt returns bytes that are already URL-safe base64 encoded.
            # We just need to decode these bytes to a string (e.g., UTF-8/ASCII).
            encrypted_token_bytes = self.fernet.encrypt(value.encode())
            return encrypted_token_bytes.decode() 
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        if not self.active:
            self.logger.debug("Decryption skipped: Fernet not active (no valid key).")
            return encrypted_value
            
        try:
            # Convert the string token back to bytes for Fernet.
            encrypted_token_bytes = encrypted_value.encode()
            decrypted_bytes = self.fernet.decrypt(encrypted_token_bytes)
            return decrypted_bytes.decode()
        except (Fernet.InvalidToken, TypeError, ValueError, base64.binascii.Error) as e: # Catch specific errors
            self.logger.error(f"Decryption error: {str(e)}")
            return encrypted_value
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()


def decrypt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Decrypt encrypted values in configuration dictionary."""
    logger = logging.getLogger(__name__) # Use a local logger instance
    encryption_key_from_config = config.get('encryption_key')
    encryptor = ConfigEncryption(encryption_key_from_config)
    
    if not encryptor.active:
        logger.warning(
            "Config decryption skipped: Encryption key not available or invalid in config or environment (EDI_ENCRYPTION_KEY)."
        )
        return config
    # The function will build and return a new dictionary structure.
    
    def decrypt_recursive(obj, path=""):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this is an encrypted field
                if key.endswith('_encrypted'):
                    # Decrypt and use original key name
                    original_key = key.replace('_encrypted', '')
                    if isinstance(value, str):
                        decrypted = encryptor.decrypt_value(value)
                        # If decrypt_value returns the original string, it means decryption failed.
                        if decrypted == value and value: # Check value is not empty
                            logger.warning(
                                f"Decryption for key '{key}' at path '{current_path}' might have failed "
                                f"(output same as input). Ensure key is correct and value is a valid encrypted token."
                            )
                        result[original_key] = decrypted
                    else:
                        logger.warning(
                            f"Value for supposed encrypted key '{key}' at path '{current_path}' is not a string "
                            f"(type: {type(value).__name__}). Skipping decryption, using original value."
                        )
                        result[original_key] = value
                else:
                    result[key] = decrypt_recursive(value, current_path)
            return result
        elif isinstance(obj, list):
            return [decrypt_recursive(item, path) for item in obj]
        else:
            return obj
    
    return decrypt_recursive(config)