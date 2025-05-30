try:
    from cryptography.fernet import Fernet
    
    # Generate key
    key = Fernet.generate_key()
    cipher = Fernet(key)
    
    # Encrypt password
    password = "admin"
    encrypted = cipher.encrypt(password.encode())
    
    print("=" * 50)
    print("ENCRYPTION RESULTS:")
    print("=" * 50)
    print(f"Key: {key.decode()}")
    print(f"Encrypted Password: {encrypted.decode()}")
    print("=" * 50)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()