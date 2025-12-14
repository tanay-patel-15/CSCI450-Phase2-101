import bcrypt
import logging

# Configure logger
logger = logging.getLogger("auth_utils")
logger.setLevel(logging.INFO)

def hash_password(password: str) -> str:
    """
    Hashes a password using bcrypt.
    Truncates input to 60 bytes to avoid bcrypt 72-byte limit and library errors.
    """
    try:
        # Truncate to 60 bytes to be safe
        password_bytes = password.encode('utf-8')[:60]
        
        # Generate salt and hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        # Return as string for storage
        return hashed.decode('utf-8')
    except Exception as e:
        logger.error(f"Hashing failed: {e}")
        raise

def verify_password(password: str, hashed: str) -> bool:
    """
    Verifies a password against a bcrypt hash.
    """
    try:
        # Truncate to match hashing logic
        password_bytes = password.encode('utf-8')[:60]
        
        # Ensure hashed is bytes
        if isinstance(hashed, str):
            hashed_bytes = hashed.encode('utf-8')
        else:
            hashed_bytes = hashed
            
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False