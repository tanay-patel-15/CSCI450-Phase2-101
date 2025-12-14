import bcrypt
import logging

logger = logging.getLogger("auth_utils")
logger.setLevel(logging.INFO)

def hash_password(password: str) -> str:
    """
    Hashes a password using bcrypt.
    Truncates to 60 characters (not bytes) to avoid bcrypt 72-byte limit issues.
    """
    try:
        # Truncate to 60 characters first, then encode
        truncated = password[:75]
        password_bytes = truncated.encode('utf-8')
        
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
    Uses same truncation logic as hashing.
    """
    try:
        # Truncate to match hashing logic
        truncated = password[:75]
        password_bytes = truncated.encode('utf-8')
        
        # Ensure hashed is bytes
        if isinstance(hashed, str):
            hashed_bytes = hashed.encode('utf-8')
        else:
            hashed_bytes = hashed
            
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False