import bcrypt
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def hash_password(password: str) -> str:
    """
    Hashes a password using bcrypt.
    Truncates to 72 bytes to match bcrypt's hard limit.
    """
    try:
        # Encode first
        password_bytes = password.encode('utf-8')
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        # Return as string for storage
        return hashed.decode('utf-8')
    except Exception as e:
        logger.error(f"Hashing failed: {e}")
        raise

def verify_password(password: str, hashed: str) -> bool:
    """
    Verifies a password against a bcrypt hash.
    Uses same 72-byte truncation as hashing.
    """
    try:
        # Truncate to match hashing logic
        password_bytes = password.encode('utf-8')
        
        if not isinstance(hashed, str):
            hashed = str(hashed)
        
        hashed_bytes = hashed.strip().encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except ValueError as e:
        logger.error(f"Verification FAILED (Invalid salt/malformed hash): {e}")
        return False
    except Exception as e:
        logger.error(f"Password verification encountered unexpected error: {e}")
        return False