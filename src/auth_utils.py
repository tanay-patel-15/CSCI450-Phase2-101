from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    # Bcrypt has a hard limit of 72 bytes - truncate to comply
    # Reducing to 60 to be extra safe against library quirks/off-by-one errors
    password_bytes = password.encode('utf-8')[:60]
    return pwd_context.hash(password_bytes.decode('utf-8', errors='ignore'))

def verify_password(password: str, hashed: str) -> bool:
    # Bcrypt has a hard limit of 72 bytes - truncate to match hashing
    password_bytes = password.encode('utf-8')[:60]
    return pwd_context.verify(password_bytes.decode('utf-8', errors='ignore'), hashed)