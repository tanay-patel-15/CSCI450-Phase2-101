import os
from src.auth_utils import hash_password

DEFAULT_ADMIN_PASSWORD = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"

print(f"Password length: {len(DEFAULT_ADMIN_PASSWORD)}")

try:
    hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
    print(f"Hash success: {hashed}")
except Exception as e:
    print(f"Hash failed: {e}")
