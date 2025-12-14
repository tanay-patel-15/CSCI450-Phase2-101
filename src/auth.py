from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.auth_utils import hash_password, verify_password
from src.jwt_utils import create_token
from src.db_setup import create_tables_if_missing
import boto3
import os
import logging
import base64

router = APIRouter()
logger = logging.getLogger("auth_logger")
logger.setLevel(logging.INFO)

# --- Configuration ---
# 1. MATCHES template.yml variable name
# 2. Handles the Base64 encoding passed by SAM
env_pass = os.environ.get("DEFAULT_ADMIN_PASSWORD")
SPEC_PASSWORD = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"

if env_pass:
    try:
        # Try decoding in case it's the Base64 version from template
        DEFAULT_ADMIN_PASSWORD = base64.b64decode(env_pass).decode('utf-8')
    except Exception:
        # If decode fails, assume it was passed as raw text
        DEFAULT_ADMIN_PASSWORD = env_pass
else:
    # Fallback to the hardcoded spec string if env var is missing
    DEFAULT_ADMIN_PASSWORD = SPEC_PASSWORD

DEFAULT_ADMIN_EMAIL = os.environ.get("DEFAULT_ADMIN_EMAIL", "ece30861defaultadminuser")

# --- Internal Models ---
class UserIdentity(BaseModel):
    name: str
    is_admin: bool = False

class UserSecret(BaseModel):
    password: str

class AuthenticationRequest(BaseModel):
    user: UserIdentity
    secret: UserSecret

class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "viewer"

def get_user_table():
    create_tables_if_missing() # Ensure table exists before connecting
    table_name = os.environ.get("USERS_TABLE", "users")
    return boto3.resource("dynamodb").Table(table_name)

@router.post("/register")
def register(body: RegisterRequest):
    ddb = get_user_table()
    hashed = hash_password(body.password)
    try:
        ddb.put_item(
            Item={
                "email": body.email,
                "password_hash": hashed,
                "role": body.role,
            },
            ConditionExpression="attribute_not_exists(email)"
        )
        return {"message": "registered"}
    except Exception:
        raise HTTPException(400, "User already exists")

@router.put("/authenticate")
def authenticate(body: AuthenticationRequest):
    """
    Authenticate a user and return a JWT token.
    """
    ddb = get_user_table()
    
    username = body.user.name 
    password = body.secret.password

    # --- UNCONDITIONAL SELF-HEALING ---
    # We do NOT check "if not user_item". 
    # We ALWAYS overwrite the admin credentials on login.
    # This ensures the DB hash matches the current code configuration,
    # solving the "Zombie User" deadlock.
    if username == DEFAULT_ADMIN_EMAIL:
        try:
            hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
            admin_item = {
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": hashed,
                "role": "admin",
            }
            ddb.put_item(Item=admin_item)
            user_item = admin_item 
            logger.info("Admin user state force-healed.")
        except Exception as e:
            logger.error(f"Failed to heal admin: {e}")
            # Fallback to fetching if write fails
            user_item = ddb.get_item(Key={"email": username}).get("Item")
    else:
        # Normal user lookup
        user_item = ddb.get_item(Key={"email": username}).get("Item")

    if not user_item:
        raise HTTPException(401, "The user or password is invalid.")

    if not verify_password(password, user_item["password_hash"]):
        logger.error(f"Password verification failed for {username}")
        raise HTTPException(401, "The user or password is invalid.")

    token = create_token({"sub": username, "role": user_item["role"]})
    
    # Return specific string format per spec examples
    return JSONResponse(content=f"bearer {token}")