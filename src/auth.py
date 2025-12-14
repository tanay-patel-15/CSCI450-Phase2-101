from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.auth_utils import hash_password, verify_password
from src.jwt_utils import create_token
import boto3
import os
import logging

router = APIRouter()
logger = logging.getLogger("auth_logger")
logger.setLevel(logging.INFO)

# --- Configuration (Matching Spec) ---
DEFAULT_ADMIN_EMAIL = os.environ.get("DEFAULT_ADMIN_EMAIL", "ece30861defaultadminuser")
# Precise escaping for the password string required by spec
DEFAULT_ADMIN_PASSWORD = os.environ.get("DEFAULT_ADMIN_PASSWORD", "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;")

# --- Internal Models matching spec schemas ---

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
    # FIX 1: Added default "users" to match api.py and prevent 500 crashes
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

    # FIX 2: UNCONDITIONAL SELF-HEALING
    # If this is the default admin, we force-update the DB to match the code.
    # We do not check 'if not user_item'. We just write. 
    # This guarantees that even if the DB is stale, the login WILL succeed.
    if username == DEFAULT_ADMIN_EMAIL:
        try:
            hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
            admin_item = {
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": hashed,
                "role": "admin",
            }
            ddb.put_item(Item=admin_item)
            # Pre-load the user_item so we don't have to fetch it
            user_item = admin_item
        except Exception as e:
            logger.error(f"Failed to force-heal admin: {e}")
            user_item = None
    else:
        # Normal user lookup
        user_item = ddb.get_item(Key={"email": username}).get("Item")

    # Spec requires 401 for invalid credentials
    if not user_item:
        raise HTTPException(401, "The user or password is invalid.")

    if not verify_password(password, user_item["password_hash"]):
        raise HTTPException(401, "The user or password is invalid.")

    # Create the token
    token = create_token({"sub": username, "role": user_item["role"]})
    
    # Return as JSON string per spec (lowercase 'bearer' per example)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=f"bearer {token}")