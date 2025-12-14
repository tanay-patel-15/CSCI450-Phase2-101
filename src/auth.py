from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.auth_utils import hash_password, verify_password
from src.jwt_utils import create_token
import boto3
import os
import logging
import json
import base64

router = APIRouter()
logger = logging.getLogger("auth_logger")
logger.setLevel(logging.INFO)

# --- Configuration (Matching Spec) ---
ADMIN_EMAIL_ENV_KEY = "ADMIN_EMAIL_ENV"
ADMIN_PASSWORD_ENV_KEY = "ADMIN_PASSWORD_ENV"
DEFAULT_ADMIN_EMAIL = os.environ.get(ADMIN_EMAIL_ENV_KEY, "ece30861defaultadminuser")
ENCODED_ADMIN_PASSWORD = os.environ.get(ADMIN_PASSWORD_ENV_KEY)
DEFAULT_ADMIN_PASSWORD = None

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

from src.db_setup import create_tables_if_missing

if ENCODED_ADMIN_PASSWORD:
    try:
        DEFAULT_ADMIN_PASSWORD = base64.b64decode(ENCODED_ADMIN_PASSWORD).decode('utf-8')
        logger.info("Admin password successfully decoded from Base64")
    except Exception as e:
        logger.error(f"FATAL: Base64 decoding failed. Error: {e}")
        DEFAULT_ADMIN_PASSWORD = None
else:
    logger.warning(f"FATAL: {ADMIN_PASSWORD_ENV_KEY} is required but missing from environment!")
    DEFAULT_ADMIN_PASSWORD = None

def get_user_table():
    create_tables_if_missing()
    table_name = os.environ.get("USERS_TABLE", "users-group101-unique-v3")
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
    """Authenticate a user and return a JWT token."""
    ddb = get_user_table()
    
    username = body.user.name 
    password = body.secret.password

    user_item = ddb.get_item(Key={"email": username}).get("Item")

    if not user_item and username == DEFAULT_ADMIN_EMAIL:
        logger.info("Admin user not found. Performing self-healing.")
        try:
            # Generate a hash with the corrected hash_password function
            hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
            
            logger.error(f"DEBUG: Storing new hash: {hashed}")
            
            admin_item = {
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": hashed,
                "role": "admin",
            }
            ddb.put_item(Item=admin_item)
            user_item = admin_item
            logger.info("Admin user force-healed in DB")
        except Exception as e:
            logger.error(f"Failed to force-heal admin: {e}")
            raise HTTPException(500, "Internal authentication setup failure.")

    # Spec requires 401 for invalid credentials
    if not user_item:
        raise HTTPException(401, "The user or password is invalid.")

    # Verification uses the corrected verify_password function
    if not verify_password(password, user_item["password_hash"]):
        raise HTTPException(401, "The user or password is invalid.")

    # Create the token
    token = create_token({"sub": username, "role": user_item["role"]})
    
    # Return as JSON string per OpenAPI spec example: '"bearer ..."'
    return JSONResponse(content=f"bearer {token}")