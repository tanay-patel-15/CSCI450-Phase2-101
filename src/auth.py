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
# We define these here to ensure the auth module can self-heal the admin account
DEFAULT_ADMIN_EMAIL = os.environ.get("DEFAULT_ADMIN_EMAIL", "ece30861defaultadminuser")
# Careful with the escaping here. 
# Spec: correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;
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

# Keep RegisterRequest for internal use/testing setup
class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "viewer"

def get_user_table():
    table_name = os.environ.get("USERS_TABLE")
    if not table_name:
        raise RuntimeError("Environment variable USERS_TABLE not set")
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
    Matches the spec: PUT /authenticate with nested user/secret body.
    """
    ddb = get_user_table()
    
    username = body.user.name 
    password = body.secret.password

    # Retrieve user
    user_item = ddb.get_item(Key={"email": username}).get("Item")

    # --- SELF-HEALING ADMIN FIX ---
    # If the default admin exists but has a STALE/WRONG password hash in the DB,
    # we must repair it immediately, or we will never be able to login to /reset.
    if username == DEFAULT_ADMIN_EMAIL:
        should_heal = False
        
        # Case 1: User is missing entirely
        if not user_item:
            should_heal = True
        
        # Case 2: User exists, but the stored hash does not match our current Configured Password
        # (This happens if the DB persists across runs but the code/password string changed)
        elif "password_hash" in user_item:
            # Check if the SOURCE OF TRUTH password works with the STORED hash
            # If verify_password returns False, the DB is stale.
            if not verify_password(DEFAULT_ADMIN_PASSWORD, user_item["password_hash"]):
                should_heal = True
        
        if should_heal:
            logger.info(f"Admin state for {username} is stale or missing. Healing database now.")
            hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
            admin_item = {
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": hashed,
                "role": "admin",
            }
            try:
                ddb.put_item(Item=admin_item)
                user_item = admin_item # Update the local variable to use the fresh data
            except Exception as e:
                logger.error(f"Failed to heal admin user: {e}")
                # We proceed, though it will likely fail 401 below
    # -----------------------------

    # Spec requires 401 for invalid credentials
    if not user_item:
        raise HTTPException(401, "The user or password is invalid.")

    if not verify_password(password, user_item["password_hash"]):
        raise HTTPException(401, "The user or password is invalid.")

    # Create the token
    token = create_token({"sub": username, "role": user_item["role"]})
    
    # Return exactly what the spec expects: A simple string "bearer <token>"
    return f"bearer {token}"