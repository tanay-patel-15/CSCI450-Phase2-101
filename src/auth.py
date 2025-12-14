from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.auth_utils import hash_password, verify_password
from src.jwt_utils import create_token
import boto3
import os

router = APIRouter()

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
        # user already exists or other constraint violation
        raise HTTPException(400, "User already exists")

@router.put("/authenticate")
def authenticate(body: AuthenticationRequest):
    """
    Authenticate a user and return a JWT token.
    Matches the spec: PUT /authenticate with nested user/secret body.
    """
    ddb = get_user_table()
    
    # Map the spec's "name" field to our DB's "email" key
    username = body.user.name 
    password = body.secret.password

    user_item = ddb.get_item(Key={"email": username}).get("Item")

    # Spec requires 401 for invalid credentials
    if not user_item:
        raise HTTPException(401, "The user or password is invalid.")

    if not verify_password(password, user_item["password_hash"]):
        raise HTTPException(401, "The user or password is invalid.")

    # Create the token
    token = create_token({"sub": username, "role": user_item["role"]})
    
    # Return exactly what the spec expects: A simple string "bearer <token>"
    return f"bearer {token}"