from fastapi import APIRouter, HTTPException
from src.auth_utils import hash_password, verify_password
from src.jwt_utils import create_token
import boto3
import os

router = APIRouter()

ddb = boto3.resource("dynamodb").Table(os.getenv("USERS_TABLE"))

@router.post("/register")
def register(email: str, password: str, role: str = "viewer"):
    hashed = hash_password(password)

    try:
        ddb.put_item(
            Item={
                "email": email,
                "password_hash": hashed,
                "role": role
            },
            ConditionExpression="attribute_not_exists(email)"
        )
        return {"message": "registered"}
    except:
        raise HTTPException(400, "User already exists")
    
@router.post("/login")
def login(email: str, password: str):
    user = ddb.get_item(Key={"email": email}).get("Item")
    if not user:
        raise HTTPException(401, "Invalid credentials")
    
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    
    token = create_token({"sub": email, "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}

