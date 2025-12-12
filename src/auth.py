from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from auth_utils import hash_password, verify_password
from jwt_utils import create_token
import boto3
import os

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "viewer"


class LoginRequest(BaseModel):
    # tests send "username" here
    username: str
    password: str


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


@router.post("/login")
def login(body: LoginRequest):
    ddb = get_user_table()
    email = body.username  # important: tests send "username"
    user = ddb.get_item(Key={"email": email}).get("Item")

    if not user:
        raise HTTPException(401, "Invalid credentials")

    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")

    token = create_token({"sub": email, "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}
