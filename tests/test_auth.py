import pytest
from fastapi.testclient import TestClient
from api import app
import os
import boto3
import uuid

client = TestClient(app)

# Credentials matching the hardcoded values in your app
TEST_ADMIN_EMAIL = "ece30861defaultadminuser" 
TEST_ADMIN_PASSWORD = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"

# DynamoDB test table config
USERS_TABLE = os.environ.get("USERS_TABLE", "users")
dynamodb = boto3.resource("dynamodb")
db_users_table = dynamodb.Table(USERS_TABLE)

@pytest.fixture(scope="module")
def test_user_email():
    # FIX: Use a random email to prevent "User already exists" (400) errors
    # caused by leftover data from previous runs.
    return f"test_{uuid.uuid4()}@example.com"

@pytest.fixture(scope="function", autouse=True)
def cleanup_user(test_user_email):
    # Attempt to remove test user before test
    try:
        db_users_table.delete_item(Key={"email": test_user_email})
    except Exception as e:
        print(f"Warning: Setup cleanup failed: {e}")
    
    yield
    
    # Attempt to remove test user after test
    try:
        db_users_table.delete_item(Key={"email": test_user_email})
    except Exception as e:
        print(f"Warning: Teardown cleanup failed: {e}")

def test_register_and_login(test_user_email):
    # Register
    r = client.post("/register", json={"email": test_user_email, "password": "secret"})
    assert r.status_code == 200, f"Register failed: {r.text}"
    assert r.json() == {"message": "registered"}

    # Login
    r = client.put("/authenticate", json={
        "user": {"name": test_user_email, "is_admin": False},
        "secret": {"password": "secret"}
    })
    assert r.status_code == 200, f"Login failed: {r.text}"
    
    # FIX: Check for string response, not dictionary
    token_response = r.json()
    assert isinstance(token_response, str)
    assert token_response.lower().startswith("bearer ")

def test_default_admin_authenticate_flow():
    """
    Test that the default admin can authenticate.
    Relies on self-healing logic in src/auth.py.
    """
    login_payload = {
        "user": {
            "name": TEST_ADMIN_EMAIL, 
            "is_admin": True,
        },
        "secret": {
            "password": TEST_ADMIN_PASSWORD
        }
    }
    
    response = client.put("/authenticate", json=login_payload)
    
    # Check for success
    if response.status_code != 200:
        print(f"Admin Login Failed (Debug): {response.status_code}")
        print(f"Response Body: {response.text}")
        
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    # FIX: Spec defines response as a simple string "bearer <token>"
    token = response.json()
    assert isinstance(token, str), f"Expected string token, got {type(token)}"
    assert token.lower().startswith("bearer ")