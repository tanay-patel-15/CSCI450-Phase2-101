import pytest
from fastapi.testclient import TestClient
from api import app
import os
import boto3

client = TestClient(app)

# Use the credentials that match your unconditional self-healing logic
TEST_ADMIN_EMAIL = "ece30861defaultadminuser" 
TEST_ADMIN_PASSWORD = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"

# DynamoDB test table config
USERS_TABLE = os.environ.get("USERS_TABLE", "users")
dynamodb = boto3.resource("dynamodb")
db_users_table = dynamodb.Table(USERS_TABLE)

@pytest.fixture(scope="module")
def test_user_email():
    return "test@a.com"

@pytest.fixture(scope="function", autouse=True)
def cleanup_user(test_user_email):
    # Remove test user before and after each test to avoid conflicts
    try:
        db_users_table.delete_item(Key={"email": test_user_email})
    except Exception:
        pass
    yield
    try:
        db_users_table.delete_item(Key={"email": test_user_email})
    except Exception:
        pass

def test_register_and_login(test_user_email):
    # Register
    r = client.post("/register", json={"email": test_user_email, "password": "secret"})
    assert r.status_code == 200
    assert r.json() == {"message": "registered"}

    # Login - Ordinary users might use the same format, but let's assume
    # checking for the "bearer" string is sufficient for now based on your API.
    r = client.put("/authenticate", json={
        "user": {"name": test_user_email, "is_admin": False},
        "secret": {"password": "secret"}
    })
    assert r.status_code == 200
    
    # FIX: Check for string response, not dictionary
    token_response = r.json()
    assert isinstance(token_response, str)
    assert token_response.lower().startswith("bearer ")

def test_default_admin_authenticate_flow():
    """
    Test that the default admin can authenticate with correct credentials.
    This test relies on the app's self-healing logic.
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
    
    # FIX: The API Spec defines the response as a simple string, not a JSON object
    token = response.json()
    
    # Assert it is a string and starts with 'bearer'
    assert isinstance(token, str), f"Expected string token, got {type(token)}"
    assert token.lower().startswith("bearer "), f"Token should start with 'bearer', got: {token}"