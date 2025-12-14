import pytest
from fastapi.testclient import TestClient
from api import app
import os
import boto3

client = TestClient(app)

TEST_ADMIN_EMAIL = "ece30861defaultadminuser" 
TEST_ADMIN_PASSWORD = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"

#DynamoDB test table
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

        # Login
        r = client.post("/login", json={"username": test_user_email, "password": "secret"})
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

def test_default_admin_authenticate_flow():
    """
    Test that the default admin can authenticate with correct credentials.
    This test relies on the app's self-healing logic.
    """
    
    # 1. Define the cleartext payload locally to avoid global config issues
    login_payload = {
        "username": TEST_ADMIN_EMAIL,
        "password": TEST_ADMIN_PASSWORD  # Guaranteed to be a valid string
    }
    
    # The first call should trigger self-healing if the user is missing
    response = client.post("/login", json=login_payload)
    
    # Check for success
    if response.status_code != 200:
        print(f"Admin Login Failed (Debug): {response.status_code}")
        print(f"Response Body: {response.text}")
        
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"