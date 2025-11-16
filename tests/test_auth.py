import pytest
from fastapi.testclient import TestClient
from api import app
import os
import boto3

client = TestClient(app)

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