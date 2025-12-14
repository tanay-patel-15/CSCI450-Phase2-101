import pytest
import boto3
import os

# CRITICAL: Set environment variables BEFORE importing the app
# This ensures DB table names are consistent throughout
os.environ["USERS_TABLE"] = "users"
os.environ["MODELS_TABLE"] = "models" 
os.environ["AUDIT_TABLE"] = "audit_logs"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from moto import mock_aws
from fastapi.testclient import TestClient

# Import AFTER setting env vars
from src.api import app 
from src.auth import DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD

@pytest.fixture(name="client")
def client_fixture():
    return TestClient(app)

@mock_aws
def test_login_flow(client):
    """
    Test that the default admin can login with correct credentials.
    The app should auto-create tables and initialize admin on first request.
    """
    # Login payload per OpenAPI spec
    login_payload = {
        "user": {
            "name": DEFAULT_ADMIN_EMAIL,
            "is_admin": True
        },
        "secret": {
            "password": DEFAULT_ADMIN_PASSWORD
        }
    }
    
    response = client.put("/authenticate", json=login_payload)
    
    # Debug info
    if response.status_code != 200:
        print(f"Login Failed: {response.status_code}")
        print(f"Response: {response.text}")
    
        # Check if table has items
        try:
            debug_table = boto3.resource("dynamodb", region_name="us-east-1").Table("users")
            items = debug_table.scan()['Items']
            print(f"Table content: {items}")
        except Exception as e:
            print(f"Could not scan table: {e}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert "bearer" in response.text.lower(), f"Expected 'bearer' in response, got: {response.text}"
