import pytest
from fastapi.testclient import TestClient
from api import app
import boto3
import os

@pytest.fixture
def client():
    """Shared TestClient fixture for all tests"""

    return TestClient(app)

@pytest.fixture
def db_users_table():
    """Create a boto3 DynamoDB table object fixture for the users table"""

    USERS_TABLE = os.environ.get("USERS_TABLE", "users")
    dynamodb = boto3.resource("dynamodb")
    return dynamodb.Table(USERS_TABLE)

    