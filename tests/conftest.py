import pytest
import boto3
import os


@pytest.fixture
def client():
    """Shared TestClient fixture for all tests"""
    from fastapi.testclient import TestClient
    try:
        from src.api import app
    except ImportError:
        from api import app
    return TestClient(app)


@pytest.fixture
def db_users_table():
    """Create a boto3 DynamoDB table object fixture for the users table"""
    USERS_TABLE = os.environ.get("USERS_TABLE", "users")
    dynamodb = boto3.resource("dynamodb")
    return dynamodb.Table(USERS_TABLE)