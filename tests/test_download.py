import pytest
import os
import requests
from fastapi.testclient import TestClient
from datetime import datetime
from botocore.exceptions import ClientError
from api import app
from tests.utils import make_jwt
import boto3

BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group102")
MODEELS_TABLE = os.environ.get("MODELS_TABLE", "models")
SECURITY_HOOK_URL = os.environ.get("SECURITY_HOOK_URL")
MAX_DOWNLOAD_SIZE_BYTES = 10 * 1024 * 1024

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
models_table = dynamodb.Table(MODEELS_TABLE)
client = TestClient(app)

@pytest.fixture
def test_model_item():
    item = {
        "model_id": "test_model_001",
        "name": "Test Model",
        "sensitive": False,
    }
    models_table.put_item(Item=item)
    yield item
    models_table.delete_item(Key={"model_id": item["model_id"]})

@pytest.fixture
def test_sensitive_model_item():
    item = {
        "model_id": "sensitive_model_001",
        "name": "Sensitive Model",
        "sensitive": True,
    }
    models_table.put_item(Item=item)
    yield item
    models_table.delete_item(Key={"model_id": item["model_id"]})

@pytest.fixture
def test_s3_file():
    key = "test/test-model-001.zip"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=b"Test data")
    yield key
    s3.delete_object(Bucket=BUCKET_NAME, Key=key)

@pytest.fixture
def test_sensitive_s3_file():
    key = "sensitive/sensitive-model-001.zip"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=b"Sensitive test data")
    yield key
    s3.delete_object(Bucket=BUCKET_NAME, Key=key)

def get_auth_headers(role="viewer"):
    token = make_jwt(role=role)
    return {"Authorization": f"Bearer {token}"}

def test_download_regular_model(test_model_item, test_s3_file):
    headers = get_auth_headers(role="viewer")
    response = client.get(f"/download?model_id={test_model_item['model_id']}", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.content == b"Test data"

def test_download_sensitive_model_admin(test_sensitive_model_item, test_sensitive_s3_file):
    headers = get_auth_headers(role="admin")
    response = client.get(f"/download/={test_sensitive_model_item['model_id']}", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.content == b"Sensitive data"

def test_download_sensitive_model_viewer_forbidden(test_sensitive_model_item, test_sensitive_s3_file):
    headers = get_auth_headers(role="viewer")
    response = client.get(f"/download/={test_sensitive_model_item['model_id']}", headers=headers)
    assert response.status_code == 403
    assert "restricted" in response.json()["detail"].lower()

def test_download_invalid_model_id():
    headers = get_auth_headers(role = "admin")
    response = client.get("/download/invalid$$$id", headers=headers)
    assert response.status_code == 400

def test_download_missing_model():
    headers = get_auth_headers(role="admin")
    response = client.get("/download/nonexistent_model_xyz", headers=headers)
    assert response.status_code == 404

def test_download_file_too_large(test_model_item):
    key = f"test/{test_model_item['model_id']}.zip"
    large_data = b"x" * (MAX_DOWNLOAD_SIZE_BYTES + 1)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=large_data)
    headers = get_auth_headers(role="admin")
    response = client.get(f"/download?model_id={test_model_item['model_id']}", headers=headers)
    assert response.status_code == 413
    s3.delete_object(Bucket=BUCKET_NAME, Key=key)

def test_security_hook_called(monkeypatch, test_sensitive_model_item, test_sensitive_s3_file):
    called = {}

    def fake_post(url, json, timeout):
        called.update(json)
        return type("Resp", (), {"status_code": 200})()
    
    monkeypatch.setattr(requests, "post", fake_post)
    headers = get_auth_headers(role="admin")
    client.get(f"/download/={test_sensitive_model_item['model_id']}", headers=headers)
    assert called.get("model_id") == test_sensitive_model_item["model_id"]
    assert called.get("sensitive") is True