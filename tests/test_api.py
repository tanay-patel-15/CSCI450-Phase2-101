import pytest
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport  # <- use this
from api import app  # PYTHONPATH points to src
from fastapi.testclient import TestClient
from fastapi import FastAPI
from tests.test_helpers import make_jwt
import os
import boto3
from time import time

@pytest.fixture(scope="module", autouse=True)
def create_initial_data():

    test_model_id = "rest-model-autograder-101"
    model_table = dynamodb.Table(MODELS_TABLE)

    model_table.put_item(
        Item={
            "model_id": test_model_id,
            "name": "Autograder Test Model",
            "version": "1.0",
            "sensitive": False,
            "status": "approved",
            "owner": "admin_user",
            "timestamp": int(time())
        }
    )
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=test_model_id,
        Body=b"Mock data for model download test"
    )
    yield

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        response_data = r.json()
        assert response_data["status"] == "ok"
        assert response_data["version"] == "N/A"
        assert isinstance(response_data["uptime_seconds"], int)
        assert isinstance(response_data["latency_ms"], int)

client = TestClient(app)
MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group102")
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "audit_logs")
USERS_TABLE = os.environ.get("USERS_TABLE", "users")
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

#------------------------------
# Test /reset route
#------------------------------
@pytest.mark.asyncio
async def test_reset_endpoint(existing_model):
    """Tests the /reset endpoint to ensure proper functionality."""
    s3_list_response = s3.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1)
    if s3_list_response.get("KeyCount", 0) == 0:
        pytest.skip(f"S3 Bucket {BUCKET_NAME} is empty.")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        token = make_jwt("admin")
        response = await async_client.post(
            "/reset",
            headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == 200
    assert response.json().get("message") == "Successfully reset the environment"

    model_table = dynamodb.Table(MODELS_TABLE)
    users_table = dynamodb.Table(USERS_TABLE)
    audit_table = dynamodb.Table(AUDIT_TABLE)

    model_scan = model_table.scan(Limit=1)
    assert model_scan.get("Count", 0) == 0, f"Models Table {MODELS_TABLE} was not cleared by /reset."

    users_scan = users_table.scan(Limit=1)
    assert users_scan.get("Count", 0) == 0, f"Users Table {USERS_TABLE} was not cleared by /reset."

    audit_scan = audit_table.scan(Limit=1)
    assert audit_scan.get("Count", 0) == 1, f"Audit Table {AUDIT_TABLE} was not cleared by /reset."

    s3_cleared_response = s3.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1)
    assert s3_cleared_response.get("KeyCount", 0) == 0, f"S3 Bucket {BUCKET_NAME} was not cleared by /reset."


# ------------------------------
# Test /models route
# ------------------------------

@pytest.fixture(scope="module")
def existing_model():
    """
    Retrieve an existing model from DynamoDB for testing.
    """
    table = dynamodb.Table(MODELS_TABLE)
    response = table.scan(Limit=1)  # Get at least one model
    items = [
        item for item in response.get("Items", []) 
        if "model_id" in item
        and "sensitive" in item
        and isinstance(item["sensitive"], bool)
    ]
    if not items:
        test_model_id = "rest-model-recreated-2025"

        table.put_item(
            Item={
                "model_id": test_model_id,
                "name": "Recreated Test Model",
                "version": "1.0",
                "sensitive": False,
                "status": "APPROVED",
                "owner": "admin_user",
                "timestamp": int(time())
            }
        )
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=test_model_id,
            Body=b"Mock data for recreated model download test"
        )

        response = table.get_item(Key={"model_id": test_model_id})
        model = response["Item"]
        return model
    return items[0]  # return the first model


def test_list_all_models(existing_model):
    token = make_jwt("viewer")
    response = client.get("/models", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    models = response.json().get("models", [])
    assert len(models) >= 1  # At least one model must exist
    assert any(m["model_id"] == existing_model["model_id"] for m in models)


def test_list_models_with_search(existing_model):
    # Use part of the model_id as regex
    token = make_jwt("viewer")
    search_str = existing_model["model_id"][:5]
    response = client.get(f"/models?search={search_str}", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    models = response.json().get("models", [])
    assert any(search_str in m["model_id"] for m in models)


def test_download_model(existing_model):
    model_id = existing_model["model_id"]
    token = make_jwt("viewer")
    # Ensure the model exists in S3 before testing download
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=model_id)
    except s3.exceptions.ClientError:
        pytest.skip(f"Model {model_id} not found in S3. Skipping download test.")

    response = client.get(f"/download/{model_id}", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.content  # Ensure some data is returned