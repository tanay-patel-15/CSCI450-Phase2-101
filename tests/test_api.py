import pytest
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport  # <- use this
from api import app  # PYTHONPATH points to src
from fastapi.testclient import TestClient
from fastapi import FastAPI
from tests.test_helpers import make_jwt
import os
import boto3

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
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

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
        pytest.skip("No models found in DynamoDB for testing.")
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