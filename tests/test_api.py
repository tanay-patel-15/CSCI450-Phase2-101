import pytest
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport  # <- use this
from api import app, router  # PYTHONPATH points to src
from fastapi.testclient import TestClient
from fastapi import FastAPI

app.include_router(router)
client = TestClient(app)

# Mock data for DynamoDB
mock_models = [
    {"id": "1", "model_name": "resnet50", "sensitive": False},
    {"id": "2", "model_name": "bert-base", "sensitive": True},
    {"id": "3", "model_name": "resnet101", "sensitive": False},
]

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

@pytest.fixture(autouse=True)
def mock_dynamodb(monkeypatch):
    class MockTable:
        def scan(self):
            return {"Items": mock_models}
        
        def get_item(self, Key):
            for m in mock_models:
                if m["id"] == Key["id"]:
                    return {"Item": m}
            return {}

    class MockResource:
        def Table(self, name):
            return MockTable()
    
    monkeypatch.setattr("src.api.boto3.resource", lambda service: MockResource())

# ----- Tests for /models -----
def test_list_all_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert len(response.json()["models"]) == 3

def test_search_models_regex():
    response = client.get("/models?search=resnet")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) == 2
    assert all("resnet" in m["model_name"] for m in models)

def test_search_no_matches():
    response = client.get("/models?search=nonexistent")
    assert response.status_code == 200
    assert response.json()["models"] == []

def test_invalid_regex():
    response = client.get("/models?search=[")
    # Expect either 400 or empty list depending on implementation
    assert response.status_code == 400 or response.json()["models"] == []

# ----- Tests for /download/{id} -----
def test_download_non_sensitive_model():
    response = client.get("/download/1")
    assert response.status_code == 200
    assert response.json()["model_name"] == "resnet50"

def test_download_sensitive_model():
    response = client.get("/download/2")
    assert response.status_code == 403 or "error" in response.json()

def test_download_nonexistent_model():
    response = client.get("/download/999")
    assert response.status_code == 404 or "error" in response.json()