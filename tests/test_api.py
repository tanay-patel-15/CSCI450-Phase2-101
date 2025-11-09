import pytest
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport  # <- use this
from api import app  # PYTHONPATH points to src
from fastapi.testclient import TestClient
from fastapi import FastAPI
import os

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

client = TestClient(app)

# ------------------------------
# Test /models route
# ------------------------------

def test_list_all_models_structure():
    """
    Check that /models returns at least 1 model and that each model
    has the expected keys.
    """
    response = client.get("/models")
    assert response.status_code == 200

    models = response.json().get("models", [])
    assert len(models) > 0, "No models returned from /models"

    for model in models:
        assert "model_id" in model
        assert "category" in model
        assert "size" in model

def test_search_param_filtering():
    """
    Test /models with search query. Should return only models
    whose names match the regex pattern.
    """
    # Take the first model's name from real data
    models_resp = client.get("/models").json().get("models", [])
    if not models_resp:
        pytest.skip("No models available for testing search")
    
    sample_model_name = models_resp[0]["model_id"]
    # Simple regex: search for part of the model_id
    search_str = sample_model_name[:3]

    response = client.get(f"/models?search={search_str}")
    assert response.status_code == 200
    filtered_models = response.json().get("models", [])
    assert any(search_str in m["model_id"] for m in filtered_models)

# ------------------------------
# Test /download/{id} route
# ------------------------------

def test_download_model():
    """
    Test /download/{id} returns a file for a non-sensitive model.
    """
    models_resp = client.get("/models").json().get("models", [])
    if not models_resp:
        pytest.skip("No models available for download test")

    # Pick the first model to download
    model_id = models_resp[0]["model_id"]

    response = client.get(f"/download/{model_id}")
    assert response.status_code == 200
    # Ensure response is a file (bytes)
    assert response.content, "Downloaded file is empty"

