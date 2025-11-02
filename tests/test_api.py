import pytest
from httpx import AsyncClient
from api import app  # <- import directly, PYTHONPATH points to src

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


