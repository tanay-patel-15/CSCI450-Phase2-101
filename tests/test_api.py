import pytest
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport  # <- use this
from api import app  # PYTHONPATH points to src

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}



