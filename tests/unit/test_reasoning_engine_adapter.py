from fastapi.testclient import TestClient

from app.fast_api_app import app


def test_reasoning_engine_routes():
    client = TestClient(app)

    # Test /api/reasoning_engine session creation
    resp = client.post(
        "/api/reasoning_engine",
        json={
            "class_method": "async_create_session",
            "input": {"user_id": "test-user"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "output" in data
    assert "id" in data["output"]
