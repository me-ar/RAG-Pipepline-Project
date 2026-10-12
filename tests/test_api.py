from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_requires_question_field():
    # Missing 'question' field should return a 422 validation error
    response = client.post("/query", json={})
    assert response.status_code == 422