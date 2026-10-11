import json
import app as service

def test_health_endpoint():
    client = service.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["feature_count"] == 30

def test_home_contains_profile_links():
    response = service.app.test_client().get("/")
    assert response.status_code == 200
    assert b"github.com/InfinitePraveen" in response.data
    assert b"linkedin.com/in/infinitepraveen" in response.data

def test_predict_rejects_missing_features():
    response = service.app.test_client().post("/predict", json={"features": {}})
    assert response.status_code == 400
    assert "missing_features" in response.get_json()
