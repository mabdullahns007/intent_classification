from fastapi.testclient import TestClient
from api.main import app
import json

# Create a test client for the application
client = TestClient(app)

# Credentials for the authenticated endpoint
AUTH_USER = "admin"
AUTH_PASS = "secretpassword"

# --- Test Data ---
SINGLE_QUERY = {"text": "Can you book a meeting with the client next week?"}
BATCH_QUERIES = {"texts": ["Find me information about the nearest coffee shop", "Send an email to HR", "What is 7 plus 8?"]}

def test_1_health_check_ok():
    """Test the GET /api/health endpoint returns 200 and model is loaded."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["api_status"] == "ok"
    assert response.json()["model_loaded"] == True

def test_2_classify_single_success():
    """Test the POST /api/classify endpoint with a single query."""
    response = client.post("/api/classify", json=SINGLE_QUERY)
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert isinstance(data["intent"], str)
    assert isinstance(data["confidence"], float)
    assert data["confidence"] > 0.5 # Confidence check

def test_3_classify_single_invalid_input():
    """Test handling of invalid input (empty body)"""
    response = client.post("/api/classify", json={})
    # FastAPI/Pydantic validation error returns 422
    assert response.status_code == 422 

def test_4_classify_batch_success():
    """Test the POST /api/classify/batch endpoint with multiple queries."""
    response = client.post("/api/classify/batch", json=BATCH_QUERIES)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3 
    
    # Check structure of results
    for result in data:
        assert all(key in result for key in ["text", "intent", "confidence"])
        assert isinstance(result["confidence"], float)

def test_5_model_info_unauthorized():
    """Test GET /api/model/info fails without authentication (401)."""
    response = client.get("/api/model/info")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers

def test_6_model_info_authorized():
    """Test GET /api/model/info succeeds with correct Basic Auth (200)."""
    response = client.get(
        "/api/model/info",
        auth=(AUTH_USER, AUTH_PASS)
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Scikit-learn Logistic Regression Pipeline"
    assert "supported_intents" in data
    assert data["model_metrics"]["overall_accuracy"] == 0.9000 # Metric check