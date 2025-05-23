import pytest
import base64
import os
import sys

# Add project root directory to sys.path for imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app  # FastAPI app instance

client = TestClient(app)

# Test cases data: [filename, expected_label, expected_status]
test_cases = [
    ("clear_id.jpg", "suspicious", "manual_review"),
    ("fake_template.jpg", "fake", "rejected"),
    ("cropped_id.jpg", "fake", "rejected"),
    ("poor_ocr.jpg", "fake", "rejected"),
    ("meme.jpg", "fake", "rejected"),
]

@pytest.mark.parametrize("filename, expected_label, expected_status", test_cases)
def test_validate_id(filename, expected_label, expected_status):
    # Prepare image path inside tests/sample_inputs
    image_path = os.path.join(os.path.dirname(__file__), "sample_inputs", filename)
    
    # Read and encode image as base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Prepare request payload
    payload = {
        "user_id": "test_user",
        "image_base64": image_base64
    }

    # Send POST request to /validate-id
    response = client.post("/validate-id", json=payload)
    
    # Basic checks
    assert response.status_code == 200
    data = response.json()

    # Validate user_id is returned correctly
    assert data["user_id"] == "test_user"

    # Validate label and status match expected
    assert data["label"] == expected_label
    assert data["status"] == expected_status

    # Validation score should be between 0 and 1
    assert 0.0 <= data["validation_score"] <= 1.0

    # Reason field should be present and non-empty string
    assert "reason" in data
    assert isinstance(data["reason"], str)
    assert len(data["reason"]) > 0