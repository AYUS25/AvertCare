"""
AvertCare Backend · Test Suite

Tests run against the full FastAPI app (no live DB required).
All mock service responses are validated against Pydantic schemas.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())

SAMPLE_PAYLOAD = {
    "patient_id": "TEST-001",
    "age": 67,
    "primary_diagnosis": "Type 2 Diabetes with complications",
    "time_in_hospital": 7,
    "num_prior_admissions": 2,
    "num_medications": 8,
    "clinical_note": (
        "Patient lives alone and has expressed concerns about inability to afford insulin. "
        "Polypharmacy noted. Discharge planned for tomorrow."
    ),
}


class TestHealthProbes:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_returns_200_with_valid_payload(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        assert r.status_code == 200

    def test_risk_score_in_range(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        data = r.json()
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_shap_features_present(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        assert len(r.json()["shap_features"]) > 0

    def test_sdoh_flags_extracted(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        flags = r.json()["sdoh_flags"]
        # "lives alone" and "cannot afford" are in the note
        assert any("alone" in f.lower() or "financial" in f.lower() for f in flags)

    def test_invalid_payload_returns_422(self):
        r = client.post("/api/predict", json={"age": -1})  # missing required fields
        assert r.status_code == 422


class TestTwinPatientEndpoint:
    def test_returns_200_with_valid_payload(self):
        r = client.post("/api/twin-patients", json=SAMPLE_PAYLOAD)
        assert r.status_code == 200

    def test_returns_correct_twin_count(self):
        r = client.post("/api/twin-patients", json=SAMPLE_PAYLOAD)
        assert len(r.json()["twins"]) == 3

    def test_rag_rate_in_range(self):
        r = client.post("/api/twin-patients", json=SAMPLE_PAYLOAD)
        rate = r.json()["rag_readmission_rate"]
        assert 0.0 <= rate <= 1.0
