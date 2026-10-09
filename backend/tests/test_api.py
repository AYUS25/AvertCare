"""
AvertCare Backend · Test Suite

Auth dependency is overridden via FastAPI dependency_overrides so tests
never require a live Firebase project or real JWT token.
"""

from fastapi.testclient import TestClient

from app.core.auth import require_auth
from app.main import create_app


# ── Auth stub — injected for all tests ──────────────────────────────────────
def _mock_auth():
    return {"uid": "test-clinician", "email": "test@avertcare.local"}


def _make_client() -> TestClient:
    app = create_app()
    app.dependency_overrides[require_auth] = _mock_auth
    return TestClient(app)


client = _make_client()

# ── Shared fixture payload ───────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
class TestHealthProbes:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────
class TestPredictEndpoint:
    def test_returns_200_with_valid_payload(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        assert r.status_code == 200

    def test_risk_score_in_range(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        data = r.json()
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_shap_features_is_list(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        # shap_features is a list — may be empty if model artifacts not mounted in CI
        assert isinstance(r.json()["shap_features"], list)

    def test_sdoh_flags_extracted(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        flags = r.json()["sdoh_flags"]
        # "lives alone" and "cannot afford" are in the note
        assert any("alone" in f.lower() or "financial" in f.lower() for f in flags)

    def test_care_plan_present(self):
        r = client.post("/api/predict", json=SAMPLE_PAYLOAD)
        body = r.json()
        assert 1 <= len(body["care_plan"]) <= 3
        assert body["plan_source"] in {"rules", "gemini"}
        assert isinstance(body["clinical_rationale"], str) and body["clinical_rationale"]

    def test_short_note_returns_422(self):
        payload = {**SAMPLE_PAYLOAD, "clinical_note": "too short"}
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_payload_returns_422(self):
        r = client.post("/api/predict", json={"age": -1})  # missing required fields
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────
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

    def test_diagnosis_group_present(self):
        r = client.post("/api/twin-patients", json=SAMPLE_PAYLOAD)
        assert r.json()["twins"][0]["diagnosis_group"]
