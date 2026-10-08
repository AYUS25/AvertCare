"""
AvertCare Backend · Service Layer
===================================
Feature-flagged service layer:
  - LIVE_RAG_ENABLED=false  → deterministic mock (default)
  - LIVE_RAG_ENABLED=true   → live Qdrant vector search
  - LIVE_LLM_ENABLED=false  → rule-based care plan (default)
  - LIVE_LLM_ENABLED=true   → Gemini API prescriptive synthesis

Swap points for M1's models are marked with: # TODO: REAL MODEL
"""

from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np
from functools import lru_cache

from app.core.config import settings
from app.schemas import (
    PatientEncounter,
    PredictResponse,
    RiskCategory,
    SHAPFeature,
    TwinPatient,
    TwinPatientResponse,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lazy singletons — only loaded when feature flags are enabled
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_qdrant_client():
    """Cached Qdrant client — initialised once per process."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=settings.QDRANT_URL, timeout=10)
    logger.info("Qdrant client connected to %s", settings.QDRANT_URL)
    return client


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Cached sentence-transformer — download once, reuse forever."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded: all-MiniLM-L6-v2")
    return model


@lru_cache(maxsize=1)
def _get_gemini_client():
    """Cached Gemini client."""
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    logger.info("Gemini client initialised (model=%s)", settings.GEMINI_MODEL)
    return client


# ─────────────────────────────────────────────────────────────
# Predict Service
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_ml_artifacts():
    """
    Load ML artifacts from the mounted volume (/app/models).
    Returns None if artifacts are not present (e.g., in CI or before M1 handoff),
    which causes run_prediction to fall back to the deterministic mock.
    """
    import joblib
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    meta_path = os.path.join(MODELS_DIR, "feature_names.json")

    if not all(os.path.exists(p) for p in [model_path, prep_path, meta_path]):
        logger.warning("ML model artifacts not found at %s — using deterministic mock.", MODELS_DIR)
        return None

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    if isinstance(preprocessor, dict):
        preprocessor = preprocessor["preprocessor"]

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info("ML artifacts loaded: model=%s, uses_rag=%s", meta.get("winning_model"), meta.get("uses_rag"))
    return {"model": model, "preprocessor": preprocessor, "meta": meta}


def run_prediction(payload: PatientEncounter) -> PredictResponse:
    """
    Hybrid inference:
    - If ML artifacts are mounted (/app/models): runs real RF_RAG model + SHAP.
    - If not available (CI / no volume mount): falls back to deterministic mock.
    """
    artifacts = _get_ml_artifacts()

    # ── FALLBACK: deterministic mock ──────────────────────────────────────────
    if artifacts is None:
        raw_score = min(
            0.3
            + (payload.num_prior_admissions * 0.12)
            + (payload.num_medications * 0.02)
            + (0.08 if payload.time_in_hospital > 5 else 0),
            1.0,
        )
        risk_category = (
            RiskCategory.HIGH if raw_score >= 0.65
            else RiskCategory.MODERATE if raw_score >= 0.40
            else RiskCategory.LOW
        )
        shap_features = [
            SHAPFeature(feature="num_prior_admissions", impact=round(payload.num_prior_admissions * 0.12, 3)),
            SHAPFeature(feature="time_in_hospital",      impact=round(payload.time_in_hospital * 0.015, 3)),
            SHAPFeature(feature="num_medications",       impact=round(payload.num_medications * 0.02, 3)),
            SHAPFeature(feature="age",                   impact=round((payload.age - 50) * 0.003, 3)),
        ]
        shap_features.sort(key=lambda x: abs(x.impact), reverse=True)
        sdoh_flags = _extract_sdoh_flags(payload.clinical_note)
        care_plan = (
            _generate_llm_care_plan(raw_score, sdoh_flags, shap_features)
            if settings.LIVE_LLM_ENABLED
            else _generate_rule_care_plan(risk_category, sdoh_flags)
        )
        cms_saved = round(15_400 * raw_score, 2) if risk_category == RiskCategory.HIGH else None
        return PredictResponse(
            patient_id=payload.patient_id,
            risk_score=round(raw_score, 4),
            risk_category=risk_category,
            shap_features=shap_features,
            sdoh_flags=sdoh_flags,
            care_plan=care_plan,
            cms_penalty_saved_usd=cms_saved,
        )

    # ── LIVE: real RF_RAG model inference ─────────────────────────────────────
    model = artifacts["model"]
    prep = artifacts["preprocessor"]
    meta = artifacts["meta"]

    # 1. Run RAG retrieval to get the readmit rate (if live)
    rag_response = run_rag_retrieval(payload)
    rag_rate = rag_response.rag_readmission_rate

    # 2. Construct raw pandas DataFrame matching preprocessor's expected columns
    data = {
        'age_numeric': [payload.age],
        'num_lab_procedures': [40],
        'num_med_changes': [0],
        'num_medications': [payload.num_medications],
        'num_meds_active': [payload.num_medications],
        'num_procedures': [0],
        'number_diagnoses': [1],
        'number_emergency': [0],
        'number_inpatient': [payload.num_prior_admissions],
        'number_outpatient': [0],
        'time_in_hospital': [payload.time_in_hospital],
        'total_prior_visits': [payload.num_prior_admissions],
        'A1Cresult': ['Not_Tested'],
        'admission_source_id': ['Other'],
        'admission_type_id': ['Other'],
        'change': ['No'],
        'diabetesMed': ['No'],
        'diag_1_group': ['Other'],
        'diag_2_group': ['Other'],
        'diag_3_group': ['Other'],
        'discharge_disposition_id': ['1'],
        'gender': ['Unknown'],
        'glimepiride': ['No'],
        'glipizide': ['No'],
        'glyburide': ['No'],
        'insulin': ['No'],
        'max_glu_serum': ['Not_Tested'],
        'medical_specialty': ['Unknown'],
        'metformin': ['No'],
        'payer_code': ['Unknown'],
        'pioglitazone': ['No'],
        'race': ['Unknown'],
        'repaglinide': ['No'],
        'rosiglitazone': ['No']
    }
    df = pd.DataFrame(data)

    # 3. Transform and append RAG feature
    X_tab = prep.transform(df)
    X_input = np.hstack([X_tab, np.array([[rag_rate]])]) if meta["uses_rag"] else X_tab

    # 4. Predict probability
    raw_score = float(model.predict_proba(X_input)[:, 1][0])

    risk_category = (
        RiskCategory.HIGH if raw_score >= 0.65
        else RiskCategory.MODERATE if raw_score >= 0.40
        else RiskCategory.LOW
    )

    # 5. SHAP values — TreeExplainer for RF; graceful fallback on failure
    shap_features: list[SHAPFeature] = []
    try:
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        sv = sv[0]
        feature_names = meta["feature_names"]
        all_shap = [
            SHAPFeature(feature=feature_names[i], impact=round(float(sv[i]), 4))
            for i in range(len(feature_names))
        ]
        all_shap.sort(key=lambda x: abs(x.impact), reverse=True)
        shap_features = all_shap[:5]
    except Exception as exc:
        logger.warning("SHAP explanation failed (%s); returning empty list.", exc)

    sdoh_flags = _extract_sdoh_flags(payload.clinical_note)

    # 6. Care plan
    care_plan = (
        _generate_llm_care_plan(raw_score, sdoh_flags, shap_features)
        if settings.LIVE_LLM_ENABLED
        else _generate_rule_care_plan(risk_category, sdoh_flags)
    )

    cms_saved = round(15_400 * raw_score, 2) if risk_category == RiskCategory.HIGH else None

    return PredictResponse(
        patient_id=payload.patient_id,
        risk_score=round(raw_score, 4),
        risk_category=risk_category,
        shap_features=shap_features,
        sdoh_flags=sdoh_flags,
        care_plan=care_plan,
        cms_penalty_saved_usd=cms_saved,
    )



# ─────────────────────────────────────────────────────────────
# Twin-Patient RAG Service
# ─────────────────────────────────────────────────────────────

def run_rag_retrieval(payload: PatientEncounter) -> TwinPatientResponse:
    """
    LIVE_RAG_ENABLED=True  → embeds clinical_note, queries Qdrant,
                              computes real RAGrisk index.
    LIVE_RAG_ENABLED=False → deterministic mock cohort.
    """
    if settings.LIVE_RAG_ENABLED:
        return _live_rag_retrieval(payload)
    return _mock_rag_retrieval(payload)


def _live_rag_retrieval(payload: PatientEncounter) -> TwinPatientResponse:
    """Query the live Qdrant collection."""
    model = _get_embedding_model()
    client = _get_qdrant_client()

    query_vector = model.encode(
        payload.clinical_note,
        normalize_embeddings=True,
    ).tolist()

    results = client.search(
        collection_name=settings.RAG_COLLECTION,
        query_vector=query_vector,
        limit=settings.RAG_TOP_K + 1,
        with_payload=True,
    )

    # Drop any near-exact self-match
    results = [r for r in results if r.score < 0.9999][: settings.RAG_TOP_K]

    twins = [
        TwinPatient(
            twin_id=r.payload.get("patient_id", f"P-{i}"),
            age=int(r.payload.get("age", 65)),
            primary_diagnosis=str(r.payload.get("primary_diagnosis", "Unknown")),
            similarity_score=round(float(r.score), 4),
            was_readmitted=bool(r.payload.get("readmitted_binary", 0)),
            successful_interventions=r.payload.get("interventions", []),
        )
        for i, r in enumerate(results)
    ]

    readmitted_count = sum(1 for t in twins if t.was_readmitted)
    rag_rate = round(readmitted_count / max(len(twins), 1), 4)
    snri = round((rag_rate + 0.15) * 0.9, 4)

    return TwinPatientResponse(
        patient_id=payload.patient_id,
        rag_readmission_rate=rag_rate,
        semantic_neighborhood_risk_index=snri,
        sdoh_flag=_extract_sdoh_flags(payload.clinical_note)[0],
        twins=twins,
    )


def _mock_rag_retrieval(payload: PatientEncounter) -> TwinPatientResponse:
    """Deterministic mock cohort (used before Qdrant is seeded)."""
    twins = [
        TwinPatient(
            twin_id="P-10042",
            age=payload.age + 3,
            primary_diagnosis=payload.primary_diagnosis,
            similarity_score=0.94,
            was_readmitted=False,
            successful_interventions=[
                "72-hr telehealth follow-up scheduled",
                "Pharmacist medication reconciliation completed",
            ],
        ),
        TwinPatient(
            twin_id="P-10087",
            age=payload.age - 5,
            primary_diagnosis=payload.primary_diagnosis,
            similarity_score=0.88,
            was_readmitted=True,
            successful_interventions=[],
        ),
        TwinPatient(
            twin_id="P-10213",
            age=payload.age + 1,
            primary_diagnosis=payload.primary_diagnosis,
            similarity_score=0.82,
            was_readmitted=False,
            successful_interventions=[
                "Home nursing visit arranged within 48 hrs",
                "Social work referral for transport assistance",
            ],
        ),
    ]
    readmitted_count = sum(1 for t in twins if t.was_readmitted)
    rag_rate = round(readmitted_count / len(twins), 4)
    snri = round((rag_rate + 0.15) * 0.9, 4)

    return TwinPatientResponse(
        patient_id=payload.patient_id,
        rag_readmission_rate=rag_rate,
        semantic_neighborhood_risk_index=snri,
        sdoh_flag=_extract_sdoh_flags(payload.clinical_note)[0],
        twins=twins,
    )


# ─────────────────────────────────────────────────────────────
# Gemini LLM Care Plan
# ─────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are an expert hospital discharge planner and clinical decision support specialist.
You will receive structured JSON containing:
  - patient_risk_score: float [0,1] — readmission probability
  - sdoh_flags: list[str] — Social Determinants of Health barriers
  - top_shap_features: list[{feature, impact}] — top XGBoost risk drivers
  - twin_interventions: list[str] — successful interventions from similar non-readmitted patients

Your task: Generate a concise, evidence-based 3-point prescriptive discharge plan.

Respond ONLY with valid JSON in exactly this format:
{
  "interventions": [
    "Specific clinical directive 1",
    "Specific clinical directive 2",
    "Specific clinical directive 3"
  ],
  "cms_penalty_savings_estimate": <integer USD>,
  "clinical_rationale": "One-sentence rationale referencing the top SHAP driver and SDoH barrier."
}
"""


def _generate_llm_care_plan(
    risk_score: float,
    sdoh_flags: list[str],
    shap_features: list[SHAPFeature],
) -> list[str]:
    """Call Gemini API to synthesise a prescriptive care plan."""
    try:
        client = _get_gemini_client()
        user_payload = json.dumps({
            "patient_risk_score": risk_score,
            "sdoh_flags": sdoh_flags,
            "top_shap_features": [{"feature": f.feature, "impact": f.impact} for f in shap_features[:3]],
            "twin_interventions": [
                "72-hr telehealth follow-up",
                "Pharmacist medication reconciliation",
                "Home nursing referral",
            ],
        })

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=f"{_LLM_SYSTEM_PROMPT}\n\nPatient Data:\n{user_payload}",
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            raw = raw.removeprefix("json")

        parsed = json.loads(raw)
        return parsed.get("interventions", _generate_rule_care_plan(RiskCategory.HIGH, sdoh_flags))

    except Exception as exc:
        logger.warning("Gemini call failed (%s), falling back to rule-based care plan.", exc)
        return _generate_rule_care_plan(RiskCategory.HIGH, sdoh_flags)


# ─────────────────────────────────────────────────────────────
# Private Helpers
# ─────────────────────────────────────────────────────────────

_SDOH_KEYWORDS: dict[str, str] = {
    "lives alone":       "Social isolation — lives alone",
    "no transport":      "Transportation barrier",
    "cannot afford":     "Financial distress / medication non-adherence risk",
    "food insecurity":   "Food insecurity",
    "homeless":          "Housing instability",
    "polypharmacy":      "Polypharmacy risk (≥5 medications)",
    "transportation barrier": "Transportation barrier",
    "uninsured":         "Uninsured — prescription cost-adherence risk",
}


def _extract_sdoh_flags(note: str) -> list[str]:
    """Keyword-based SDoH extraction (stub for ClinicalBERT NER)."""
    note_lower = note.lower()
    flags = [label for kw, label in _SDOH_KEYWORDS.items() if kw in note_lower]
    return flags if flags else ["No SDoH signals detected"]


def _generate_rule_care_plan(risk: RiskCategory, sdoh_flags: list[str]) -> list[str]:
    """Rule-based care plan fallback."""
    base = [
        "Schedule structured telephone follow-up within 7 days of discharge.",
        "Conduct pharmacist-led medication reconciliation before discharge.",
    ]
    if risk == RiskCategory.HIGH:
        base.insert(0, "⚠️ PRIORITY: Initiate 72-hour post-discharge telehealth check-in.")
    if any("transport" in f.lower() for f in sdoh_flags):
        base.append("Arrange non-emergency medical transport for follow-up appointments.")
    if any("financial" in f.lower() or "uninsured" in f.lower() for f in sdoh_flags):
        base.append("Refer to hospital financial navigator for prescription cost-assistance programs.")
    if any("isolation" in f.lower() or "alone" in f.lower() for f in sdoh_flags):
        base.append("Enroll patient in community care coordination for daily wellness check-ins.")
    return base
