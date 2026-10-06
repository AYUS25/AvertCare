"""
AvertCare Backend · Mock Service Layer

All business logic lives HERE — routes are kept paper-thin.
Swap the `# TODO: REAL MODEL` blocks with actual inference calls
when M1 delivers her .pkl artefacts.
"""

from __future__ import annotations

from app.schemas import (
    PatientEncounter,
    PredictResponse,
    RiskCategory,
    SHAPFeature,
    TwinPatient,
    TwinPatientResponse,
)


# ─────────────────────────────────────────────────────────────
# Predict Service
# ─────────────────────────────────────────────────────────────

def run_prediction(payload: PatientEncounter) -> PredictResponse:
    """
    TODO: REAL MODEL → load XGBoost + ClinicalBERT from MODEL_PATH,
    run inference, compute TreeSHAP values, extract SDoH from clinical_note.

    For now: deterministic mock based on payload signals.
    """

    # ── Risk heuristic (mock) ────────────────────────────────
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

    # ── Mock SHAP features ───────────────────────────────────
    shap_features = [
        SHAPFeature(feature="num_prior_admissions", impact=round(payload.num_prior_admissions * 0.12, 3)),
        SHAPFeature(feature="time_in_hospital", impact=round(payload.time_in_hospital * 0.015, 3)),
        SHAPFeature(feature="num_medications", impact=round(payload.num_medications * 0.02, 3)),
        SHAPFeature(feature="age", impact=round((payload.age - 50) * 0.003, 3)),
    ]
    shap_features.sort(key=lambda x: abs(x.impact), reverse=True)

    # ── Mock SDoH NLP extraction ─────────────────────────────
    sdoh_flags = _extract_sdoh_flags(payload.clinical_note)

    # ── Mock GenAI care plan ─────────────────────────────────
    care_plan = _generate_care_plan(risk_category, sdoh_flags)

    # ── CMS penalty estimate ─────────────────────────────────
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
    TODO: REAL RAG → embed clinical_note via Bio_ClinicalBERT,
    query Qdrant for top-k twins, compute RAGrisk index.

    For now: static mock cohort with realistic clinical data.
    """

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
    snri = round((rag_rate + 0.15) * 0.9, 4)  # mock composite

    return TwinPatientResponse(
        patient_id=payload.patient_id,
        rag_readmission_rate=rag_rate,
        semantic_neighborhood_risk_index=snri,
        sdoh_flag=_extract_sdoh_flags(payload.clinical_note)[0],
        twins=twins,
    )


# ─────────────────────────────────────────────────────────────
# Private Helpers
# ─────────────────────────────────────────────────────────────

_SDOH_KEYWORDS: dict[str, str] = {
    "lives alone": "Social isolation — lives alone",
    "no transport": "Transportation barrier",
    "cannot afford": "Financial distress / medication non-adherence risk",
    "food insecurity": "Food insecurity",
    "homeless": "Housing instability",
    "polypharmacy": "Polypharmacy risk (≥5 medications)",
}


def _extract_sdoh_flags(note: str) -> list[str]:
    """Keyword-based SDoH extraction (stub for ClinicalBERT NER)."""
    note_lower = note.lower()
    flags = [label for kw, label in _SDOH_KEYWORDS.items() if kw in note_lower]
    return flags if flags else ["No SDoH signals detected"]


def _generate_care_plan(risk: RiskCategory, sdoh_flags: list[str]) -> list[str]:
    """Rule-based care plan (stub for GenAI LLM synthesis)."""
    base = [
        "Schedule structured telephone follow-up within 7 days of discharge.",
        "Conduct pharmacist-led medication reconciliation before discharge.",
    ]
    if risk == RiskCategory.HIGH:
        base.insert(0, "⚠️ PRIORITY: Initiate 72-hour post-discharge telehealth check-in.")
    if any("transport" in f.lower() for f in sdoh_flags):
        base.append("Arrange non-emergency medical transport for follow-up appointments.")
    if any("financial" in f.lower() for f in sdoh_flags):
        base.append("Refer to hospital financial navigator for prescription cost-assistance programs.")
    if any("isolation" in f.lower() or "alone" in f.lower() for f in sdoh_flags):
        base.append("Enroll patient in community care coordination programme for daily wellness check-ins.")
    return base
