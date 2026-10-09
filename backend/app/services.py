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
from functools import lru_cache

import numpy as np
import pandas as pd

from app.core.config import settings
from app.note_template import build_discharge_note
from app.schemas import (
    PatientEncounter,
    PlanSource,
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
    client = QdrantClient(
        url=settings.QDRANT_URL, 
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
        timeout=10
    )
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


def _diagnosis_group(text: str) -> str:
    """Map a free-text diagnosis onto the nine groups used in training."""
    lowered = (text or "").lower()
    rules = (
        ("diabet", "Diabetes"),
        ("heart", "Circulatory"),
        ("cardiac", "Circulatory"),
        ("coronary", "Circulatory"),
        ("respir", "Respiratory"),
        ("pneumo", "Respiratory"),
        ("copd", "Respiratory"),
        ("kidney", "Genitourinary"),
        ("renal", "Genitourinary"),
        ("digest", "Digestive"),
        ("gastro", "Digestive"),
        ("liver", "Digestive"),
        ("cancer", "Neoplasms"),
        ("neoplasm", "Neoplasms"),
        ("fracture", "Injury"),
        ("trauma", "Injury"),
        ("arthritis", "Musculoskeletal"),
    )
    for needle, group in rules:
        if needle in lowered:
            return group
    return "Other"


def _a1c(value: str) -> str:
    if value in {">7", ">8", "Norm", "Not_Tested"}:
        return value
    return "Not_Tested"


_DISPOSITIONS = {"1", "2", "3", "4", "5", "6", "7", "18", "22", "25", "Other"}
_DIAG_GROUPS = {
    "Circulatory", "Respiratory", "Digestive", "Genitourinary",
    "Neoplasms", "Musculoskeletal", "Injury", "Diabetes", "Other",
}


def _model_frame(payload: PatientEncounter) -> pd.DataFrame:
    """Build the 34 training columns from the clinician payload.

    Fields the UI collects are passed through. Fields the screen does not
    collect stay at the same safe defaults the preprocessor was fit to accept.
    """
    inpatient = int(payload.number_inpatient)
    # Older clients sent prior admits on num_prior_admissions only.
    if inpatient == 0 and int(payload.num_prior_admissions) > 0:
        inpatient = int(payload.num_prior_admissions)
    emergency = int(payload.number_emergency)
    diabetes_med = payload.diabetes_med if payload.diabetes_med in {"Yes", "No"} else "No"
    gender = payload.gender if payload.gender in {"Male", "Female", "Unknown"} else "Unknown"
    disposition = payload.discharge_disposition if payload.discharge_disposition in _DISPOSITIONS else "Other"
    diag_group = payload.diag_1_group if payload.diag_1_group in _DIAG_GROUPS else _diagnosis_group(payload.primary_diagnosis)
    return pd.DataFrame({
        "age_numeric": [payload.age],
        "num_lab_procedures": [40],
        "num_med_changes": [0],
        "num_medications": [payload.num_medications],
        "num_meds_active": [1 if diabetes_med == "Yes" else 0],
        "num_procedures": [0],
        "number_diagnoses": [int(payload.number_diagnoses)],
        "number_emergency": [emergency],
        "number_inpatient": [inpatient],
        "number_outpatient": [0],
        "time_in_hospital": [payload.time_in_hospital],
        "total_prior_visits": [inpatient + emergency],
        "A1Cresult": [_a1c(payload.a1c_result)],
        "admission_source_id": ["Other"],
        "admission_type_id": ["Other"],
        "change": ["No"],
        "diabetesMed": [diabetes_med],
        "diag_1_group": [diag_group],
        "diag_2_group": ["Other"],
        "diag_3_group": ["Other"],
        "discharge_disposition_id": [disposition],
        "gender": [gender],
        "glimepiride": ["No"],
        "glipizide": ["No"],
        "glyburide": ["No"],
        "insulin": ["No"],
        "max_glu_serum": ["Not_Tested"],
        "medical_specialty": ["Unknown"],
        "metformin": ["No"],
        "payer_code": ["Unknown"],
        "pioglitazone": ["No"],
        "race": ["Unknown"],
        "repaglinide": ["No"],
        "rosiglitazone": ["No"],
    })


@lru_cache(maxsize=1)
def _get_safe_rag_index():
    """Train-only embedding index. None when the safe index has not been built."""
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(base_dir), "models")
    index_path = os.path.join(models_dir, "rag_train_index.npz")
    meta_path = os.path.join(models_dir, "rag_train_meta.csv")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None
    loaded = np.load(index_path)
    embeddings = loaded["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)
    labels = loaded["labels"].astype(np.float32)
    meta = pd.read_csv(meta_path)
    logger.info("Loaded leakage-safe RAG index (%d training encounters).", len(labels))
    return embeddings, labels, meta


def _rag_k(meta: dict | None) -> int:
    if not meta:
        return 10
    try:
        return int(meta.get("rag_k") or 10)
    except (TypeError, ValueError):
        return 10


def _safe_rag_retrieval(payload: PatientEncounter, k: int) -> TwinPatientResponse | None:
    """Neighborhood rate and twins from the training index only."""
    index = _get_safe_rag_index()
    if index is None:
        return None
    embeddings, labels, meta = index
    note = build_discharge_note(_model_frame(payload).iloc[0].to_dict())
    query = _get_embedding_model().encode(note, normalize_embeddings=True).astype(np.float32)
    sims = embeddings @ query
    k = max(1, min(int(k), len(labels)))
    chosen = np.argpartition(-sims, k - 1)[:k]
    chosen = chosen[np.argsort(-sims[chosen])]
    rate = round(float(labels[chosen].mean()), 4)
    snri = round(min((rate + 0.15) * 0.9, 1.0), 4)
    twins = []
    for idx in chosen[: settings.RAG_TOP_K]:
        row = meta.iloc[int(idx)]
        interventions = [part.strip() for part in str(row.get("interventions", "")).split("|") if part.strip()]
        score = float(min(max(sims[idx], 0.0), 1.0))
        twins.append(TwinPatient(
            twin_id=str(int(row["encounter_id"])),
            age=int(float(row["age"])),
            primary_diagnosis=str(row["primary_diagnosis"]),
            diagnosis_group=str(row["primary_diagnosis"]),
            similarity_score=round(score, 4),
            was_readmitted=bool(int(row["readmitted_binary"])),
            successful_interventions=interventions,
        ))
    return TwinPatientResponse(
        patient_id=payload.patient_id,
        rag_readmission_rate=rate,
        semantic_neighborhood_risk_index=snri,
        sdoh_flag=_extract_sdoh_flags(payload.clinical_note)[0],
        twins=twins,
    )


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
    import os

    import joblib

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    meta_path = os.path.join(MODELS_DIR, "feature_names.json")

    if not all(os.path.exists(p) for p in [model_path, prep_path, meta_path]):
        logger.warning("ML model artifacts not found at %s — using deterministic mock.", MODELS_DIR)
        return None

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
    except Exception as exc:
        logger.warning("ML artifacts failed to load (%s) — using deterministic mock.", exc)
        return None
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
        care_plan, plan_source, rationale = _compose_care(
            risk_category, raw_score, sdoh_flags, shap_features, [], _a1c(payload.a1c_result)
        )
        cms_saved = round(15_400 * raw_score, 2) if risk_category == RiskCategory.HIGH else None
        return PredictResponse(
            patient_id=payload.patient_id,
            risk_score=round(raw_score, 4),
            risk_category=risk_category,
            shap_features=shap_features,
            sdoh_flags=sdoh_flags,
            care_plan=care_plan,
            plan_source=plan_source,
            clinical_rationale=rationale,
            cms_penalty_saved_usd=cms_saved,
        )

    # ── LIVE: real RF_RAG model inference ─────────────────────────────────────
    model = artifacts["model"]
    prep = artifacts["preprocessor"]
    meta = artifacts["meta"]

    # 1. Same 34-column row the preprocessor was fit on, then leakage-safe neighbors.
    frame = _model_frame(payload)
    rag_response = run_rag_retrieval(payload)
    rag_rate = rag_response.rag_readmission_rate

    # 2. Transform and append the neighborhood rate when the exported model uses it.
    X_tab = prep.transform(frame)
    if hasattr(X_tab, "toarray"):
        X_tab = X_tab.toarray()
    X_input = np.hstack([X_tab, np.array([[rag_rate]], dtype=float)]) if meta.get("uses_rag") else np.asarray(X_tab)

    # 4. Predict probability
    raw_score = float(model.predict_proba(X_input)[:, 1][0])

    # F1-optimal cutoffs near 0.12 belong to calibrated scores (base rate ~11%).
    # Using that cutoff as "high risk" would flag about a third of discharges.
    # High is the top of the score range (~2x base rate); moderate is above the F1 cut.
    f1_cut = float(meta.get("decision_threshold") or 0.65)
    if f1_cut >= 0.40:
        high_cut, moderate_cut = f1_cut, min(0.40, f1_cut)
    else:
        high_cut, moderate_cut = 0.20, f1_cut
    risk_category = (
        RiskCategory.HIGH if raw_score >= high_cut
        else RiskCategory.MODERATE if raw_score >= moderate_cut
        else RiskCategory.LOW
    )

    # 5. SHAP values — TreeExplainer for RF; graceful fallback on failure
    shap_features: list[SHAPFeature] = []
    try:
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[0, :, 1]
        elif sv.ndim == 2:
            sv = sv[0]
        feature_names = meta["feature_names"]
        row = np.asarray(X_input)[0]
        all_shap = []
        for i in range(len(feature_names)):
            # A zero one-hot column is an option the clinician did not select.
            if feature_names[i].startswith("cat__") and float(row[i]) == 0.0:
                continue
            all_shap.append(SHAPFeature(feature=feature_names[i], impact=round(float(sv[i]), 4)))
        all_shap.sort(key=lambda x: abs(x.impact), reverse=True)
        shap_features = all_shap[:5]
    except Exception as exc:
        logger.warning("SHAP explanation failed (%s); returning empty list.", exc)

    sdoh_flags = _extract_sdoh_flags(payload.clinical_note)
    twin_interventions: list[str] = []
    for twin in rag_response.twins:
        if not twin.was_readmitted:
            for item in twin.successful_interventions:
                if item not in twin_interventions:
                    twin_interventions.append(item)

    care_plan, plan_source, rationale = _compose_care(
        risk_category, raw_score, sdoh_flags, shap_features, twin_interventions, _a1c(payload.a1c_result)
    )

    cms_saved = round(15_400 * raw_score, 2) if risk_category == RiskCategory.HIGH else None

    return PredictResponse(
        patient_id=payload.patient_id,
        risk_score=round(raw_score, 4),
        risk_category=risk_category,
        shap_features=shap_features,
        sdoh_flags=sdoh_flags,
        care_plan=care_plan,
        plan_source=plan_source,
        clinical_rationale=rationale,
        cms_penalty_saved_usd=cms_saved,
    )



# ─────────────────────────────────────────────────────────────
# Twin-Patient RAG Service
# ─────────────────────────────────────────────────────────────

def run_rag_retrieval(payload: PatientEncounter) -> TwinPatientResponse:
    """
    Prefer the leakage-safe training index. Fall back to live Qdrant, then
    to the deterministic mock cohort.
    """
    artifacts = _get_ml_artifacts()
    meta = artifacts["meta"] if artifacts else {}
    safe = _safe_rag_retrieval(payload, _rag_k(meta))
    if safe is not None:
        return safe
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
            diagnosis_group=str(r.payload.get("primary_diagnosis", payload.diag_1_group)),
            similarity_score=round(float(min(max(r.score, 0.0), 1.0)), 4),
            was_readmitted=bool(r.payload.get("readmitted_binary", 0)),
            successful_interventions=r.payload.get("interventions", []),
        )
        for i, r in enumerate(results)
    ]

    readmitted_count = sum(1 for t in twins if t.was_readmitted)
    rag_rate = round(readmitted_count / max(len(twins), 1), 4)
    snri = round(min((rag_rate + 0.15) * 0.9, 1.0), 4)

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
            diagnosis_group=payload.diag_1_group,
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
            diagnosis_group=payload.diag_1_group,
            similarity_score=0.88,
            was_readmitted=True,
            successful_interventions=[],
        ),
        TwinPatient(
            twin_id="P-10213",
            age=payload.age + 1,
            primary_diagnosis=payload.primary_diagnosis,
            diagnosis_group=payload.diag_1_group,
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
    snri = round(min((rag_rate + 0.15) * 0.9, 1.0), 4)

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
  - top_shap_features: list[{feature, impact}] — top risk-model drivers
  - twin_interventions: list[str] — successful interventions from similar non-readmitted patients

Your task: Generate a concise, evidence-based 3-point prescriptive discharge plan.
If a1c_result is Not_Tested, one action must tell the clinician to order HbA1c before discharge.
If a1c_result is >7 or >8, one action must arrange follow-up for the elevated HbA1c.

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


def _plain_feature(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")


def _rule_rationale(shap_features: list[SHAPFeature], sdoh_flags: list[str], a1c_result: str) -> str:
    driver = _plain_feature(shap_features[0].feature) if shap_features else "the chart"
    sentence = f"The strongest chart driver is {driver}."
    if a1c_result == "Not_Tested":
        sentence += " HbA1c was not measured during this stay."
    elif a1c_result in {">7", ">8"}:
        sentence += " HbA1c was elevated."
    if sdoh_flags and not sdoh_flags[0].lower().startswith("no sdoh"):
        sentence += f" The note also shows {sdoh_flags[0].rstrip('.').lower()}."
    return sentence


def _a1c_action(a1c_result: str) -> str | None:
    if a1c_result == "Not_Tested":
        return "Order an HbA1c test before discharge."
    if a1c_result in {">7", ">8"}:
        return "Arrange prompt diabetes follow-up for an elevated HbA1c."
    return None


def _apply_a1c(steps: list[str], a1c_result: str) -> list[str]:
    action = _a1c_action(a1c_result)
    if action is None:
        return _cap_plan(steps)
    rest = [step for step in steps if "a1c" not in step.lower() and "hba1c" not in step.lower()]
    return _cap_plan([action, *rest])


def _cap_plan(steps: list[str]) -> list[str]:
    cleaned: list[str] = []
    for step in steps:
        text = step.replace("⚠️ PRIORITY: ", "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:3]


def _compose_care(
    risk: RiskCategory,
    risk_score: float,
    sdoh_flags: list[str],
    shap_features: list[SHAPFeature],
    twin_interventions: list[str],
    a1c_result: str,
) -> tuple[list[str], PlanSource, str]:
    rationale = _rule_rationale(shap_features, sdoh_flags, a1c_result)
    if settings.LIVE_LLM_ENABLED:
        drafted = _generate_llm_care_plan(
            risk_score, sdoh_flags, shap_features, twin_interventions, a1c_result
        )
        if drafted is not None:
            steps, drafted_rationale = drafted
            return _apply_a1c(steps, a1c_result), PlanSource.GEMINI, drafted_rationale or rationale
    steps = _apply_a1c(_generate_rule_care_plan(risk, sdoh_flags), a1c_result)
    return steps, PlanSource.RULES, rationale


def _generate_llm_care_plan(
    risk_score: float,
    sdoh_flags: list[str],
    shap_features: list[SHAPFeature],
    twin_interventions: list[str] | None = None,
    a1c_result: str = "Not_Tested",
) -> tuple[list[str], str] | None:
    """Call Gemini. None means the caller should use the rule checklist."""
    try:
        client = _get_gemini_client()
        interventions = twin_interventions or [
            "72-hr telehealth follow-up",
            "Pharmacist medication reconciliation",
            "Home nursing referral",
        ]
        user_payload = json.dumps({
            "patient_risk_score": risk_score,
            "sdoh_flags": sdoh_flags,
            "top_shap_features": [{"feature": f.feature, "impact": f.impact} for f in shap_features[:3]],
            "twin_interventions": interventions,
            "a1c_result": a1c_result,
        })

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=f"{_LLM_SYSTEM_PROMPT}\n\nPatient Data:\n{user_payload}",
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            raw = raw.removeprefix("json")

        parsed = json.loads(raw)
        steps = _cap_plan(parsed.get("interventions") or [])
        if not steps:
            return None
        rationale = str(parsed.get("clinical_rationale") or "").strip()
        return steps, rationale or _rule_rationale(shap_features, sdoh_flags, a1c_result)

    except Exception as exc:
        logger.warning("Gemini call failed (%s), falling back to rule-based care plan.", exc)
        return None


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
    """Rule-based care plan. Social needs come first. The caller keeps three steps."""
    steps: list[str] = []
    if risk == RiskCategory.HIGH:
        steps.append("Start a 72-hour post-discharge telehealth check-in.")
    if any("transport" in f.lower() for f in sdoh_flags):
        steps.append("Arrange non-emergency medical transport for follow-up appointments.")
    if any("financial" in f.lower() or "uninsured" in f.lower() for f in sdoh_flags):
        steps.append("Refer to the hospital financial navigator for prescription cost assistance.")
    if any("isolation" in f.lower() or "alone" in f.lower() for f in sdoh_flags):
        steps.append("Arrange a daily wellness check because the patient lives alone.")
    steps.append("Conduct pharmacist-led medication reconciliation before discharge.")
    steps.append("Schedule a telephone follow-up within 7 days of discharge.")
    return steps
