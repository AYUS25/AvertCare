"""
AvertCare Backend · Pydantic v2 Request / Response Schemas

All payloads mirror HL7 FHIR Encounter / DocumentReference structure
so swapping mock → real FHIR ingestion is a 1-line change.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────
# Inbound Patient Payload
# ─────────────────────────────────────────────────────────────

class PatientEncounter(BaseModel):
    """FHIR-inspired inbound payload for a single patient encounter."""

    patient_id: str = Field(..., min_length=1, description="Unique patient identifier (MRN or FHIR resource ID)")
    age: int = Field(..., ge=0, le=130, description="Patient age in years")
    primary_diagnosis: str = Field(..., min_length=2, description="ICD-10 primary diagnosis description")
    time_in_hospital: int = Field(..., ge=0, description="Length of stay in days")
    num_prior_admissions: int = Field(default=0, ge=0, description="Count of admissions in the past 12 months")
    num_medications: int = Field(default=0, ge=0, description="Active medication count at discharge")
    clinical_note: str = Field(..., min_length=10, description="Unstructured discharge summary / clinical note")
    # Extended clinical fields — populated by the UI dropdowns; used directly by RF_RAG model
    a1c_result: str = Field(default="None", description="A1C lab result: None, Norm, >7, >8")
    number_inpatient: int = Field(default=0, ge=0, description="Number of inpatient visits in the prior year")
    number_emergency: int = Field(default=0, ge=0, description="Emergency visits in the prior year")
    number_diagnoses: int = Field(default=1, ge=1, le=16, description="Diagnoses recorded this stay")
    diabetes_med: str = Field(default="No", description="Whether patient is on diabetes medication: Yes or No")
    gender: str = Field(default="Unknown", description="Patient gender")
    discharge_disposition: str = Field(default="1", description="Discharge disposition code used by the trained model")
    diag_1_group: str = Field(default="Diabetes", description="Primary diagnosis group")

    @field_validator("primary_diagnosis")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ─────────────────────────────────────────────────────────────
# Predict Endpoint Response
# ─────────────────────────────────────────────────────────────

class RiskCategory(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"


class PlanSource(str, Enum):
    GEMINI = "gemini"
    RULES = "rules"


class SHAPFeature(BaseModel):
    feature: str
    impact: float


class PredictResponse(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Readmission probability [0, 1]")
    risk_category: RiskCategory
    shap_features: list[SHAPFeature] = Field(description="Top SHAP drivers (descending impact)")
    sdoh_flags: list[str] = Field(description="Extracted Social Determinants of Health flags")
    care_plan: list[str] = Field(description="Up to three discharge actions")
    plan_source: PlanSource = Field(description="gemini when the assistant wrote the list, otherwise rules")
    clinical_rationale: str = Field(description="One sentence a clinician can read under the checklist")
    cms_penalty_saved_usd: float | None = Field(default=None, description="Estimated CMS penalty avoidance in USD. Null unless risk is high.")


# ─────────────────────────────────────────────────────────────
# Twin-Patient RAG Response
# ─────────────────────────────────────────────────────────────

class TwinPatient(BaseModel):
    twin_id: str
    age: int
    primary_diagnosis: str
    diagnosis_group: str = Field(description="Clinical group used for matching, such as Diabetes or Circulatory")
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    was_readmitted: bool
    successful_interventions: list[str]


class TwinPatientResponse(BaseModel):
    patient_id: str
    rag_readmission_rate: float = Field(..., ge=0.0, le=1.0, description="Readmission rate among retrieved twin cohort")
    semantic_neighborhood_risk_index: float = Field(..., ge=0.0, le=1.0, description="RAGrisk score")
    sdoh_flag: str = Field(description="Primary SDoH signal extracted from the clinical note")
    twins: list[TwinPatient]
