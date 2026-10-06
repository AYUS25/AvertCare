"""
AvertCare Backend · API Routes

Paper-thin router — all logic delegated to services.py.
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas import PatientEncounter, PredictResponse, TwinPatientResponse
from app import services

router = APIRouter(prefix="/api", tags=["Clinical AI"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid readmission risk prediction",
    description=(
        "Accepts a patient encounter payload and returns a readmission risk score, "
        "SHAP explainability features, SDoH flags, and a GenAI-generated care plan."
    ),
)
async def predict_readmission(payload: PatientEncounter) -> PredictResponse:
    try:
        return services.run_prediction(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc


@router.post(
    "/twin-patients",
    response_model=TwinPatientResponse,
    status_code=status.HTTP_200_OK,
    summary="RAG twin-patient retrieval",
    description=(
        "Embeds the patient's clinical note, queries the vector DB for semantically "
        "similar historical encounters, and returns the RAG risk index with successful interventions."
    ),
)
async def retrieve_twin_patients(payload: PatientEncounter) -> TwinPatientResponse:
    try:
        return services.run_rag_retrieval(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG retrieval error: {exc}",
        ) from exc
