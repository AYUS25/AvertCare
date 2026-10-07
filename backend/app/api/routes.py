"""
AvertCare Backend · API Routes

Paper-thin router — all logic delegated to services.py.
Auth dependency is a dev-bypass passthrough when FIREBASE_PROJECT_ID is unset.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app import services
from app.core.auth import require_auth
from app.schemas import PatientEncounter, PredictResponse, TwinPatientResponse

router = APIRouter(prefix="/api", tags=["Clinical AI"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid readmission risk prediction",
)
async def predict_readmission(
    payload: PatientEncounter,
    _token: dict = Depends(require_auth),
) -> PredictResponse:
    try:
        return services.run_prediction(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@router.post(
    "/twin-patients",
    response_model=TwinPatientResponse,
    status_code=status.HTTP_200_OK,
    summary="RAG twin-patient retrieval",
)
async def retrieve_twin_patients(
    payload: PatientEncounter,
    _token: dict = Depends(require_auth),
) -> TwinPatientResponse:
    try:
        return services.run_rag_retrieval(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG retrieval error: {exc}") from exc
