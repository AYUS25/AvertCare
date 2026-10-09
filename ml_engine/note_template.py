"""Discharge-time clinical note template.

The note is the text that gets embedded for Twin-Patient retrieval.
It may contain only fields that exist at discharge prediction time.
It must not mention readmission, the original readmitted column, or
readmitted_binary.
"""

from __future__ import annotations


def _as_int(value, default=0) -> int:
    try:
        if value is None or (isinstance(value, float) and value != value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def build_discharge_note(row) -> str:
    """Build a deterministic note from one encounter row or mapping."""
    def get(key, default="Unknown"):
        if hasattr(row, "get"):
            value = row.get(key, default)
        else:
            value = getattr(row, key, default)
        if value is None or (isinstance(value, float) and value != value):
            return default
        return value

    note = (
        f"Patient is a {_as_int(get('age_numeric', 65), 65)}-year-old "
        f"{get('gender')} {get('race')} patient. "
        f"Admission type {get('admission_type_id')}, source {get('admission_source_id')}. "
        f"Primary diagnosis group {get('diag_1_group')}. "
        f"Secondary diagnosis group {get('diag_2_group')}. "
        f"Third diagnosis group {get('diag_3_group')}. "
        f"Length of stay {_as_int(get('time_in_hospital', 0))} days. "
        f"Prior inpatient visits {_as_int(get('number_inpatient', 0))}. "
        f"Prior emergency visits {_as_int(get('number_emergency', 0))}. "
        f"Prior outpatient visits {_as_int(get('number_outpatient', 0))}. "
        f"Lab procedures {_as_int(get('num_lab_procedures', 0))}. "
        f"Procedures {_as_int(get('num_procedures', 0))}. "
        f"Diagnoses recorded {_as_int(get('number_diagnoses', 0))}. "
        f"Discharge medications {_as_int(get('num_medications', 0))}. "
        f"Active diabetes medications {_as_int(get('num_meds_active', 0))}. "
        f"Medication changes {_as_int(get('num_med_changes', 0))}. "
        f"Diabetes medication prescribed {get('diabetesMed')}. "
        f"Regimen change flag {get('change')}. "
        f"Insulin status {get('insulin')}. "
        f"Metformin status {get('metformin')}. "
        f"A1C result {get('A1Cresult')}. "
        f"Glucose serum result {get('max_glu_serum')}. "
        f"Discharge disposition {get('discharge_disposition_id')}. "
        f"Medical specialty {get('medical_specialty')}."
    )
    lowered = note.lower()
    if "readmit" in lowered or "readmission" in lowered:
        raise ValueError("Discharge note template contains an outcome phrase.")
    return note


def historical_interventions(row) -> list[str]:
    """Interventions to display for a past encounter that was not readmitted.

    Chosen from that encounter's discharge-time fields. The caller's label
    decides whether these are shown. They are not model inputs.
    """
    items: list[str] = []
    if _as_int(row.get("number_inpatient", 0)) >= 1 or _as_int(row.get("time_in_hospital", 0)) >= 5:
        items.append("72-hr post-discharge telehealth follow-up scheduled")
    if _as_int(row.get("num_medications", 0)) >= 8 or _as_int(row.get("num_med_changes", 0)) >= 1:
        items.append("Pharmacist medication reconciliation completed")
    if _as_int(row.get("number_emergency", 0)) >= 1:
        items.append("Home nursing visit arranged within 48 hrs")
    a1c = str(row.get("A1Cresult", ""))
    if a1c in {">7", ">8"} or str(row.get("diabetesMed", "")) == "Yes":
        items.append("Diabetes self-management education completed")
    if str(row.get("payer_code", "")) in {"Unknown", "?", ""}:
        items.append("Social work referral for medication-cost assistance")
    if not items:
        items.append("Primary care physician notified of discharge")
    return items[:3]
