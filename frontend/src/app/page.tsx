"use client";

import { useState } from "react";
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  FileText, 
  Users,
  ShieldAlert,
  Loader2,
  DollarSign,
  LogOut
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

// Types matching our Backend Pydantic Schemas
type MedStatus = "No" | "Steady" | "Up" | "Down";

type PatientEncounter = {
  patient_id: string;
  age: number;
  primary_diagnosis: string;
  time_in_hospital: number;
  num_prior_admissions: number;
  num_medications: number;
  clinical_note: string;
  // Extended clinical fields for richer ML inference
  a1c_result: "None" | ">7" | ">8" | "Norm";
  number_inpatient: number;
  number_emergency: number;
  number_diagnoses: number;
  diabetes_med: "Yes" | "No";
  gender: "Male" | "Female" | "Unknown";
  discharge_disposition: string;
  diag_1_group: string;
  num_lab_procedures: number;
  num_procedures: number;
  number_outpatient: number;
  admission_type_id: string;
  admission_source_id: string;
  change: "No" | "Ch";
  diag_2_group: string;
  diag_3_group: string;
  max_glu_serum: "Not_Tested" | "Norm" | ">200" | ">300";
  medical_specialty: string;
  payer_code: string;
  race: string;
  metformin: MedStatus;
  repaglinide: MedStatus;
  glimepiride: MedStatus;
  glipizide: MedStatus;
  glyburide: MedStatus;
  pioglitazone: MedStatus;
  rosiglitazone: MedStatus;
  insulin: MedStatus;
};

const DIAG_GROUPS = [
  "Diabetes",
  "Circulatory",
  "Respiratory",
  "Digestive",
  "Genitourinary",
  "Neoplasms",
  "Musculoskeletal",
  "Injury",
  "Other",
] as const;

const DISCHARGE_OPTIONS = [
  { value: "1", label: "Home" },
  { value: "6", label: "Home with home health" },
  { value: "3", label: "Skilled nursing facility" },
  { value: "4", label: "Intermediate care" },
  { value: "22", label: "Rehab facility" },
  { value: "2", label: "Another short-term hospital" },
  { value: "5", label: "Other inpatient facility" },
  { value: "7", label: "Left against medical advice" },
  { value: "Other", label: "Other / not listed" },
] as const;

function shapLabel(feature: string): string {
  const named: Record<string, string> = {
    "num__number_inpatient": "Prior inpatient visits",
    "num__total_prior_visits": "Total prior visits",
    "num__number_emergency": "Emergency visits",
    "num__number_diagnoses": "Diagnosis count",
    "num__time_in_hospital": "Length of stay",
    "num__age_numeric": "Age",
    "num__num_medications": "Medication count",
    "num__num_meds_active": "Active diabetes meds",
    "num__num_lab_procedures": "Lab procedures",
    "rag_readmit_rate": "Similar-patient readmit rate",
    "cat__discharge_disposition_id_1": "Discharged home",
    "cat__discharge_disposition_id_2": "Transferred to another hospital",
    "cat__discharge_disposition_id_3": "Discharged to skilled nursing",
    "cat__discharge_disposition_id_4": "Discharged to intermediate care",
    "cat__discharge_disposition_id_5": "Discharged to other inpatient care",
    "cat__discharge_disposition_id_6": "Home with home health",
    "cat__discharge_disposition_id_7": "Left against medical advice",
    "cat__discharge_disposition_id_22": "Discharged to rehab",
    "cat__discharge_disposition_id_Other": "Other discharge destination",
    "cat__diag_1_group_Diabetes": "Primary diagnosis: diabetes",
    "cat__diag_1_group_Circulatory": "Primary diagnosis: circulatory",
    "cat__diag_1_group_Respiratory": "Primary diagnosis: respiratory",
    "cat__payer_code_Unknown": "Payer unknown",
  };
  return named[feature] ?? feature.replace(/^num__|^cat__/, "").replaceAll("_", " ");
}

type SHAPFeature = {
  feature: string;
  impact: number;
};

type PredictResponse = {
  patient_id: string;
  risk_score: number;
  risk_category: "Low Risk" | "Moderate Risk" | "High Risk";
  shap_features: SHAPFeature[];
  sdoh_flags: string[];
  care_plan: string[];
  plan_source?: "gemini" | "rules";
  clinical_rationale?: string;
  cms_penalty_saved_usd?: number | null;
};

type TwinPatient = {
  twin_id: string;
  age: number;
  primary_diagnosis: string;
  diagnosis_group?: string;
  similarity_score: number;
  was_readmitted: boolean;
  successful_interventions: string[];
};

type TwinPatientResponse = {
  patient_id: string;
  rag_readmission_rate: number;
  semantic_neighborhood_risk_index: number;
  sdoh_flag: string;
  twins: TwinPatient[];
};

const fieldClass =
  "w-full bg-[hsl(var(--muted))] border border-[hsl(var(--border))] rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-[hsl(var(--primary))]";

const NOTE_HINT =
  "Check the chart. The note must be at least one short sentence, and the diagnosis must be filled in.";

const MED_STATUSES: MedStatus[] = ["No", "Steady", "Up", "Down"];
const DRUG_FIELDS: { key: keyof PatientEncounter; label: string }[] = [
  { key: "insulin", label: "Insulin" },
  { key: "metformin", label: "Metformin" },
  { key: "glipizide", label: "Glipizide" },
  { key: "glyburide", label: "Glyburide" },
  { key: "glimepiride", label: "Glimepiride" },
  { key: "pioglitazone", label: "Pioglitazone" },
  { key: "rosiglitazone", label: "Rosiglitazone" },
  { key: "repaglinide", label: "Repaglinide" },
];
const SPECIALTIES = [
  "Unknown", "InternalMedicine", "Cardiology", "Emergency/Trauma",
  "Family/GeneralPractice", "Surgery-General", "Orthopedics",
  "Orthopedics-Reconstructive", "Nephrology", "Radiologist", "Other",
];
const RACES = ["Unknown", "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"];
const PAYERS = ["Unknown", "MC", "HM", "SP", "BC", "MD", "CP", "UN", "CM", "OG", "PO", "DM", "CH", "WC", "OT", "MP", "SI"];
const ADMISSION_TYPES = [
  { value: "1", label: "1 Emergency" },
  { value: "2", label: "2 Urgent" },
  { value: "3", label: "3 Elective" },
  { value: "5", label: "5 Not available" },
  { value: "6", label: "6 Not mapped" },
  { value: "Other", label: "Other" },
];
const ADMISSION_SOURCES = [
  { value: "7", label: "7 Emergency room" },
  { value: "1", label: "1 Physician referral" },
  { value: "2", label: "2 Clinic referral" },
  { value: "4", label: "4 Transfer from a hospital" },
  { value: "5", label: "5 Transfer from a skilled nursing facility" },
  { value: "6", label: "6 Transfer from another facility" },
  { value: "17", label: "17 Not available" },
  { value: "Other", label: "Other" },
];

const TRAINED_DEFAULTS = {
  num_lab_procedures: 40,
  num_procedures: 0,
  number_outpatient: 0,
  admission_type_id: "Other",
  admission_source_id: "Other",
  change: "No" as const,
  diag_2_group: "Other",
  diag_3_group: "Other",
  max_glu_serum: "Not_Tested" as const,
  medical_specialty: "Unknown",
  payer_code: "Unknown",
  race: "Unknown",
  metformin: "No" as const,
  repaglinide: "No" as const,
  glimepiride: "No" as const,
  glipizide: "No" as const,
  glyburide: "No" as const,
  pioglitazone: "No" as const,
  rosiglitazone: "No" as const,
  insulin: "Steady" as const,
};

const QUIET_DISCHARGE: PatientEncounter = {
  patient_id: "MRN-782910",
  age: 54,
  primary_diagnosis: "Type 2 diabetes",
  time_in_hospital: 3,
  num_prior_admissions: 0,
  num_medications: 6,
  clinical_note:
    "Patient lives with spouse. Blood glucose stable. Transport home is arranged. Follow-up already booked.",
  a1c_result: "Norm",
  number_inpatient: 0,
  number_emergency: 0,
  number_diagnoses: 3,
  diabetes_med: "Yes",
  gender: "Female",
  discharge_disposition: "1",
  diag_1_group: "Diabetes",
  ...TRAINED_DEFAULTS,
};

const SAMPLE_CHART: PatientEncounter = {
  patient_id: "MRN-782910",
  age: 67,
  primary_diagnosis: "Type 2 Diabetes with complications",
  time_in_hospital: 7,
  num_prior_admissions: 2,
  num_medications: 8,
  clinical_note:
    "Patient lives alone and has expressed concerns about inability to afford insulin. Polypharmacy noted. Blood glucose stabilizing. Discharge planned for tomorrow. Transport home is currently unarranged.",
  a1c_result: ">8",
  number_inpatient: 2,
  number_emergency: 0,
  number_diagnoses: 5,
  diabetes_med: "Yes",
  gender: "Female",
  discharge_disposition: "1",
  diag_1_group: "Diabetes",
  ...TRAINED_DEFAULTS,
  insulin: "Up",
};

const HIGH_UTIL_DISCHARGE: PatientEncounter = {
  patient_id: "MRN-782910",
  age: 74,
  primary_diagnosis: "Heart failure",
  time_in_hospital: 8,
  num_prior_admissions: 4,
  num_medications: 12,
  clinical_note:
    "Patient lives alone and cannot afford insulin. No transport. Polypharmacy noted.",
  a1c_result: ">8",
  number_inpatient: 4,
  number_emergency: 2,
  number_diagnoses: 8,
  diabetes_med: "Yes",
  gender: "Male",
  discharge_disposition: "3",
  diag_1_group: "Circulatory",
  ...TRAINED_DEFAULTS,
  number_outpatient: 2,
  num_lab_procedures: 55,
  num_procedures: 1,
  admission_type_id: "1",
  admission_source_id: "7",
  change: "Ch",
  diag_2_group: "Diabetes",
  max_glu_serum: ">200",
  medical_specialty: "Cardiology",
  insulin: "Up",
};

function messageForStatus(status: number): string {
  if (status === 401) return "Sign in again. The login expired.";
  if (status === 422) return NOTE_HINT;
  return String(status);
}

function riskTone(category: string): { bar: string; text: string } {
  if (category === "High Risk") return { bar: "bg-rose-500", text: "text-rose-400" };
  if (category === "Moderate Risk") return { bar: "bg-amber-500", text: "text-amber-400" };
  return { bar: "bg-emerald-500", text: "text-emerald-400" };
}

function clampInt(raw: string, min: number, max: number): number | null {
  if (raw.trim() === "") return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return Math.min(max, Math.max(min, Math.trunc(n)));
}

export default function Dashboard() {
  const [patient, setPatient] = useState<PatientEncounter>(SAMPLE_CHART);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [reviewStep, setReviewStep] = useState<string | null>(null);
  const [predictData, setPredictData] = useState<PredictResponse | null>(null);
  const [ragData, setRagData] = useState<TwinPatientResponse | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [ragError, setRagError] = useState<string | null>(null);

  const { token, logout, user } = useAuth();
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const noteTooShort = patient.clinical_note.trim().length < 10;

  const applyPreset = (next: PatientEncounter) => {
    setPatient(next);
    setPredictData(null);
    setRagData(null);
    setPredictError(null);
    setRagError(null);
  };

  const handleAnalyze = async () => {
    if (noteTooShort) return;

    setIsAnalyzing(true);
    setPredictError(null);
    setRagError(null);
    try {
      let authToken = token;
      if (user) {
        try {
          authToken = await user.getIdToken();
        } catch {
          const expired = "Sign in again. The login expired.";
          setPredictData(null);
          setRagData(null);
          setPredictError(expired);
          setRagError(expired);
          return;
        }
      }

      const headers = {
        "Content-Type": "application/json",
        ...(authToken && { Authorization: `Bearer ${authToken}` }),
      };
      const body = JSON.stringify({
        ...patient,
        num_prior_admissions: patient.number_inpatient,
      });

      setReviewStep("Searching similar stays…");
      const ragResult = await Promise.allSettled([
        fetch(`${API_URL}/api/twin-patients`, { method: "POST", headers, body }),
      ]).then((results) => results[0]);

      setReviewStep("Scoring this discharge…");
      const predictResult = await Promise.allSettled([
        fetch(`${API_URL}/api/predict`, { method: "POST", headers, body }),
      ]).then((results) => results[0]);

      if (predictResult.status === "fulfilled") {
        const res = predictResult.value;
        if (res.ok) {
          setPredictData(await res.json());
        } else {
          setPredictData(null);
          setPredictError(messageForStatus(res.status));
        }
      } else {
        setPredictData(null);
        setPredictError("The API is not reachable.");
      }

      if (ragResult.status === "fulfilled") {
        const res = ragResult.value;
        if (res.ok) {
          setRagData(await res.json());
        } else {
          setRagData(null);
          setRagError(messageForStatus(res.status));
        }
      } else {
        setRagData(null);
        setRagError("The API is not reachable.");
      }
    } finally {
      setReviewStep(null);
      setIsAnalyzing(false);
    }
  };

  const tone = predictData ? riskTone(predictData.risk_category) : null;
  const noSocialSignals =
    !!predictData &&
    (predictData.sdoh_flags.length === 0 ||
      (predictData.sdoh_flags.length === 1 &&
        predictData.sdoh_flags[0] === "No SDoH signals detected"));
  const maxAbsImpact = predictData
    ? Math.max(0, ...predictData.shap_features.map((feat) => Math.abs(feat.impact)))
    : 0;
  const hasOutcome = predictData || ragData || predictError || ragError;

  return (
    <div className="min-h-screen bg-[hsl(var(--background))] text-[hsl(var(--foreground))] p-6 font-sans">
      <header className="flex items-center justify-between pb-6 border-b border-[hsl(var(--border))] mb-8">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] rounded-lg">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">AvertCare</h1>
            <p className="text-sm text-[hsl(var(--muted-foreground))]">Prescriptive Clinical Decision Support</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="px-4 py-2 text-sm font-medium border rounded-full border-[hsl(var(--border))] text-[hsl(var(--muted-foreground))]">
            Sample chart
          </div>
          {user?.email && (
            <span className="text-sm text-[hsl(var(--muted-foreground))]">{user.email}</span>
          )}
          {user && (
            <button 
              onClick={logout}
              className="p-2 text-[hsl(var(--muted-foreground))] hover:text-rose-400 hover:bg-rose-400/10 rounded-lg transition-colors flex items-center gap-2"
              title="Sign Out"
            >
              <LogOut className="w-5 h-5" />
            </button>
          )}
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-8 max-w-[1600px] mx-auto">
        
        {/* Left Column: Patient Profile */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium flex items-center gap-2">
                <Users className="w-5 h-5 text-[hsl(var(--muted-foreground))]" />
                Patient Encounter
              </h2>
              <span className="text-xs bg-[hsl(var(--accent))] text-[hsl(var(--accent-foreground))] px-2 py-1 rounded-md font-mono">
                {patient.patient_id}
              </span>
            </div>

            <div className="flex gap-2 mb-6">
              <button
                type="button"
                onClick={() => applyPreset(QUIET_DISCHARGE)}
                className="flex-1 text-sm border border-[hsl(var(--border))] rounded-lg px-3 py-2 hover:bg-[hsl(var(--muted))]"
              >
                Quiet discharge
              </button>
              <button
                type="button"
                onClick={() => applyPreset(HIGH_UTIL_DISCHARGE)}
                className="flex-1 text-sm border border-[hsl(var(--border))] rounded-lg px-3 py-2 hover:bg-[hsl(var(--muted))]"
              >
                High-utilization discharge
              </button>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Age</label>
                  <input
                    type="number"
                    min={0}
                    max={130}
                    className={fieldClass}
                    value={patient.age}
                    onChange={(e) => {
                      const age = clampInt(e.target.value, 0, 130);
                      if (age !== null) setPatient({ ...patient, age });
                    }}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Length of Stay</label>
                  <input
                    type="number"
                    min={0}
                    max={30}
                    className={fieldClass}
                    value={patient.time_in_hospital}
                    onChange={(e) => {
                      const time_in_hospital = clampInt(e.target.value, 0, 30);
                      if (time_in_hospital !== null) setPatient({ ...patient, time_in_hospital });
                    }}
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Primary Diagnosis</label>
                <input
                  type="text"
                  minLength={2}
                  className={fieldClass}
                  value={patient.primary_diagnosis}
                  onChange={(e) => setPatient({ ...patient, primary_diagnosis: e.target.value })}
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Medication count</label>
                <input
                  type="number"
                  min={0}
                  max={40}
                  className={fieldClass}
                  value={patient.num_medications}
                  onChange={(e) => {
                    const num_medications = clampInt(e.target.value, 0, 40);
                    if (num_medications !== null) setPatient({ ...patient, num_medications });
                  }}
                />
              </div>

              <div className="space-y-2 pt-4">
                <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">
                  Clinical Lab & Medication Profile
                </label>
                <div className="grid grid-cols-2 gap-3">

                  {/* A1C Result */}
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">A1C Result</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.a1c_result}
                      onChange={(e) => setPatient({...patient, a1c_result: e.target.value as PatientEncounter["a1c_result"]})}
                    >
                      <option value="None">Not Tested</option>
                      <option value="Norm">Normal (&lt;7)</option>
                      <option value=">7">High (&gt;7)</option>
                      <option value=">8">Very High (&gt;8)</option>
                    </select>
                  </div>

                  {/* Gender */}
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Gender</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.gender}
                      onChange={(e) => setPatient({...patient, gender: e.target.value as PatientEncounter["gender"]})}
                    >
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Unknown">Unknown</option>
                    </select>
                  </div>

                  {/* Diabetes Meds */}
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Diabetes Meds</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.diabetes_med}
                      onChange={(e) => setPatient({...patient, diabetes_med: e.target.value as "Yes" | "No"})}
                    >
                      <option value="Yes">Yes</option>
                      <option value="No">No</option>
                    </select>
                  </div>

                  {/* Number of Inpatient Visits */}
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Inpatient Visits (1yr)</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.number_inpatient}
                      onChange={(e) => {
                        const number_inpatient = parseInt(e.target.value);
                        setPatient({
                          ...patient,
                          number_inpatient,
                          num_prior_admissions: number_inpatient,
                        });
                      }}
                    >
                      {[0,1,2,3,4,5,6,7,8].map(n => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Emergency Visits (1yr)</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.number_emergency}
                      onChange={(e) => setPatient({...patient, number_emergency: parseInt(e.target.value)})}
                    >
                      {[0,1,2,3,4,5,6,7,8].map(n => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Diagnosis Count</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.number_diagnoses}
                      onChange={(e) => setPatient({...patient, number_diagnoses: parseInt(e.target.value)})}
                    >
                      {[1,2,3,4,5,6,7,8,9].map(n => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1 col-span-2">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Discharge Destination</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.discharge_disposition}
                      onChange={(e) => setPatient({...patient, discharge_disposition: e.target.value})}
                    >
                      {DISCHARGE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1 col-span-2">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Primary Diagnosis Group</label>
                    <select
                      className={`${fieldClass} cursor-pointer`}
                      value={patient.diag_1_group}
                      onChange={(e) => setPatient({...patient, diag_1_group: e.target.value})}
                    >
                      {DIAG_GROUPS.map((group) => (
                        <option key={group} value={group}>{group}</option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Second diagnosis group</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.diag_2_group} onChange={(e) => setPatient({ ...patient, diag_2_group: e.target.value })}>
                      {DIAG_GROUPS.map((group) => <option key={group} value={group}>{group}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Third diagnosis group</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.diag_3_group} onChange={(e) => setPatient({ ...patient, diag_3_group: e.target.value })}>
                      {DIAG_GROUPS.map((group) => <option key={group} value={group}>{group}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Outpatient visits</label>
                    <input type="number" min={0} max={40} className={fieldClass} value={patient.number_outpatient} onChange={(e) => { const number_outpatient = clampInt(e.target.value, 0, 40); if (number_outpatient !== null) setPatient({ ...patient, number_outpatient }); }} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Lab procedures</label>
                    <input type="number" min={0} max={140} className={fieldClass} value={patient.num_lab_procedures} onChange={(e) => { const num_lab_procedures = clampInt(e.target.value, 0, 140); if (num_lab_procedures !== null) setPatient({ ...patient, num_lab_procedures }); }} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Procedures</label>
                    <input type="number" min={0} max={10} className={fieldClass} value={patient.num_procedures} onChange={(e) => { const num_procedures = clampInt(e.target.value, 0, 10); if (num_procedures !== null) setPatient({ ...patient, num_procedures }); }} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Glucose serum</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.max_glu_serum} onChange={(e) => setPatient({ ...patient, max_glu_serum: e.target.value as PatientEncounter["max_glu_serum"] })}>
                      <option value="Not_Tested">Not tested</option>
                      <option value="Norm">Normal</option>
                      <option value=">200">&gt;200</option>
                      <option value=">300">&gt;300</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Admission type</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.admission_type_id} onChange={(e) => setPatient({ ...patient, admission_type_id: e.target.value })}>
                      {ADMISSION_TYPES.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Admission source</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.admission_source_id} onChange={(e) => setPatient({ ...patient, admission_source_id: e.target.value })}>
                      {ADMISSION_SOURCES.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Regimen changed</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.change} onChange={(e) => setPatient({ ...patient, change: e.target.value as PatientEncounter["change"] })}>
                      <option value="No">No</option>
                      <option value="Ch">Yes</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Specialty</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.medical_specialty} onChange={(e) => setPatient({ ...patient, medical_specialty: e.target.value })}>
                      {SPECIALTIES.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Payer code</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.payer_code} onChange={(e) => setPatient({ ...patient, payer_code: e.target.value })}>
                      {PAYERS.map((code) => <option key={code} value={code}>{code}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-[hsl(var(--muted-foreground))]">Race</label>
                    <select className={`${fieldClass} cursor-pointer`} value={patient.race} onChange={(e) => setPatient({ ...patient, race: e.target.value })}>
                      {RACES.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </div>
                  {DRUG_FIELDS.map((drug) => (
                    <div key={drug.key} className="space-y-1">
                      <label className="text-xs text-[hsl(var(--muted-foreground))]">{drug.label}</label>
                      <select
                        className={`${fieldClass} cursor-pointer`}
                        value={patient[drug.key] as MedStatus}
                        onChange={(e) => setPatient({ ...patient, [drug.key]: e.target.value as MedStatus })}
                      >
                        {MED_STATUSES.map((status) => <option key={status} value={status}>{status}</option>)}
                      </select>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2 pt-2">
                <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider flex items-center gap-2">
                  <FileText className="w-4 h-4" /> Unstructured Clinical Note
                </label>
                <textarea
                  className={`${fieldClass} h-32`}
                  value={patient.clinical_note}
                  onChange={(e) => setPatient({...patient, clinical_note: e.target.value})}
                />
                {noteTooShort && (
                  <p className="text-sm text-[hsl(var(--muted-foreground))]">{NOTE_HINT}</p>
                )}
              </div>

              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="w-full mt-4 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:opacity-90 font-medium py-3 rounded-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                {isAnalyzing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Activity className="w-5 h-5" />}
                {isAnalyzing ? (reviewStep ?? "Reviewing chart…") : "Analyze patient"}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: AI Insights */}
        <div className="lg:col-span-8 space-y-6">
          {!hasOutcome ? (
            <div className="h-full min-h-[400px] border border-dashed border-[hsl(var(--border))] rounded-xl flex flex-col items-center justify-center text-[hsl(var(--muted-foreground))] space-y-4">
              <ShieldAlert className="w-12 h-12 opacity-20" />
              <p>Run analysis to generate prescriptive insights.</p>
            </div>
          ) : (
            <div className="space-y-6">
              
              {predictError && !predictData && (
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <p className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-2">Readmission Risk</p>
                  <p className="text-sm text-rose-400">{predictError}</p>
                </div>
              )}

              {predictData && tone && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Risk Score */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6 relative overflow-hidden">
                  <div className={`absolute top-0 left-0 w-1 h-full ${tone.bar}`} />
                  <p className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-2">Readmission Risk</p>
                  <div className="flex items-end gap-3">
                    <span className="text-4xl font-bold">{(predictData.risk_score * 100).toFixed(1)}%</span>
                    <span className={`text-sm font-medium mb-1 ${tone.text}`}>
                      {predictData.risk_category}
                    </span>
                  </div>
                  <p className="text-xs text-[hsl(var(--muted-foreground))] mt-3">
                    Estimate for a diabetic discharge, based on historical hospital stays. Not an order.
                  </p>
                </div>

                {/* SDoH Extraction */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <p className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-3">From the discharge note</p>
                  {noSocialSignals ? (
                    <p className="text-sm text-[hsl(var(--muted-foreground))]">No SDoH signals detected</p>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {predictData.sdoh_flags.map((flag, idx) => (
                        <span key={idx} className="bg-rose-500/10 text-rose-400 border border-rose-500/20 text-xs px-2.5 py-1 rounded-full flex items-center gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          {flag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Financial ROI */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <p className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-2">CMS Penalty Avoidance</p>
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg">
                      <DollarSign className="w-6 h-6" />
                    </div>
                    <div>
                      {typeof predictData.cms_penalty_saved_usd === "number" ? (
                        <>
                          <div className="text-2xl font-bold text-emerald-400">${predictData.cms_penalty_saved_usd.toLocaleString()}</div>
                          <div className="text-xs text-[hsl(var(--muted-foreground))]">Est. HRRP savings</div>
                        </>
                      ) : (
                        <div className="text-sm text-[hsl(var(--muted-foreground))]">Estimate appears when risk is high</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              )}

              {/* Middle Row: similar stays and checklist */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* Prescriptive Care Plan */}
                {predictData && (
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    Discharge checklist
                  </h3>
                  {predictData.plan_source === "gemini" && (
                    <p className="text-xs text-[hsl(var(--muted-foreground))] mb-4">Written by the assistant.</p>
                  )}
                  <div className="space-y-3">
                    {predictData.care_plan.map((step, idx) => (
                      <div key={idx} className="flex gap-3 bg-[hsl(var(--muted))] p-3 rounded-lg border border-[hsl(var(--border))]">
                        <div className="w-6 h-6 rounded-full bg-[hsl(var(--background))] border border-[hsl(var(--border))] flex items-center justify-center text-xs font-bold shrink-0">
                          {idx + 1}
                        </div>
                        <p className="text-sm leading-relaxed">{step}</p>
                      </div>
                    ))}
                  </div>
                  {predictData.clinical_rationale && (
                    <p className="text-sm text-[hsl(var(--muted-foreground))] mt-4">{predictData.clinical_rationale}</p>
                  )}
                </div>
                )}

                {/* Similar past stays */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6 flex flex-col">
                  {ragData ? (
                    <>
                      <div className="flex justify-between items-start mb-4">
                        <h3 className="text-lg font-medium flex items-center gap-2">
                          <Users className="w-5 h-5 text-blue-400" />
                          Similar past stays
                        </h3>
                        <div className="text-right">
                          <div className="text-xs text-[hsl(var(--muted-foreground))]">How often similar patients returned</div>
                          <div className="text-lg font-bold text-amber-400">{(ragData.rag_readmission_rate * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                      
                      <div className="flex-1 overflow-y-auto space-y-3 pr-2">
                        {ragData.twins.map(twin => (
                          <div key={twin.twin_id} className="p-3 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-sm font-medium">{twin.twin_id}</span>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                                {(twin.similarity_score * 100).toFixed(0)}% similar
                              </span>
                            </div>
                            <div className="flex items-center gap-2 text-xs text-[hsl(var(--muted-foreground))] mb-2 flex-wrap">
                              <span className={`px-1.5 py-0.5 rounded ${twin.was_readmitted ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
                                {twin.was_readmitted ? "Came back within 30 days" : "Stayed home"}
                              </span>
                              <span>•</span>
                              <span>Age {twin.age}</span>
                              {twin.diagnosis_group && (
                                <>
                                  <span>•</span>
                                  <span>{twin.diagnosis_group} group</span>
                                </>
                              )}
                            </div>
                            {!twin.was_readmitted && twin.successful_interventions.length > 0 && (
                              <div className="mt-2 text-xs border-t border-[hsl(var(--border))] pt-2">
                                <span className="text-[hsl(var(--muted-foreground))] mb-1 block">What was done for patients who stayed home</span>
                                <ul className="list-disc pl-4 space-y-0.5 text-[hsl(var(--foreground))]">
                                  {twin.successful_interventions.map((inv, i) => (
                                    <li key={i}>{inv}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <>
                      <h3 className="text-lg font-medium flex items-center gap-2 mb-4">
                        <Users className="w-5 h-5 text-blue-400" />
                        Similar past stays
                      </h3>
                      <p className="text-sm text-rose-400">{ragError}</p>
                    </>
                  )}
                </div>
              </div>

              {/* Bottom Row: why this score */}
              {predictData && (
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-purple-400" />
                    Why this score
                  </h3>
                  <div className="space-y-4">
                    {predictData.shap_features.map((feat) => {
                      const width = maxAbsImpact === 0 ? 0 : (Math.abs(feat.impact) / maxAbsImpact) * 100;
                      return (
                        <div key={feat.feature} className="flex items-center gap-4">
                          <div className="w-1/4 text-sm text-[hsl(var(--muted-foreground))] text-right truncate" title={feat.feature}>
                            {shapLabel(feat.feature)}
                          </div>
                          <div className="flex-1 h-3 bg-[hsl(var(--muted))] rounded-full overflow-hidden flex">
                            <div className="w-1/2 flex justify-end">
                              {feat.impact < 0 && width > 0 && (
                                <div className="h-full bg-emerald-400" style={{ width: `${width}%` }} />
                              )}
                            </div>
                            <div className="w-1/2 flex justify-start">
                              {feat.impact > 0 && width > 0 && (
                                <div className="h-full bg-rose-400" style={{ width: `${width}%` }} />
                              )}
                            </div>
                          </div>
                          <div className={`w-16 text-right text-xs font-medium ${feat.impact > 0 ? "text-rose-400" : "text-emerald-400"}`}>
                            {feat.impact > 0 ? "+" : ""}{feat.impact.toFixed(3)}
                          </div>
                        </div>
                      );
                    })}
                    <div className="flex justify-between text-xs text-[hsl(var(--muted-foreground))] mt-2 px-[25%] border-t border-[hsl(var(--border))] pt-2">
                      <span>Lowers Risk</span>
                      <span>Increases Risk</span>
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}
        </div>
      </main>

      <footer className="max-w-[1600px] mx-auto mt-10 pt-6 border-t border-[hsl(var(--border))]">
        <a
          href="http://localhost:3001"
          target="_blank"
          rel="noreferrer"
          className="text-sm text-[hsl(var(--muted-foreground))] underline"
        >
          Operations board
        </a>
      </footer>
    </div>
  );
}
