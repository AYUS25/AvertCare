"use client";

import { useState } from "react";
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  FileText, 
  Pill, 
  Stethoscope, 
  Users,
  ShieldAlert,
  Loader2,
  DollarSign
} from "lucide-react";

// Types matching our Backend Pydantic Schemas
type PatientEncounter = {
  patient_id: string;
  age: number;
  primary_diagnosis: string;
  time_in_hospital: number;
  num_prior_admissions: number;
  num_medications: number;
  clinical_note: string;
};

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
  cms_penalty_saved_usd?: number;
};

type TwinPatient = {
  twin_id: string;
  age: number;
  primary_diagnosis: string;
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

// Mock Data Contract
const MOCK_PATIENT: PatientEncounter = {
  patient_id: "MRN-782910",
  age: 67,
  primary_diagnosis: "Type 2 Diabetes with complications",
  time_in_hospital: 7,
  num_prior_admissions: 2,
  num_medications: 8,
  clinical_note: "Patient lives alone and has expressed concerns about inability to afford insulin. Polypharmacy noted. Blood glucose stabilizing. Discharge planned for tomorrow. Transport home is currently unarranged.",
};

export default function Dashboard() {
  const [patient, setPatient] = useState<PatientEncounter>(MOCK_PATIENT);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictData, setPredictData] = useState<PredictResponse | null>(null);
  const [ragData, setRagData] = useState<TwinPatientResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const [predictRes, ragRes] = await Promise.all([
        fetch(`${API_URL}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patient),
        }),
        fetch(`${API_URL}/api/twin-patients`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patient),
        }),
      ]);

      if (!predictRes.ok || !ragRes.ok) throw new Error("Failed to fetch from AvertCare API");

      setPredictData(await predictRes.json());
      setRagData(await ragRes.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

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
        <div className="px-4 py-2 text-sm font-medium border rounded-full border-[hsl(var(--border))] text-[hsl(var(--muted-foreground))]">
          FHIR Sync: <span className="text-emerald-400">Connected</span>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-8 max-w-[1600px] mx-auto">
        
        {/* Left Column: Patient Profile */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium flex items-center gap-2">
                <Users className="w-5 h-5 text-[hsl(var(--muted-foreground))]" />
                Patient Encounter
              </h2>
              <span className="text-xs bg-[hsl(var(--accent))] text-[hsl(var(--accent-foreground))] px-2 py-1 rounded-md font-mono">
                {patient.patient_id}
              </span>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Age</label>
                  <div className="font-medium text-lg">{patient.age} yrs</div>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Length of Stay</label>
                  <div className="font-medium text-lg flex items-center gap-2">
                    <Clock className="w-4 h-4 text-amber-500" />
                    {patient.time_in_hospital} days
                  </div>
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Primary Diagnosis</label>
                <div className="font-medium flex items-start gap-2">
                  <Stethoscope className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                  {patient.primary_diagnosis}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="space-y-1">
                  <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Prior Admits (12m)</label>
                  <div className="font-medium text-lg flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-rose-400" />
                    {patient.num_prior_admissions}
                  </div>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Discharge Meds</label>
                  <div className="font-medium text-lg flex items-center gap-2">
                    <Pill className="w-4 h-4 text-purple-400" />
                    {patient.num_medications}
                  </div>
                </div>
              </div>

              <div className="space-y-2 pt-4">
                <label className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider flex items-center gap-2">
                  <FileText className="w-4 h-4" /> Unstructured Clinical Note
                </label>
                <textarea
                  className="w-full h-32 bg-[hsl(var(--muted))] border border-[hsl(var(--border))] rounded-lg p-3 text-sm focus:outline-none focus:ring-1 focus:ring-[hsl(var(--primary))]"
                  value={patient.clinical_note}
                  onChange={(e) => setPatient({...patient, clinical_note: e.target.value})}
                />
              </div>

              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="w-full mt-4 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:opacity-90 font-medium py-3 rounded-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                {isAnalyzing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Activity className="w-5 h-5" />}
                {isAnalyzing ? "Processing RAG & Models..." : "Analyze Patient Record"}
              </button>
              
              {error && <div className="text-rose-500 text-sm mt-2 text-center">{error}</div>}
            </div>
          </div>
        </div>

        {/* Right Column: AI Insights */}
        <div className="lg:col-span-8 space-y-6">
          {!predictData ? (
            <div className="h-full min-h-[400px] border border-dashed border-[hsl(var(--border))] rounded-xl flex flex-col items-center justify-center text-[hsl(var(--muted-foreground))] space-y-4">
              <ShieldAlert className="w-12 h-12 opacity-20" />
              <p>Run analysis to generate prescriptive insights.</p>
            </div>
          ) : (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              
              {/* Top Metrics Row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Risk Score */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6 relative overflow-hidden">
                  <div className={`absolute top-0 left-0 w-1 h-full ${predictData.risk_category === 'High Risk' ? 'bg-rose-500' : 'bg-amber-500'}`} />
                  <p className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-2">Readmission Risk</p>
                  <div className="flex items-end gap-3">
                    <span className="text-4xl font-bold">{(predictData.risk_score * 100).toFixed(1)}%</span>
                    <span className={`text-sm font-medium mb-1 ${predictData.risk_category === 'High Risk' ? 'text-rose-400' : 'text-amber-400'}`}>
                      {predictData.risk_category}
                    </span>
                  </div>
                </div>

                {/* SDoH Extraction */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <p className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-3">SDoH Extracted (ClinicalBERT)</p>
                  <div className="flex flex-wrap gap-2">
                    {predictData.sdoh_flags.map((flag, idx) => (
                      <span key={idx} className="bg-rose-500/10 text-rose-400 border border-rose-500/20 text-xs px-2.5 py-1 rounded-full flex items-center gap-1.5">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        {flag}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Financial ROI */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <p className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-2">CMS Penalty Avoidance</p>
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg">
                      <DollarSign className="w-6 h-6" />
                    </div>
                    <div>
                      {predictData.cms_penalty_saved_usd ? (
                        <>
                          <div className="text-2xl font-bold text-emerald-400">${predictData.cms_penalty_saved_usd.toLocaleString()}</div>
                          <div className="text-xs text-[hsl(var(--muted-foreground))]">Est. HRRP savings</div>
                        </>
                      ) : (
                        <div className="text-sm text-[hsl(var(--muted-foreground))]">N/A (Low Risk)</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Middle Row: RAG & GenAI Plan */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* Prescriptive Care Plan */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                  <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    GenAI Copilot Care Plan
                  </h3>
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
                </div>

                {/* Twin-Patient RAG Data */}
                {ragData && (
                  <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6 flex flex-col">
                    <div className="flex justify-between items-start mb-4">
                      <h3 className="text-lg font-medium flex items-center gap-2">
                        <Users className="w-5 h-5 text-blue-400" />
                        Twin-Patient RAG Retrieval
                      </h3>
                      <div className="text-right">
                        <div className="text-xs text-[hsl(var(--muted-foreground))]">Cohort Readmission Rate</div>
                        <div className="text-lg font-bold text-amber-400">{(ragData.rag_readmission_rate * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                    
                    <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
                      {ragData.twins.map(twin => (
                        <div key={twin.twin_id} className="p-3 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium">{twin.twin_id}</span>
                            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                              {(twin.similarity_score * 100).toFixed(0)}% Match
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-[hsl(var(--muted-foreground))] mb-2">
                            <span className={`px-1.5 py-0.5 rounded ${twin.was_readmitted ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
                              {twin.was_readmitted ? 'Readmitted' : 'Success'}
                            </span>
                            <span>•</span>
                            <span>Age {twin.age}</span>
                          </div>
                          {!twin.was_readmitted && twin.successful_interventions.length > 0 && (
                            <div className="mt-2 text-xs border-t border-[hsl(var(--border))] pt-2">
                              <span className="text-[hsl(var(--muted-foreground))] mb-1 block">Successful Interventions:</span>
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
                  </div>
                )}
              </div>

              {/* Bottom Row: SHAP Explainability */}
              <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl p-6">
                <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-purple-400" />
                  XAI Model Explainability (SHAP)
                </h3>
                <div className="space-y-4">
                  {predictData.shap_features.map((feat) => (
                    <div key={feat.feature} className="flex items-center gap-4">
                      <div className="w-1/4 text-sm text-[hsl(var(--muted-foreground))] text-right truncate">
                        {feat.feature}
                      </div>
                      <div className="flex-1 h-3 bg-[hsl(var(--muted))] rounded-full overflow-hidden flex">
                        {/* Visualization trick: positive impacts push right, negative push left. Simplified for UI. */}
                        <div className="w-1/2 flex justify-end">
                          {feat.impact < 0 && <div className="h-full bg-emerald-400" style={{ width: `${Math.abs(feat.impact) * 200}%` }} />}
                        </div>
                        <div className="w-1/2 flex justify-start">
                          {feat.impact > 0 && <div className="h-full bg-rose-400" style={{ width: `${feat.impact * 200}%` }} />}
                        </div>
                      </div>
                      <div className={`w-16 text-right text-xs font-medium ${feat.impact > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        {feat.impact > 0 ? '+' : ''}{feat.impact.toFixed(3)}
                      </div>
                    </div>
                  ))}
                  <div className="flex justify-between text-xs text-[hsl(var(--muted-foreground))] mt-2 px-[25%] border-t border-[hsl(var(--border))] pt-2">
                    <span>Lowers Risk</span>
                    <span>Increases Risk</span>
                  </div>
                </div>
              </div>

            </div>
          )}
        </div>
      </main>
    </div>
  );
}
