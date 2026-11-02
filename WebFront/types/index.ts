export type Patient = {
  id: number;
  name: string;
  birthDate: string;
  gender: string;
  lastVisit: string;
};

export type RiskLevel = "low" | "moderate" | "high";

export type MlResults = {
  primary_label: string;
  tb_probability: number | null;
  risk_level: RiskLevel;
  classifier?: string | null;
  threshold?: number;
  top_features?: { feature: string; importance: number; value: number | null }[];
  decision_basis?: string;
};

export type SegmentationSummary = {
  detected: boolean;
  mask_area_fraction: number;
  num_components: number;
  zones: string[];
  max_probability: number;
};

// Stage-1 structured report produced by the report LLM.
export type Report = {
  clinical_overview: string;
  tb_risk_summary: { level: RiskLevel; rationale: string };
  lesion_summary: {
    num_regions: number;
    zones: string[];
    area_fraction: number | null;
    morphology: string;
    texture: string;
  };
  ml_summary: {
    primary_label: string;
    tb_probability: number | null;
    model_confidence: RiskLevel;
    reliability_comment: string;
  };
  recommendations_for_clinician: string[];
  supporting_evidence: { json_key: string; category: string; role: string; explanation: string }[];
  limitations: {
    data_limitations: string[];
    model_limitations: string[];
    usage_restrictions: string;
  };
};

// LLM usage / provenance for one analysis (report stage -> summary stage).
export type LlmMeta = {
  report_model: string;
  summary_model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  prompt_version: string;
};

export type Analysis = {
  diagnosis_id: number;
  overlay_url: string | null;
  mask_url: string | null;
  ml_results: MlResults | null;
  segmentation: SegmentationSummary | null;
  report: Report | null;
  doctor_feedback: boolean | null;
  doctor_note: string | null;
  llm?: LlmMeta;
};

export type Message = {
  id: number;
  session_id?: number;
  role: "user" | "assistant";
  content: string;
  image_path?: string | null;
  created_at?: string | null;
  analysis?: Analysis;
  /** client-only: assistant reply is still being streamed */
  streaming?: boolean;
  /** client-only: current pipeline stage text while streaming */
  status?: string;
  /** client-only: failure detail shown inside the bubble */
  error?: string;
};

export type Session = {
  id: number;
  title: string;
  created_at: string;
  message_count: number;
  last_message_at: string | null;
};

export type FeedbackResult = {
  diagnosis_id: number;
  doctor_feedback: boolean;
  doctor_note: string | null;
};

export type StreamStage = "segmenting" | "radiomics" | "classifying" | "reporting" | "summarizing" | "answering";

export type StreamStatus = { stage: StreamStage; message: string };

// Callbacks for POST /analyze/stream (server-sent events).
export type StreamHandlers = {
  onStatus: (status: StreamStatus) => void;
  onAnalysis: (analysis: Analysis) => void;
  onDelta: (text: string) => void;
  onDone: (message: Message) => void;
  onError: (detail: string) => void;
};

export type AuthUser = {
  id: number;
  email: string;
  full_name: string;
};

export type LoginResponse = {
  access_token: string;
  token_type: string;
  user: AuthUser;
};
