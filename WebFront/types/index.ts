export type PatientDoctor = { id: number; full_name: string };

export type Patient = {
  id: number;
  name: string;
  birthDate: string;
  gender: string;
  lastVisit: string;
  patient_code: string | null;
  /** Attending doctor; admins see other doctors' patients with `scope=unit`. */
  doctor: PatientDoctor | null;
};

// GET /patients query: free-text search (name / code), gender filter, and the admin-only unit scope.
export type PatientListParams = {
  q?: string;
  gender?: "M" | "F";
  scope?: "mine" | "unit";
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
  /** Patient-friendly explanation generated alongside the clinician summary (`Message.content`). */
  patient_summary?: string | null;
};

// How the pipeline classified the clinician's question.
export type QuestionCategory = "analysis_request" | "follow_up" | "general_medical" | "chit_chat" | "out_of_scope";

// Pipeline metadata on assistant messages. Absent on messages persisted before the LangGraph backend.
export type MessageMeta = {
  question_category?: QuestionCategory;
  /** Suggested next questions; the UI offers up to three as chips. */
  follow_ups?: string[];
  /** The question was refused by the scope / moderation gate; `content` holds the refusal text. */
  blocked?: boolean;
  block_reason?: string;
  /** Auto-generated session title, sent with the reply that produced it. */
  title?: string | null;
};

export type MessageRole = "user" | "assistant" | "patient";

export type Message = {
  id: number;
  session_id?: number;
  /** "patient": a question asked through a share link; the assistant reply follows it. */
  role: MessageRole;
  content: string;
  image_path?: string | null;
  created_at?: string | null;
  analysis?: Analysis;
  meta?: MessageMeta;
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
  /** Visit the session was attached to at creation (optional). */
  visit_id: number | null;
};

export type FeedbackResult = {
  diagnosis_id: number;
  doctor_feedback: boolean;
  doctor_note: string | null;
};

// Known pipeline stages. The server may add stages at any time, so any other string is accepted too:
// render `StreamStatus.message` instead of switching on the stage.
export type StreamStage =
  | "validating"
  | "moderating"
  | "classifying"
  | "retrieving"
  | "segmenting"
  | "radiomics"
  | "reporting"
  | "summarizing"
  | "explaining"
  | "checking"
  | "answering"
  | "finalizing"
  | (string & {});

export type StreamStatus = { stage: StreamStage; message: string };

// Callbacks for POST /analyze/stream (server-sent events).
export type StreamHandlers = {
  onStatus: (status: StreamStatus) => void;
  onAnalysis: (analysis: Analysis) => void;
  onDelta: (text: string) => void;
  onDone: (message: Message) => void;
  onError: (detail: string) => void;
};

export type UserRole = "doctor" | "admin";

export type HospitalUnit = { id: number; code: string; name: string };

export type AuthUser = {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  hospital_unit: HospitalUnit | null;
};

export type LoginResponse = {
  access_token: string;
  token_type: string;
  user: AuthUser;
};

// ---- Visits ----

// One analysis performed during a visit (compact form for the visit list / badges).
export type VisitAnalysis = {
  diagnosis_id: number;
  session_id: number;
  created_at: string;
  risk_level: RiskLevel;
  tb_probability: number | null;
  primary_label: string;
  overlay_url: string | null;
};

export type Visit = {
  id: number;
  visited_at: string;
  symptoms: string | null;
  treatment: string | null;
  notes: string | null;
  session_count: number;
  analyses: VisitAnalysis[];
};

// POST /patients/{id}/visits and PATCH /visits/{id} body (every field optional).
export type VisitInput = {
  /** ISO datetime or YYYY-MM-DD */
  visited_at?: string;
  symptoms?: string;
  treatment?: string;
  notes?: string;
};

// ---- Timeline (per-patient trend) ----

export type TimelinePoint = {
  date: string;
  diagnosis_id: number;
  session_id: number;
  visit_id: number | null;
  tb_probability: number | null;
  risk_level: RiskLevel;
  primary_label: string;
  mask_area_fraction: number | null;
};

export type TimelineVisit = { id: number; visited_at: string; symptoms: string | null };

export type Timeline = {
  /** Oldest first. */
  points: TimelinePoint[];
  visits: TimelineVisit[];
};

// ---- Patient share links (doctor side) ----

export type LinkExpirePolicy = "time" | "manual" | "both";

export type LinkStatus = "active" | "expired" | "revoked";

export type PatientLink = {
  id: number;
  token: string;
  url: string;
  session_id: number;
  expire_policy: LinkExpirePolicy;
  expires_at: string | null;
  revoked_at: string | null;
  max_questions: number | null;
  questions_used: number;
  created_at: string;
  last_accessed_at: string | null;
  status: LinkStatus;
};

export type CreateLinkInput = {
  session_id: number;
  expire_policy: LinkExpirePolicy;
  hours?: number;
  max_questions?: number;
};

export type LinkAccessKind = "view" | "question" | "denied";

export type LinkAccess = {
  accessed_at: string;
  kind: LinkAccessKind;
  question: string | null;
  client_ip: string | null;
};

// ---- Public share view (no auth) ----

export type ShareMessage = {
  role: "patient" | "assistant";
  content: string;
  created_at: string | null;
};

export type ShareView = {
  status: "active";
  patient: { name: string };
  doctor: { full_name: string };
  session: { title: string; created_at: string };
  analysis: { patient_summary: string | null; overlay_url: string | null; created_at: string } | null;
  follow_ups: string[];
  questions_remaining: number;
  expires_at: string | null;
  messages: ShareMessage[];
};

export type ShareAskResult = {
  message: { role: "assistant"; content: string; meta: { follow_ups: string[] } };
  questions_remaining: number;
};

// ---- Dashboard ----

export type DashboardRecent = {
  diagnosis_id: number;
  patient_id: number;
  patient_name: string;
  created_at: string;
  risk_level: RiskLevel;
  tb_probability: number | null;
  primary_label: string;
};

export type DashboardDoctorUsage = {
  doctor_id: number;
  full_name: string;
  patients: number;
  analyses: number;
  llm_cost_usd: number;
  avg_latency_ms: number | null;
  last_active: string | null;
};

export type DashboardOverview = {
  period_days: number;
  patients: number;
  analyses: number;
  tb_suspected: number;
  low_likelihood: number;
  no_lesion: number;
  avg_latency_ms: number | null;
  llm_cost_usd: number;
  feedback: { agree: number; disagree: number };
  recent: DashboardRecent[];
  /** Admins only. */
  by_doctor?: DashboardDoctorUsage[];
};

// ---- Admin (user management) ----

export type AdminUser = {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  patients: number;
};

export type CreateAdminUserInput = {
  email: string;
  full_name: string;
  password: string;
  role: UserRole;
};

export type PatchAdminUserInput = {
  is_active?: boolean;
  role?: UserRole;
  full_name?: string;
};
