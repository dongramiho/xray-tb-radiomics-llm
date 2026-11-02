export type Patient = {
  id: number;
  name: string;
  birthDate: string;
  gender: string;
  lastVisit: string;
};

export type MlResults = {
  primary_label: string;
  tb_probability: number | null;
  risk_level: "low" | "moderate" | "high";
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

export type Analysis = {
  diagnosis_id: number;
  overlay_url: string | null;
  mask_url: string | null;
  ml_results: MlResults | null;
  segmentation: SegmentationSummary | null;
  report: Record<string, unknown> | null;
  doctor_feedback: boolean | null;
};

export type Message = {
  id: number;
  session_id?: number;
  role: "user" | "assistant";
  content: string;
  image_path?: string | null;
  created_at?: string | null;
  analysis?: Analysis;
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
