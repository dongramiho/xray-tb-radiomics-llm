export type Patient = {
  id: number;
  name: string;
  birthDate: string;
  gender: string;
  lastVisit: string;
};

export type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  image_path?: string;
  created_at?: string;
};