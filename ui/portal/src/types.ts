export type UserRole = "admin" | "user" | "readonly";

export interface User {
  sub: string;
  role: UserRole;
  isGuest?: boolean;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  attachments?: Array<{ name: string; content: string }>;
}

export interface ChatThread {
  id: string;
  title: string;
  projectId: string | null;
  updatedAt: number;
  messages: ChatMessage[];
}

export interface Project {
  id: string;
  name: string;
  createdAt: number;
}

export interface TrainingStatus {
  is_running: boolean;
  elapsed_sec: number | null;
  current_loss: number | null;
  val_loss: number | null;
  epoch: number | null;
  total_epochs: number;
  perplexity: number | null;
  epoch_history: Array<Record<string, unknown>>;
  recent_steps: Array<Record<string, unknown>>;
}

export interface ModelRecord {
  name: string;
  size_mb: number;
  epoch: number | null;
  loss: number | null;
  is_best: boolean;
  ver: string;
}

export interface SystemInfo {
  os: string;
  cpu: string;
  cores: number;
  ram_total: number;
  ram_avail: number;
  gpu: string | null;
  gpu_vram: number | null;
  device: string;
}

export interface StreamEvent {
  type: string;
  payload: Record<string, unknown>;
  timestamp: number;
}
