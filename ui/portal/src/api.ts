import type {
  ModelRecord,
  Project,
  StreamEvent,
  SystemInfo,
  TokenPair,
  TrainingStatus,
  User
} from "./types";
import { fixtureAdminRequest, fixtureAssistantAnswer, fixtureMetricEvent } from "./devFixtures";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const ACCESS_KEY = "dgb.access";
const REFRESH_KEY = "dgb.refresh";
const USER_KEY = "dgb.user";
const PREVIEW_KEY = "dgb.uiPreview";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function stored(key: string): string | null {
  return sessionStorage.getItem(key);
}

export function currentUser(): User | null {
  const raw = stored(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as User;
  } catch {
    clearSession();
    return null;
  }
}

export function isPreviewSession(): boolean {
  return import.meta.env.DEV && sessionStorage.getItem(PREVIEW_KEY) === "true";
}

export function enterUiPreview(role: "admin" | "user"): User {
  if (!import.meta.env.DEV) throw new Error("UI preview fixtures are only available in development.");
  const user: User = { sub: role === "admin" ? "preview-admin" : "preview-user", role };
  clearSession();
  sessionStorage.setItem(PREVIEW_KEY, "true");
  sessionStorage.setItem(USER_KEY, JSON.stringify(user));
  return user;
}

export function saveSession(tokens: TokenPair, user: User): void {
  sessionStorage.removeItem(PREVIEW_KEY);
  sessionStorage.setItem(ACCESS_KEY, tokens.access_token);
  sessionStorage.setItem(REFRESH_KEY, tokens.refresh_token);
  sessionStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function updateUser(user: User): void {
  sessionStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearSession(): void {
  sessionStorage.removeItem(ACCESS_KEY);
  sessionStorage.removeItem(REFRESH_KEY);
  sessionStorage.removeItem(USER_KEY);
  sessionStorage.removeItem(PREVIEW_KEY);
}

async function responseError(response: Response): Promise<ApiError> {
  let message = `Request failed (${response.status})`;
  try {
    const body = await response.json() as { detail?: string; message?: string };
    message = body.detail ?? body.message ?? message;
  } catch {
    // Keep the status-based message for non-JSON API responses.
  }
  return new ApiError(message, response.status);
}

async function refreshAccessToken(): Promise<boolean> {
  const refreshToken = stored(REFRESH_KEY);
  if (!refreshToken) return false;
  const response = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken })
  });
  if (!response.ok) {
    clearSession();
    return false;
  }
  const data = await response.json() as { access_token: string };
  sessionStorage.setItem(ACCESS_KEY, data.access_token);
  return true;
}

async function request<T>(path: string, init: RequestInit = {}, retried = false): Promise<T> {
  if (isPreviewSession() && path.startsWith("/v1/admin/")) {
    return await fixtureAdminRequest(path, init) as T;
  }
  const headers = new Headers(init.headers);
  const token = stored(ACCESS_KEY);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (response.status === 401 && !retried && await refreshAccessToken()) {
    return request<T>(path, init, true);
  }
  if (!response.ok) throw await responseError(response);
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function login(username: string, password: string): Promise<User> {
  const body = new URLSearchParams({ username, password });
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body
  });
  if (!response.ok) throw await responseError(response);
  const tokens = await response.json() as TokenPair;
  const meResponse = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${tokens.access_token}` }
  });
  if (!meResponse.ok) throw await responseError(meResponse);
  const user = await meResponse.json() as User;
  saveSession(tokens, user);
  return user;
}

export async function getMe(): Promise<User> {
  return request<User>("/auth/me");
}

export async function getHealth(): Promise<{ status: string; version: string; uptime: number }> {
  return request("/health");
}

export async function completeChat(
  model: string,
  messages: Array<{ role: string; content: string }>,
  signal: AbortSignal,
  onToken: (token: string) => void
): Promise<void> {
  if (isPreviewSession()) {
    const answer = fixtureAssistantAnswer(messages[messages.length - 1]?.content ?? "");
    const words = answer.match(/\S+\s*/g) ?? [];
    for (const word of words) {
      if (signal.aborted) throw new DOMException("Response stopped.", "AbortError");
      onToken(word);
      await new Promise((resolve) => window.setTimeout(resolve, 24));
    }
    return;
  }
  const headers = new Headers({ "Content-Type": "application/json" });
  const token = stored(ACCESS_KEY);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let response = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model, messages, max_tokens: 512, temperature: 0.8, stream: true }),
    signal
  });
  if (response.status === 401 && await refreshAccessToken()) {
    headers.set("Authorization", `Bearer ${stored(ACCESS_KEY)}`);
    response = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model, messages, max_tokens: 512, temperature: 0.8, stream: true }),
      signal
    });
  }
  if (!response.ok) throw await responseError(response);
  if (!response.body) throw new ApiError("Streaming is not supported by this response.", response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      for (const line of block.split(/\r?\n/)) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        try {
          const parsed = JSON.parse(data) as {
            choices?: Array<{ delta?: { content?: string } }>;
          };
          const fragment = parsed.choices?.[0]?.delta?.content;
          if (fragment) onToken(fragment);
        } catch {
          throw new ApiError("The inference stream returned malformed event data.", response.status);
        }
      }
    }
    if (done) break;
  }
}

export const adminApi = {
  health: () => request<{ status: string; version: string; model: string; time: number }>("/v1/admin/health"),
  system: () => request<SystemInfo>("/v1/admin/system"),
  training: () => request<TrainingStatus>("/v1/admin/training/status"),
  models: () => request<{ models: ModelRecord[] }>("/v1/admin/models"),
  pipeline: () => request<Record<string, unknown>>("/v1/admin/pipeline/status"),
  metrics: (n = 100) => request<{ epoch_history: Array<Record<string, unknown>>; step_history: Array<Record<string, unknown>> }>(`/v1/admin/metrics/history?n=${n}`),
  launchTraining: (values: { epochs: number; batch_size: number; lr: number; warmup: number }) =>
    request<{ status: string; pid?: number }>("/v1/admin/training/launch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values)
    }),
  stopTraining: () => request<{ status: string }>("/v1/admin/training/stop", { method: "POST" }),
  runPipeline: () => request<{ status: string; pid?: number }>("/v1/admin/pipeline/run", { method: "POST" }),
  deployModel: (model: string) => request<{ status: string; model: string }>("/v1/admin/models/deploy", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model })
  }),
  rollbackModel: (model: string) => request<{ status: string; model: string }>("/v1/admin/models/rollback", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model })
  })
};

export async function consumeAdminEvents(
  type: "metrics" | "logs",
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  if (isPreviewSession()) {
    onEvent(fixtureMetricEvent());
    await new Promise<void>((resolve) => {
      const interval = window.setInterval(() => onEvent(fixtureMetricEvent()), 4000);
      signal.addEventListener("abort", () => {
        window.clearInterval(interval);
        resolve();
      }, { once: true });
    });
    return;
  }
  const headers = new Headers();
  const token = stored(ACCESS_KEY);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE}/stream/${type}`, { headers, signal });
  if (!response.ok) throw await responseError(response);
  if (!response.body) throw new ApiError("Event streaming is not available.", response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (!signal.aborted) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block.split(/\r?\n/).find((line) => line.startsWith("data:"));
      if (!data) continue;
      try {
        onEvent(JSON.parse(data.slice(5).trim()) as StreamEvent);
      } catch {
        throw new ApiError("The monitoring stream returned malformed event data.", response.status);
      }
    }
  }
}

export type { Project };
