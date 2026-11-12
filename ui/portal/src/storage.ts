import type { ChatThread, Project } from "./types";

function dataKey(userId: string, key: string): string {
  return `dgb:${encodeURIComponent(userId)}:${key}`;
}

function read<T>(userId: string, key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(dataKey(userId, key));
    return raw ? JSON.parse(raw) as T : fallback;
  } catch {
    return fallback;
  }
}

function write<T>(userId: string, key: string, value: T): void {
  localStorage.setItem(dataKey(userId, key), JSON.stringify(value));
}

export const userStorage = {
  threads: (userId: string): ChatThread[] =>
    read<ChatThread[]>(userId, "threads", []).sort((a, b) => b.updatedAt - a.updatedAt),
  saveThreads: (userId: string, threads: ChatThread[]) => write(userId, "threads", threads),
  projects: (userId: string): Project[] => read<Project[]>(userId, "projects", []),
  saveProjects: (userId: string, projects: Project[]) => write(userId, "projects", projects)
};
