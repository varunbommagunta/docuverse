import type { Document, QueryDebug } from "./types";

const BASE = "/api";

export interface QueryResponse {
  answer: string;
  citations: number[];
  retrieved_chunks: Array<{
    chunk_index: number;
    chunk_id: string;
    text: string;
    score: number;
    metadata: Record<string, unknown>;
  }>;
  rewritten_query?: string | null;
  debug?: QueryDebug | null;
}

export async function uploadDocument(
  file: File,
  sessionId: string
): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function sendQuery(
  query: string,
  history: Array<{ role: string; content: string }>,
  sessionId: string
): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, session_id: sessionId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Query failed: ${res.status}`);
  }
  return res.json();
}

export async function clearCorpus(sessionId: string): Promise<void> {
  await fetch(`${BASE}/corpus?session_id=${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

export async function getCorpusInfo(): Promise<{
  chunk_count: number;
  documents: Array<{ filename: string; chunk_count: number }>;
  is_preloaded: boolean;
}> {
  const res = await fetch(`${BASE}/corpus/info`);
  if (!res.ok) return { chunk_count: 0, documents: [], is_preloaded: false };
  return res.json();
}
