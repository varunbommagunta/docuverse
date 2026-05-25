"use client";

import { useEffect, useState } from "react";
import DocumentSidebar from "@/components/DocumentSidebar";
import ChatPanel from "@/components/ChatPanel";
import InternalsPanel from "@/components/InternalsPanel";
import { clearCorpus, getCorpusInfo, sendQuery, uploadDocument } from "@/lib/api";
import type { Document, Message, QueryDebug } from "@/lib/types";

function generateId() {
  return typeof crypto !== "undefined"
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
}

export default function Home() {
  const [sessionId] = useState(() => generateId());
  const [documents, setDocuments] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [internalsVisible, setInternalsVisible] = useState(false);
  const [lastQueryDebug, setLastQueryDebug] = useState<QueryDebug | null>(null);
  const [lastIngestion, setLastIngestion] = useState<Document | null>(null);
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);

  // Detect pre-loaded corpus on mount
  useEffect(() => {
    getCorpusInfo().then((info) => {
      if (info.is_preloaded && info.documents.length > 0) {
        const preloaded: Document[] = info.documents.map((d) => ({
          document_id: d.filename,
          filename: d.filename,
          chunk_count: d.chunk_count,
          document_type: d.filename.includes("constitution") ? "legal" : "academic",
          classification_confidence: 0.9,
          classification_method: "rules",
          chunker_used: d.filename.includes("constitution") ? "legal" : "academic",
        }));
        setDocuments(preloaded);
      }
    });
  }, []);

  async function handleUpload(file: File) {
    setUploading(true);
    try {
      const doc = await uploadDocument(file, sessionId);
      setDocuments((prev) => [...prev, doc]);
      setLastIngestion(doc);
    } catch (err) {
      console.error(err);
    } finally {
      setUploading(false);
    }
  }

  async function handleClear() {
    await clearCorpus(sessionId);
    setDocuments([]);
    setMessages([]);
    setLastQueryDebug(null);
    setLastIngestion(null);
  }

  async function handleSend(text: string) {
    const userMsg: Message = { id: generateId(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setQuerying(true);

    // Build history from prior messages (exclude the one we just appended)
    const history = messages.slice(-6).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const data = await sendQuery(text, history, sessionId);
      const assistantMsg: Message = {
        id: generateId(),
        role: "assistant",
        content: data.answer,
        citations: data.citations,
        chunks: data.retrieved_chunks,
        debug: data.debug ?? undefined,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      if (data.debug) setLastQueryDebug(data.debug);
    } catch (err) {
      const errMsg: Message = {
        id: generateId(),
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : String(err)}`,
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setQuerying(false);
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <DocumentSidebar
        documents={documents}
        uploading={uploading}
        onUpload={handleUpload}
        onClear={handleClear}
      />
      <ChatPanel
        messages={messages}
        querying={querying}
        internalsVisible={internalsVisible}
        onToggleInternals={() => setInternalsVisible((v) => !v)}
        onSend={handleSend}
      />
      {internalsVisible && (
        <InternalsPanel queryDebug={lastQueryDebug} lastIngestion={lastIngestion} />
      )}
    </div>
  );
}
