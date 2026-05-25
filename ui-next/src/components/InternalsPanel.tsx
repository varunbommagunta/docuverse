"use client";

import type { QueryDebug } from "@/lib/types";
import type { Document } from "@/lib/types";

interface Props {
  queryDebug: QueryDebug | null;
  lastIngestion: Document | null;
}

export default function InternalsPanel({ queryDebug, lastIngestion }: Props) {
  return (
    <aside className="w-[280px] flex-shrink-0 border-l border-gray-200 flex flex-col h-full overflow-y-auto">
      <div className="px-4 py-3 border-b border-gray-200">
        <span className="font-semibold text-sm">Pipeline Internals</span>
      </div>

      <div className="flex flex-col gap-3 px-3 py-3">
        <QueryPipelineCard debug={queryDebug} />
        <RetrievedChunksCard debug={queryDebug} />
        <IngestionCard doc={lastIngestion} />
      </div>
    </aside>
  );
}

/* ── Query pipeline card ─────────────────────────────────────────────────── */

function QueryPipelineCard({ debug }: { debug: QueryDebug | null }) {
  const af = debug?.article_filter;

  const steps: Array<{ label: string; value: string; done: boolean }> = debug
    ? [
        { label: "Original query",       value: debug.original_query,          done: true },
        {
          label: "Rewritten query",
          value: debug.rewritten_query === debug.original_query
            ? "(unchanged)"
            : debug.rewritten_query,
          done: true,
        },
        {
          label: "Article filter",
          value: af?.matched
            ? `article_id = ${af.article_id} → ${af.pinned_count} chunk${af.pinned_count !== 1 ? "s" : ""} pinned`
            : "no match",
          done: true,
        },
        { label: "Retrieval strategy",   value: debug.retrieval_strategy,      done: true },
        {
          label: "Reranker",
          value: debug.reranker
            ? `${debug.reranker.candidates_in} candidates → ${debug.reranker.results_out} kept`
            : "n/a",
          done: !!debug.reranker,
        },
      ]
    : [];

  return (
    <Card icon={<PipelineIcon />} title="Query pipeline">
      {!debug ? (
        <Empty />
      ) : (
        <ul className="space-y-2">
          {steps.map((s) => (
            <li key={s.label} className="flex gap-2 items-start">
              <span className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${s.done ? "bg-green-500" : "bg-gray-300"}`} />
              <div className="min-w-0">
                <p className="text-[10px] text-gray-400 leading-none">{s.label}</p>
                <p className="text-xs mt-0.5 break-words">{s.value}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/* ── Retrieved chunks card ───────────────────────────────────────────────── */

function RetrievedChunksCard({ debug }: { debug: QueryDebug | null }) {
  const chunks = debug?.chunks ?? [];
  const maxScore = chunks.reduce((m, c) => Math.max(m, c.score), 0.001);

  return (
    <Card icon={<ChunksIcon />} title="Retrieved chunks">
      {chunks.length === 0 ? (
        <Empty />
      ) : (
        <ul className="space-y-2.5">
          {chunks.map((c) => (
            <li key={c.id} className="space-y-1">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="font-mono text-[10px] text-gray-600">{c.id}</span>
                {c.pinned && (
                  <span className="text-[9px] px-1 py-0.5 bg-purple-100 text-purple-700 rounded">pinned</span>
                )}
                <span className="text-[10px] text-gray-400 ml-auto">{c.score.toFixed(3)}</span>
              </div>
              <div className="text-[10px] text-gray-500 truncate">
                {c.source}{c.article_id ? ` §${c.article_id}` : ""}{c.section_title ? ` / ${c.section_title}` : ""}
              </div>
              <p className="text-[11px] text-gray-600 leading-relaxed line-clamp-2">{c.preview}</p>
              {/* Score bar */}
              <div className="w-full h-1 bg-gray-100 rounded-full">
                <div
                  className="h-full bg-green-500 rounded-full"
                  style={{ width: `${(c.score / maxScore) * 100}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/* ── Ingestion card ──────────────────────────────────────────────────────── */

function IngestionCard({ doc }: { doc: Document | null }) {
  return (
    <Card icon={<IngestIcon />} title="Last ingestion">
      {!doc ? (
        <Empty />
      ) : (
        <ul className="space-y-1.5 text-xs">
          <Row label="File"       value={doc.filename} />
          <Row label="Type"       value={doc.document_type} />
          <Row label="Confidence" value={`${(doc.classification_confidence * 100).toFixed(0)}%`} />
          <Row label="Method"     value={doc.classification_method} />
          <Row label="Chunker"    value={doc.chunker_used} />
          <Row label="Chunks"     value={String(doc.chunk_count)} />
        </ul>
      )}
    </Card>
  );
}

/* ── Shared primitives ───────────────────────────────────────────────────── */

function Card({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 border-b border-gray-200">
        <span className="text-gray-400 w-4 h-4">{icon}</span>
        <span className="text-xs font-medium">{title}</span>
      </div>
      <div className="px-3 py-3">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <li className="flex justify-between gap-2">
      <span className="text-gray-400">{label}</span>
      <span className="text-right font-medium truncate max-w-[140px]" title={value}>{value}</span>
    </li>
  );
}

function Empty() {
  return <p className="text-xs text-gray-400">No data yet.</p>;
}

function PipelineIcon() {
  return (
    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-4 h-4">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
    </svg>
  );
}

function ChunksIcon() {
  return (
    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-4 h-4">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  );
}

function IngestIcon() {
  return (
    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-4 h-4">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
    </svg>
  );
}
