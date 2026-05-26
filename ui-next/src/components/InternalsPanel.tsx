"use client";

import { useState } from "react";
import type { QueryDebug, Document } from "@/lib/types";

interface Props {
  queryDebug: QueryDebug | null;
  lastIngestion: Document | null;
}

export default function InternalsPanel({ queryDebug, lastIngestion }: Props) {
  return (
    <aside className="w-[280px] flex-shrink-0 border-l border-white/[0.06] flex flex-col h-full overflow-y-auto bg-white/[0.01]">
      <div className="px-4 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <div className="w-1 h-4 rounded-full bg-gradient-to-b from-teal-400 to-cyan-500" />
          <span className="font-semibold text-sm text-white">Pipeline Internals</span>
        </div>
      </div>
      <div className="flex flex-col gap-3 px-3 py-3">
        <QueryPipelineCard debug={queryDebug} />
        <RetrievedChunksCard debug={queryDebug} />
        <IngestionCard doc={lastIngestion} />
      </div>
    </aside>
  );
}

function QueryPipelineCard({ debug }: { debug: QueryDebug | null }) {
  const steps: Array<{ label: string; value: string; done: boolean }> = debug
    ? [
        { label: "Original query",     value: debug.original_query,    done: true },
        {
          label: "Rewritten query",
          value: debug.rewritten_query === debug.original_query ? "(unchanged)" : debug.rewritten_query,
          done: true,
        },
        { label: "Retrieval strategy", value: debug.retrieval_strategy, done: true },
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
    <Card gradient="from-violet-500 to-indigo-500" icon={<PipelineIcon />} title="Query pipeline">
      {!debug ? (
        <Empty />
      ) : (
        <ul className="space-y-2.5">
          {steps.map((s) => (
            <li key={s.label} className="flex gap-2.5 items-start">
              <span
                className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                  s.done ? "bg-emerald-400 shadow-sm shadow-emerald-400/50" : "bg-white/10"
                }`}
              />
              <div className="min-w-0">
                <p className="text-[9px] text-slate-600 uppercase tracking-wider font-medium leading-none">{s.label}</p>
                <p className="text-[11px] text-slate-300 mt-0.5 break-words leading-relaxed">{s.value}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function RetrievedChunksCard({ debug }: { debug: QueryDebug | null }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const chunks = debug?.chunks ?? [];
  const maxScore = chunks.reduce((m, c) => Math.max(m, c.score), 0.001);

  return (
    <Card gradient="from-teal-500 to-cyan-500" icon={<ChunksIcon />} title="Retrieved chunks">
      {chunks.length === 0 ? (
        <Empty />
      ) : (
        <ul className="space-y-3">
          {chunks.map((c) => {
            const isExpanded = expandedId === c.id;
            return (
              <li
                key={c.id}
                className="space-y-1.5 cursor-pointer select-none rounded-lg p-1.5 -mx-1.5 transition-colors hover:bg-white/[0.03]"
                onClick={() => setExpandedId(isExpanded ? null : c.id)}
              >
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="font-mono text-[10px] text-slate-400">{c.id}</span>
                  {c.pinned && (
                    <span className="text-[9px] px-1.5 py-0.5 bg-violet-500/15 text-violet-300 rounded-md font-medium">pinned</span>
                  )}
                  <span className="text-[10px] text-teal-400 ml-auto font-mono">{c.score.toFixed(3)}</span>
                  <svg
                    className={`w-3 h-3 text-slate-500 flex-shrink-0 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                    fill="none" stroke="currentColor" viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
                <div className="text-[10px] text-slate-600 truncate">
                  {c.source}{c.article_id ? ` §${c.article_id}` : ""}{c.section_title ? ` / ${c.section_title}` : ""}
                </div>
                {isExpanded && c.text ? (
                  <p className="text-[11px] text-slate-300 leading-relaxed whitespace-pre-wrap break-words">{c.text}</p>
                ) : (
                  <p className="text-[11px] text-slate-400 leading-relaxed line-clamp-2">{c.preview}</p>
                )}
                <div className="w-full h-1 bg-white/[0.05] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-teal-500 to-cyan-400 rounded-full"
                    style={{ width: `${(c.score / maxScore) * 100}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}

function IngestionCard({ doc }: { doc: Document | null }) {
  return (
    <Card gradient="from-amber-500 to-orange-500" icon={<IngestIcon />} title="Last ingestion">
      {!doc ? (
        <Empty />
      ) : (
        <ul className="space-y-2">
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

function Card({
  gradient,
  icon,
  title,
  children,
}: {
  gradient: string;
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-white/[0.06]">
        <div className={`w-5 h-5 rounded-md bg-gradient-to-br ${gradient} flex items-center justify-center opacity-90 flex-shrink-0`}>
          {icon}
        </div>
        <span className="text-xs font-medium text-slate-300">{title}</span>
      </div>
      <div className="px-3 py-3">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <li className="flex justify-between gap-2 text-xs">
      <span className="text-slate-600 text-[10px] uppercase tracking-wide font-medium">{label}</span>
      <span className="text-slate-300 text-[11px] font-medium truncate max-w-[140px] text-right" title={value}>{value}</span>
    </li>
  );
}

function Empty() {
  return <p className="text-[11px] text-slate-600">No data yet.</p>;
}

function PipelineIcon() {
  return (
    <svg fill="none" stroke="white" viewBox="0 0 24 24" className="w-3 h-3">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
    </svg>
  );
}

function ChunksIcon() {
  return (
    <svg fill="none" stroke="white" viewBox="0 0 24 24" className="w-3 h-3">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  );
}

function IngestIcon() {
  return (
    <svg fill="none" stroke="white" viewBox="0 0 24 24" className="w-3 h-3">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
    </svg>
  );
}
