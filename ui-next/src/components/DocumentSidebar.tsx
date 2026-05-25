"use client";

import { useRef } from "react";
import type { Document } from "@/lib/types";

const TYPE_COLORS: Record<string, { pill: string; dot: string }> = {
  legal:     { pill: "bg-violet-500/10 text-violet-300",  dot: "bg-violet-400" },
  academic:  { pill: "bg-teal-500/10 text-teal-300",      dot: "bg-teal-400" },
  prose:     { pill: "bg-blue-500/10 text-blue-300",      dot: "bg-blue-400" },
  technical: { pill: "bg-amber-500/10 text-amber-300",    dot: "bg-amber-400" },
  default:   { pill: "bg-slate-500/10 text-slate-400",    dot: "bg-slate-500" },
};

interface Props {
  documents: Document[];
  uploading: boolean;
  onUpload: (file: File) => void;
  onClear: () => void;
}

export default function DocumentSidebar({ documents, uploading, onUpload, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    e.target.value = "";
  }

  return (
    <aside className="w-[220px] flex-shrink-0 flex flex-col h-full border-r border-white/[0.06] bg-white/[0.015]">
      {/* Logo */}
      <div className="px-4 py-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/30 flex-shrink-0">
            <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <p className="font-bold text-sm text-white tracking-tight leading-none">DocuVerse</p>
            <p className="text-[9px] text-slate-500 mt-0.5">Hybrid RAG · Cited Answers</p>
          </div>
        </div>
      </div>

      {/* Upload zone */}
      <div className="px-3 pt-3">
        <button
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="w-full relative rounded-xl border border-dashed border-white/[0.1] hover:border-violet-500/40 p-4 flex flex-col items-center gap-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed group overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-violet-600/0 to-indigo-600/0 group-hover:from-violet-600/[0.06] group-hover:to-indigo-600/[0.06] transition-all duration-300 rounded-xl" />
          <div className="relative w-9 h-9 rounded-lg bg-white/[0.04] border border-white/[0.08] flex items-center justify-center group-hover:bg-violet-500/10 group-hover:border-violet-500/25 transition-all duration-200">
            {uploading ? (
              <svg className="w-4 h-4 text-violet-400 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4 text-slate-500 group-hover:text-violet-400 transition-colors duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            )}
          </div>
          <span className="relative text-xs font-medium text-slate-400 group-hover:text-violet-300 transition-colors duration-200">
            {uploading ? "Ingesting…" : "Upload PDF"}
          </span>
          {!uploading && (
            <span className="relative text-[9px] text-slate-600">up to 50 MB</span>
          )}
        </button>
        <input ref={inputRef} type="file" accept=".pdf" className="hidden" onChange={handleFiles} />
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-1.5">
        {documents.length === 0 ? (
          <div className="flex flex-col items-center mt-8 gap-2.5">
            <div className="w-10 h-10 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
              <svg className="w-5 h-5 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <p className="text-[10px] text-slate-600 text-center">No documents yet</p>
          </div>
        ) : (
          documents.map((doc) => {
            const colors = TYPE_COLORS[doc.document_type] ?? TYPE_COLORS.default;
            return (
              <div
                key={doc.document_id}
                className="rounded-lg border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/[0.1] transition-all duration-150 p-2.5 animate-slideUp"
              >
                <p className="text-xs font-medium text-slate-200 truncate" title={doc.filename}>
                  {doc.filename}
                </p>
                <div className="flex items-center justify-between mt-1.5 gap-1">
                  <span className="text-[10px] text-slate-600">{doc.chunk_count} chunks</span>
                  <span className={`inline-flex items-center gap-1 text-[9px] font-medium px-1.5 py-0.5 rounded-md ${colors.pill}`}>
                    <span className={`w-1 h-1 rounded-full flex-shrink-0 ${colors.dot}`} />
                    {doc.document_type}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Clear button */}
      <div className="px-3 pb-3 pt-2 border-t border-white/[0.06]">
        <button
          onClick={onClear}
          className="w-full text-[10px] text-slate-600 hover:text-red-400 py-1.5 border border-white/[0.06] hover:border-red-500/25 hover:bg-red-500/[0.04] rounded-lg transition-all duration-150"
        >
          Clear corpus
        </button>
      </div>
    </aside>
  );
}
