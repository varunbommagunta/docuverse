"use client";

import { useRef } from "react";
import type { Document } from "@/lib/types";

const TYPE_BADGE: Record<string, string> = {
  legal:    "bg-purple-100 text-purple-700",
  academic: "bg-teal-100 text-teal-700",
  prose:    "bg-blue-100 text-blue-700",
  technical:"bg-amber-100 text-amber-700",
  default:  "bg-gray-100 text-gray-600",
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
    <aside className="w-[220px] flex-shrink-0 border-r border-gray-200 flex flex-col h-full">
      {/* Logo */}
      <div className="px-4 py-4 border-b border-gray-200">
        <span className="font-semibold text-base tracking-tight">DocuVerse</span>
        <p className="text-xs text-gray-400 mt-0.5">Hybrid RAG · Cited Answers</p>
      </div>

      {/* Upload zone */}
      <div className="px-3 pt-3">
        <button
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="w-full border-2 border-dashed border-gray-300 rounded-lg p-4 flex flex-col items-center gap-1.5 text-gray-500 hover:border-gray-400 hover:text-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <span className="text-xs font-medium">
            {uploading ? "Ingesting…" : "Upload PDF"}
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={handleFiles}
        />
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        {documents.length === 0 ? (
          <p className="text-xs text-gray-400 text-center mt-4">No documents yet</p>
        ) : (
          documents.map((doc) => (
            <div key={doc.document_id} className="border border-gray-200 rounded-md p-2">
              <p className="text-xs font-medium truncate" title={doc.filename}>
                {doc.filename}
              </p>
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs text-gray-400">{doc.chunk_count} chunks</span>
                <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${TYPE_BADGE[doc.document_type] ?? TYPE_BADGE.default}`}>
                  {doc.document_type}
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Clear button */}
      <div className="px-3 pb-3 pt-2 border-t border-gray-200">
        <button
          onClick={onClear}
          className="w-full text-xs text-gray-500 hover:text-red-600 py-1.5 border border-gray-200 rounded-md hover:border-red-300 transition-colors"
        >
          Clear corpus
        </button>
      </div>
    </aside>
  );
}
