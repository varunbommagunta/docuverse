"use client";

import { useEffect, useRef, useState } from "react";
import type { Message } from "@/lib/types";

interface Props {
  messages: Message[];
  querying: boolean;
  internalsVisible: boolean;
  onToggleInternals: () => void;
  onSend: (text: string) => void;
}

export default function ChatPanel({
  messages,
  querying,
  internalsVisible,
  onToggleInternals,
  onSend,
}: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, querying]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || querying) return;
    setInput("");
    onSend(text);
  }

  return (
    <main className="flex-1 flex flex-col min-w-0 h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-white/[0.06] bg-white/[0.01]">
        <div className="flex items-center gap-2">
          <div className="w-1 h-4 rounded-full bg-gradient-to-b from-violet-500 to-indigo-500" />
          <span className="font-semibold text-white text-sm">Chat</span>
        </div>
        <button
          onClick={onToggleInternals}
          className={`text-[10px] px-2.5 py-1 rounded-md border transition-all duration-200 font-medium ${
            internalsVisible
              ? "bg-violet-500/15 border-violet-500/30 text-violet-300 shadow-sm shadow-violet-500/10"
              : "border-white/[0.08] text-slate-500 hover:border-white/[0.15] hover:text-slate-300"
          }`}
        >
          {internalsVisible ? "internals on" : "internals off"}
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-6 space-y-5">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 pb-12 animate-fadeIn">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-indigo-600/20 border border-violet-500/20 flex items-center justify-center shadow-xl shadow-violet-500/10">
              <svg className="w-8 h-8 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
              </svg>
            </div>
            <div className="text-center">
              <p className="text-slate-300 font-medium text-sm">Ask anything about your documents</p>
              <p className="text-slate-600 text-xs mt-1">Upload a PDF to get started</p>
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {querying && <ThinkingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="px-5 py-4 border-t border-white/[0.06] bg-white/[0.01]">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents…"
            disabled={querying}
            className="flex-1 bg-white/[0.05] border border-white/[0.08] rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-violet-500/50 focus:bg-white/[0.07] focus:shadow-lg focus:shadow-violet-500/10 transition-all duration-200 disabled:opacity-40"
          />
          <button
            type="submit"
            disabled={querying || !input.trim()}
            className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 active:scale-95 transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          </button>
        </form>
      </div>
    </main>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex animate-fadeIn">
      <div className="inline-flex items-center gap-1 bg-white/[0.04] border border-white/[0.07] rounded-2xl rounded-tl-sm px-3.5 py-2.5">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="w-1.5 h-1.5 rounded-full bg-violet-400"
            style={{ animation: `pulseDot 1.2s ease-in-out ${delay}ms infinite` }}
          />
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end animate-slideUp">
        <div className="max-w-[78%] bg-gradient-to-br from-violet-600 to-indigo-600 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-white shadow-lg shadow-violet-500/20 leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  const citedIndices = new Set(message.citations ?? []);

  return (
    <div className="flex flex-col gap-2.5 max-w-[88%] animate-slideUp">
      <div className="bg-white/[0.04] border border-white/[0.08] rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-slate-200 whitespace-pre-wrap leading-relaxed shadow-sm">
        {message.content}
      </div>
      {message.chunks && message.chunks.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pl-1">
          {message.chunks
            .filter((c) => citedIndices.has(c.chunk_index))
            .map((c) => {
              const pinned = message.debug?.chunks?.[c.chunk_index]?.pinned;
              const artId = c.metadata?.article_id as string | undefined;
              const fname = c.metadata?.filename as string | undefined;
              return (
                <span
                  key={c.chunk_id}
                  className="inline-flex items-center gap-1.5 text-[10px] px-2 py-1 rounded-lg border border-white/[0.07] bg-white/[0.03] text-slate-500 hover:bg-white/[0.06] transition-colors duration-150 cursor-default"
                  title={c.text.slice(0, 200)}
                >
                  <span className="font-mono text-slate-600">#{c.chunk_index}</span>
                  {pinned && (
                    <span className="bg-violet-500/15 text-violet-300 px-1 rounded font-medium">pinned</span>
                  )}
                  <span className="text-violet-400/70 font-mono">{c.score.toFixed(3)}</span>
                  {fname && <span className="truncate max-w-[80px] text-slate-600">{fname}</span>}
                  {artId && <span className="text-slate-700">§{artId}</span>}
                </span>
              );
            })}
        </div>
      )}
    </div>
  );
}
