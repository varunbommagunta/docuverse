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
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200">
        <span className="font-semibold">Chat</span>
        <button
          onClick={onToggleInternals}
          className={`text-xs px-3 py-1.5 rounded border transition-colors ${
            internalsVisible
              ? "bg-sky-50 border-sky-300 text-sky-700"
              : "border-gray-200 text-gray-500 hover:border-gray-300"
          }`}
        >
          {internalsVisible ? "internals on" : "internals off"}
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-gray-400 text-center mt-12 text-sm">
            Upload a PDF and ask a question.
          </p>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {querying && (
          <div className="flex gap-1 pl-1">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 px-5 py-3 border-t border-gray-200"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your documents…"
          disabled={querying}
          className="flex-1 border border-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-gray-400 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={querying || !input.trim()}
          className="px-4 py-2 bg-gray-900 text-white text-sm rounded-md hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Send
        </button>
      </form>
    </main>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-gray-100 rounded-lg px-4 py-2.5 text-sm">
          {message.content}
        </div>
      </div>
    );
  }

  const citedIndices = new Set(message.citations ?? []);

  return (
    <div className="flex flex-col gap-2 max-w-[90%]">
      <div className="bg-white border border-gray-200 rounded-lg px-4 py-3 text-sm whitespace-pre-wrap leading-relaxed">
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
                  className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded border border-gray-200 text-gray-600 bg-gray-50"
                  title={c.text.slice(0, 200)}
                >
                  <span className="font-mono">chunk_{c.chunk_index}</span>
                  {pinned && (
                    <span className="bg-purple-100 text-purple-700 px-1 rounded">pinned</span>
                  )}
                  <span className="text-gray-400">{c.score.toFixed(3)}</span>
                  {fname && <span className="truncate max-w-[80px]">{fname}</span>}
                  {artId && <span className="text-gray-400">§{artId}</span>}
                </span>
              );
            })}
        </div>
      )}
    </div>
  );
}
