import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bot, Send, Square, RotateCcw, User, Loader2 } from "lucide-react";
import clsx from "clsx";
import { api, ChatMessage } from "../api";

interface ChatPlaygroundProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

const DEFAULT_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content: "You are a helpful assistant.",
};

export function ChatPlayground({
  selectedModel,
  onModelRequired,
}: ChatPlaygroundProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    tokens_generated: number;
    generation_time_ms: number;
  } | null>(null);

  const streamAbortRef = useRef<AbortController | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);

  const visibleMessages = useMemo(
    () => messages.filter((m) => m.role !== "system"),
    [messages],
  );

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visibleMessages, isStreaming]);

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
    };
  }, []);

  const stopStreaming = () => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
    setIsStreaming(false);
  };

  const clearChat = () => {
    stopStreaming();
    setMessages([]);
    setError(null);
    setStats(null);
  };

  const sendMessage = () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    if (!selectedModel) {
      onModelRequired();
      return;
    }

    setError(null);
    setStats(null);

    const userMessage: ChatMessage = { role: "user", content: text };
    const assistantPlaceholder: ChatMessage = { role: "assistant", content: "" };

    const historyWithSystem =
      messages.length > 0 && messages[0].role === "system"
        ? messages
        : [DEFAULT_SYSTEM_PROMPT, ...messages];
    const requestMessages = [...historyWithSystem, userMessage];

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setInput("");
    setIsStreaming(true);

    streamAbortRef.current = api.chatCompletionsStream(
      {
        model_id: selectedModel,
        messages: requestMessages,
        max_tokens: 512,
      },
      {
        onDelta: (delta) => {
          setMessages((prev) => {
            if (prev.length === 0) return prev;
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              updated[updated.length - 1] = {
                ...last,
                content: `${last.content}${delta}`,
              };
            }
            return updated;
          });
        },
        onDone: (message, streamStats) => {
          setMessages((prev) => {
            if (prev.length === 0) return prev;
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              updated[updated.length - 1] = {
                ...last,
                content: message,
              };
            }
            return updated;
          });
          setStats(streamStats);
          setIsStreaming(false);
          streamAbortRef.current = null;
        },
        onError: (message) => {
          setError(message);
          setIsStreaming(false);
          streamAbortRef.current = null;
        },
      },
    );
  };

  return (
    <div className="card p-4 flex flex-col h-[calc(100vh-12rem)] min-h-[560px]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/20">
            <Bot className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">Chat</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Text conversation with Qwen3
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isStreaming && (
            <button onClick={stopStreaming} className="btn btn-secondary text-xs">
              <Square className="w-3.5 h-3.5" />
              Stop
            </button>
          )}
          <button
            onClick={clearChat}
            className="btn btn-ghost text-xs"
            disabled={messages.length === 0 && !isStreaming}
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Clear
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto pr-1 space-y-3 mb-4">
        {visibleMessages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-sm">
              <p className="text-sm text-gray-400 mb-2">
                Ask anything to start chatting.
              </p>
              <p className="text-xs text-gray-600">
                Load the Qwen3 chat model and send your first message.
              </p>
            </div>
          </div>
        ) : (
          visibleMessages.map((message, idx) => {
            const isUser = message.role === "user";
            const isLastAssistant =
              !isUser && idx === visibleMessages.length - 1 && isStreaming;

            return (
              <motion.div
                key={`${message.role}-${idx}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={clsx("flex gap-3", isUser && "justify-end")}
              >
                {!isUser && (
                  <div className="w-7 h-7 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-cyan-300" />
                  </div>
                )}

                <div
                  className={clsx(
                    "max-w-[85%] rounded-lg px-3 py-2.5 text-sm whitespace-pre-wrap break-words",
                    isUser
                      ? "bg-white text-black"
                      : "bg-[#1a1a1a] border border-[#2a2a2a] text-gray-200",
                  )}
                >
                  {message.content}
                  {isLastAssistant && (
                    <span className="inline-flex items-center ml-1">
                      <Loader2 className="w-3 h-3 animate-spin text-gray-400" />
                    </span>
                  )}
                </div>

                {isUser && (
                  <div className="w-7 h-7 rounded-lg bg-white/10 border border-white/20 flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-gray-200" />
                  </div>
                )}
              </motion.div>
            );
          })
        )}
        <div ref={listEndRef} />
      </div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-3 p-2 rounded bg-red-950/50 border border-red-900/50 text-red-300 text-xs"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {stats && !isStreaming && (
        <div className="mb-3 text-xs text-gray-500">
          {stats.tokens_generated} tokens in{" "}
          {Math.round(stats.generation_time_ms)} ms
        </div>
      )}

      <div className="border border-[#2a2a2a] rounded-lg bg-[#141414]">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Message Qwen3..."
          className="w-full bg-transparent px-3 py-3 text-sm text-white placeholder-gray-600 resize-none focus:outline-none min-h-[80px]"
          disabled={isStreaming}
        />
        <div className="flex items-center justify-between px-3 py-2 border-t border-[#222]">
          <span className="text-xs text-gray-600">Enter to send</span>
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isStreaming}
            className="btn btn-primary text-xs"
          >
            <Send className="w-3.5 h-3.5" />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
