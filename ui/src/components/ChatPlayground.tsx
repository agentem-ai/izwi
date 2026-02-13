import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Send,
  Square,
  User,
  Loader2,
  ChevronDown,
  ChevronRight,
  Settings2,
} from "lucide-react";
import clsx from "clsx";
import { api, ChatMessage } from "../api";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface ChatPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady: boolean;
  supportsThinking: boolean;
  modelLabel?: string | null;
  modelOptions: ModelOption[];
  onSelectModel: (variant: string) => void;
  onOpenModelManager: () => void;
  onModelRequired: () => void;
}

const THINKING_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful assistant. Keep internal reasoning concise. If you use <think>, always close it with </think> and then provide a final answer.",
};

const DEFAULT_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content: "You are a helpful assistant.",
};

interface ParsedAssistantContent {
  thinking: string;
  answer: string;
  hasThink: boolean;
  hasIncompleteThink: boolean;
}

function parseAssistantContent(content: string): ParsedAssistantContent {
  const openTag = "<think>";
  const closeTag = "</think>";

  const thinkingParts: string[] = [];
  const answerParts: string[] = [];
  let cursor = 0;
  let hasIncompleteThink = false;

  while (true) {
    const openIdx = content.indexOf(openTag, cursor);
    if (openIdx === -1) {
      answerParts.push(content.slice(cursor));
      break;
    }

    answerParts.push(content.slice(cursor, openIdx));
    const thinkStart = openIdx + openTag.length;
    const closeIdx = content.indexOf(closeTag, thinkStart);

    if (closeIdx === -1) {
      thinkingParts.push(content.slice(thinkStart));
      hasIncompleteThink = true;
      break;
    }

    thinkingParts.push(content.slice(thinkStart, closeIdx));
    cursor = closeIdx + closeTag.length;
  }

  return {
    thinking: thinkingParts.join("\n\n").trim(),
    answer: answerParts.join("").trim(),
    hasThink: thinkingParts.length > 0,
    hasIncompleteThink,
  };
}

function getStatusTone(option: ModelOption): string {
  if (option.isReady) {
    return "text-emerald-300 bg-emerald-500/10 border-emerald-500/25";
  }
  if (
    option.statusLabel.toLowerCase().includes("downloading") ||
    option.statusLabel.toLowerCase().includes("loading")
  ) {
    return "text-amber-300 bg-amber-500/10 border-amber-500/25";
  }
  if (option.statusLabel.toLowerCase().includes("error")) {
    return "text-red-300 bg-red-500/10 border-red-500/25";
  }
  return "text-gray-300 bg-white/5 border-white/15";
}

export function ChatPlayground({
  selectedModel,
  selectedModelReady,
  supportsThinking,
  modelLabel,
  modelOptions,
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
}: ChatPlaygroundProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [expandedThoughts, setExpandedThoughts] = useState<
    Record<string, boolean>
  >({});
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    tokens_generated: number;
    generation_time_ms: number;
  } | null>(null);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);

  const streamAbortRef = useRef<AbortController | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const visibleMessages = useMemo(
    () => messages.filter((m) => m.role !== "system"),
    [messages],
  );
  const hasConversation = visibleMessages.length > 0 || isStreaming;

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return modelOptions.find((option) => option.value === selectedModel) || null;
  }, [selectedModel, modelOptions]);

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

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (
        modelMenuRef.current &&
        event.target instanceof Node &&
        !modelMenuRef.current.contains(event.target)
      ) {
        setIsModelMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, []);

  const stopStreaming = () => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
    setIsStreaming(false);
  };

  const sendMessage = () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    setError(null);
    setStats(null);

    const userMessage: ChatMessage = { role: "user", content: text };
    const assistantPlaceholder: ChatMessage = {
      role: "assistant",
      content: "",
    };

    const systemPrompt = supportsThinking
      ? THINKING_SYSTEM_PROMPT
      : DEFAULT_SYSTEM_PROMPT;
    const requestMessages = [
      systemPrompt,
      ...messages.filter((message) => message.role !== "system"),
      userMessage,
    ];

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setInput("");
    setIsStreaming(true);

    streamAbortRef.current = api.chatCompletionsStream(
      {
        model_id: selectedModel,
        messages: requestMessages,
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

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager();
  };

  const renderModelSelector = () => (
    <div className="relative z-40 inline-block w-[300px] max-w-[80vw]" ref={modelMenuRef}>
      <button
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={clsx(
          "h-9 w-full px-3 rounded-xl border inline-flex items-center justify-between gap-2 text-xs transition-colors",
          selectedOption?.isReady
            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
            : "border-white/20 bg-[#1a1a1a] text-gray-300 hover:border-white/30",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown className="w-3.5 h-3.5 shrink-0 opacity-80" />
      </button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className="absolute left-0 right-0 bottom-11 rounded-2xl border border-[#2a2a2a] bg-[#1a1a1a] p-2 shadow-2xl z-[90]"
          >
            <div className="max-h-64 overflow-y-auto pr-1 space-y-1">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={clsx(
                    "w-full text-left rounded-xl px-2.5 py-2 transition-colors border",
                    selectedOption?.value === option.value
                      ? "bg-[#24272c] border-[#464a52]"
                      : "bg-transparent border-transparent hover:bg-[#202328]",
                  )}
                >
                  <div className="text-xs text-gray-100 truncate">
                    {option.label}
                  </div>
                  <span
                    className={clsx(
                      "mt-1 inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px]",
                      getStatusTone(option),
                    )}
                  >
                    {option.statusLabel}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  const renderComposer = (centered = false) => (
    <div
      className={clsx(
        "relative rounded-[32px] border shadow-[0_20px_60px_rgba(0,0,0,0.45)]",
        centered ? "border-[#3b3c3f]" : "border-[#303236]",
      )}
    >
      <div className="rounded-[32px] overflow-visible bg-[linear-gradient(180deg,#1d1e21,#17181b)]">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              sendMessage();
            }
          }}
          placeholder={
            !selectedModel
              ? "Choose a model and ask anything..."
              : !selectedModelReady
                ? "Model selected but not loaded. Open Models to load it."
                : "Ask anything..."
          }
          className={clsx(
            "w-full bg-transparent px-5 pt-5 pb-3 text-sm text-white placeholder-gray-500 resize-none focus:outline-none",
            centered ? "min-h-[132px]" : "min-h-[96px]",
          )}
          disabled={isStreaming}
        />

        <div className="flex items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={handleOpenModels}
              className="inline-flex items-center gap-1.5 h-9 px-3 rounded-xl text-xs text-gray-200 border border-[#34363a] bg-white/[0.02] hover:bg-white/[0.05] hover:border-[#4a4c52] transition-colors"
            >
              <Settings2 className="w-3.5 h-3.5" />
              Models
            </button>
          </div>

          <div className="flex items-center gap-2">
            {renderModelSelector()}

            <button
              onClick={isStreaming ? stopStreaming : sendMessage}
              disabled={!isStreaming && !input.trim()}
              className="h-9 px-3 rounded-xl bg-white text-black text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
            >
              {isStreaming ? (
                <Square className="w-3.5 h-3.5" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              {isStreaming ? "Cancel" : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div
      className={clsx(
        "relative flex flex-col",
        hasConversation
          ? "h-[calc(100dvh-9rem)] lg:h-[calc(100dvh-6.5rem)] px-0 overflow-hidden"
          : "min-h-[calc(100vh-10rem)] sm:min-h-[calc(100vh-12rem)] px-4 sm:px-6",
      )}
    >
      {!hasConversation ? (
        <div className="relative flex-1 flex items-center justify-center">
          <div className="w-full max-w-3xl">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22 }}
            >
              {renderComposer(true)}
            </motion.div>

            <div className="mt-3 text-center text-xs min-h-[18px]">
              {selectedModel ? (
                selectedModelReady ? (
                  <span className="text-emerald-300">
                    {modelLabel || selectedModel} is loaded and ready.
                  </span>
                ) : (
                  <span className="text-gray-500">
                    {modelLabel || selectedModel} is selected but not loaded.
                  </span>
                )
              ) : (
                <span className="text-gray-500">No model selected.</span>
              )}
            </div>

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-3 mx-auto max-w-3xl p-2 rounded bg-red-950/50 border border-red-900/50 text-red-300 text-xs"
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      ) : (
        <div className="relative flex-1 min-h-0 flex flex-col">
          <div className="px-4 sm:px-6 pb-3 border-b border-[#1c1c1c]">
            <div>
              <h2 className="text-sm font-medium text-white">Conversation</h2>
              <p className="text-xs text-gray-500 mt-1">
                {selectedModelReady
                  ? `Using ${modelLabel || selectedModel}`
                  : "Model not loaded"}
              </p>
            </div>
          </div>

          <div className="relative flex-1 min-h-0">
            <div className="h-full overflow-y-auto px-4 sm:px-6 pb-56 pt-4">
              <div className="max-w-4xl mx-auto space-y-3">
                {visibleMessages.map((message, idx) => {
                  const isUser = message.role === "user";
                  const isLastAssistant =
                    !isUser && idx === visibleMessages.length - 1 && isStreaming;
                  const parsed = isUser
                    ? null
                    : supportsThinking
                      ? parseAssistantContent(message.content || "")
                      : null;
                  const messageKey = `${idx}-${message.role}`;
                  const isThoughtExpanded = !!expandedThoughts[messageKey];
                  const showStreamingThinking =
                    !isUser &&
                    !!parsed &&
                    isLastAssistant &&
                    parsed.thinking.length > 0 &&
                    (parsed.hasIncompleteThink || parsed.answer.length === 0);
                  const showAnswerOnly =
                    !isUser &&
                    !!parsed &&
                    parsed.answer.length > 0 &&
                    parsed.hasThink &&
                    !showStreamingThinking;

                  return (
                    <motion.div
                      key={messageKey}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={clsx("flex gap-3", isUser && "justify-end")}
                    >
                      {!isUser && (
                        <div className="w-7 h-7 rounded-lg bg-[#121212] border border-[#222] flex items-center justify-center flex-shrink-0">
                          <Bot className="w-4 h-4 text-gray-400" />
                        </div>
                      )}

                      <div
                        className={clsx(
                          "max-w-[85%] rounded-lg px-3 py-2.5 text-sm whitespace-pre-wrap break-words",
                          isUser
                            ? "bg-white text-black"
                            : "bg-[#17181c] border border-[#27282c] text-gray-200",
                        )}
                      >
                        {isUser ? (
                          message.content
                        ) : (
                          <>
                            {showStreamingThinking && parsed && (
                              <div className="mb-2 rounded border border-[#2f2f2f] bg-[#151515] px-2.5 py-2 text-xs text-gray-500">
                                <div className="mb-1.5 flex items-center gap-1.5 uppercase tracking-wide text-[10px] text-gray-600">
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                  Thinking
                                </div>
                                <div className="whitespace-pre-wrap text-gray-500">
                                  {parsed.thinking}
                                </div>
                              </div>
                            )}

                            {parsed && parsed.answer.length > 0 ? (
                              <div className="text-gray-200">{parsed.answer}</div>
                            ) : parsed && parsed.hasThink ? (
                              <div className="text-gray-500 italic">
                                {isLastAssistant
                                  ? "Thinking..."
                                  : "No final answer was generated."}
                              </div>
                            ) : (
                              <div className="text-gray-200">{message.content}</div>
                            )}

                            {parsed && parsed.hasThink && !showStreamingThinking && (
                              <div className="mt-2">
                                <button
                                  onClick={() =>
                                    setExpandedThoughts((prev) => ({
                                      ...prev,
                                      [messageKey]: !prev[messageKey],
                                    }))
                                  }
                                  className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                                >
                                  {isThoughtExpanded ? (
                                    <ChevronDown className="w-3 h-3" />
                                  ) : (
                                    <ChevronRight className="w-3 h-3" />
                                  )}
                                  {isThoughtExpanded
                                    ? "Hide thinking"
                                    : "Show thinking"}
                                </button>
                              </div>
                            )}

                            {parsed &&
                              parsed.hasThink &&
                              !showStreamingThinking &&
                              isThoughtExpanded && (
                                <div className="mt-2 rounded border border-[#2f2f2f] bg-[#151515] px-2.5 py-2 text-xs text-gray-500 whitespace-pre-wrap">
                                  {parsed.thinking}
                                </div>
                              )}

                            {isLastAssistant &&
                              ((parsed && parsed.answer.length > 0) ||
                                !showAnswerOnly) && (
                                <span className="inline-flex items-center ml-1">
                                  <Loader2 className="w-3 h-3 animate-spin text-gray-400" />
                                </span>
                              )}
                          </>
                        )}
                      </div>

                      {isUser && (
                        <div className="w-7 h-7 rounded-lg bg-white/10 border border-white/20 flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-gray-200" />
                        </div>
                      )}
                    </motion.div>
                  );
                })}
                <div ref={listEndRef} />
              </div>
            </div>

            <div className="fixed z-30 bottom-10 left-4 right-4 sm:left-6 sm:right-6 lg:left-[calc(16rem+2rem)] lg:right-8">
              <div className="max-w-4xl mx-auto">
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
                  <div className="mb-2 text-xs text-gray-500 text-center">
                    {stats.tokens_generated} tokens in{" "}
                    {Math.round(stats.generation_time_ms)} ms
                  </div>
                )}

                {renderComposer(false)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
