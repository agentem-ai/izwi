import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  Plus,
  Trash2,
  MessageSquare,
} from "lucide-react";
import clsx from "clsx";
import { api, ChatMessage, ChatThread, ChatThreadMessageRecord } from "../api";
import { MarkdownContent } from "./ui/MarkdownContent";

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
    return "chat-model-status-ready";
  }
  if (
    option.statusLabel.toLowerCase().includes("downloading") ||
    option.statusLabel.toLowerCase().includes("loading")
  ) {
    return "chat-model-status-loading";
  }
  if (option.statusLabel.toLowerCase().includes("error")) {
    return "chat-model-status-error";
  }
  return "chat-model-status-idle";
}

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return fallback;
}

function formatThreadTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  if (isToday) {
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function extractLatestStats(messages: ChatThreadMessageRecord[]): {
  tokens_generated: number;
  generation_time_ms: number;
} | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (
      message.role === "assistant" &&
      typeof message.tokens_generated === "number" &&
      typeof message.generation_time_ms === "number"
    ) {
      return {
        tokens_generated: message.tokens_generated,
        generation_time_ms: message.generation_time_ms,
      };
    }
  }
  return null;
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
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatThreadMessageRecord[]>([]);
  const [expandedThoughts, setExpandedThoughts] = useState<
    Record<string, boolean>
  >({});
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingThreadId, setStreamingThreadId] = useState<string | null>(null);
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    tokens_generated: number;
    generation_time_ms: number;
  } | null>(null);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);

  const initializedRef = useRef(false);
  const activeThreadIdRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return modelOptions.find((option) => option.value === selectedModel) || null;
  }, [selectedModel, modelOptions]);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) ?? null,
    [threads, activeThreadId],
  );

  const visibleMessages = useMemo(
    () => messages.filter((message) => message.role !== "system"),
    [messages],
  );

  const hasConversation =
    visibleMessages.length > 0 || isStreaming || messagesLoading;

  const refreshThreadList = useCallback(async (preferredThreadId?: string) => {
    try {
      const listedThreads = await api.listChatThreads();
      if (listedThreads.length === 0) {
        setThreads([]);
        setActiveThreadId(null);
        return;
      }

      setThreads(listedThreads);
      setActiveThreadId((previousThreadId) => {
        const candidateThreadId = preferredThreadId ?? previousThreadId;
        if (
          candidateThreadId &&
          listedThreads.some((thread) => thread.id === candidateThreadId)
        ) {
          return candidateThreadId;
        }
        return listedThreads[0].id;
      });
    } catch {
      // Keep current thread state on refresh failures.
    }
  }, []);

  useEffect(() => {
    if (initializedRef.current) {
      return;
    }
    initializedRef.current = true;

    const initializeThreads = async () => {
      setThreadsLoading(true);
      try {
        const listedThreads = await api.listChatThreads();
        if (listedThreads.length === 0) {
          const newThread = await api.createChatThread({
            model_id: selectedModel ?? undefined,
          });
          setThreads([newThread]);
          setActiveThreadId(newThread.id);
        } else {
          setThreads(listedThreads);
          setActiveThreadId(listedThreads[0].id);
        }
      } catch (loadError) {
        setError(getErrorMessage(loadError, "Failed to load chat threads."));
      } finally {
        setThreadsLoading(false);
      }
    };

    void initializeThreads();
  }, [selectedModel]);

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visibleMessages, isStreaming, activeThreadId]);

  useEffect(() => {
    activeThreadIdRef.current = activeThreadId;
  }, [activeThreadId]);

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

  useEffect(() => {
    if (!activeThreadId) {
      setMessages([]);
      setExpandedThoughts({});
      setStats(null);
      return;
    }

    let cancelled = false;

    const loadThread = async () => {
      setMessagesLoading(true);
      try {
        const detail = await api.getChatThread(activeThreadId);
        if (cancelled) {
          return;
        }
        setMessages(detail.messages);
        setExpandedThoughts({});
        setStats(extractLatestStats(detail.messages));
        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === detail.thread.id ? detail.thread : thread,
          ),
        );
      } catch (loadError) {
        if (!cancelled) {
          setError(getErrorMessage(loadError, "Failed to load this conversation."));
        }
      } finally {
        if (!cancelled) {
          setMessagesLoading(false);
        }
      }
    };

    void loadThread();

    return () => {
      cancelled = true;
    };
  }, [activeThreadId]);

  const stopStreaming = useCallback(() => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }

    setIsStreaming(false);
    setStreamingThreadId(null);

    if (activeThreadId) {
      void api
        .getChatThread(activeThreadId)
        .then((detail) => {
          if (detail.thread.id !== activeThreadId) {
            return;
          }
          setMessages(detail.messages);
          setStats(extractLatestStats(detail.messages));
        })
        .catch(() => {
          // Ignore follow-up sync failures after cancel.
        });
      void refreshThreadList(activeThreadId);
    }
  }, [activeThreadId, refreshThreadList]);

  const handleCreateThread = useCallback(async () => {
    if (isStreaming) {
      return;
    }

    try {
      const thread = await api.createChatThread({
        model_id: selectedModel ?? undefined,
      });
      setThreads((previous) => [thread, ...previous]);
      setActiveThreadId(thread.id);
      setMessages([]);
      setExpandedThoughts({});
      setStats(null);
      setError(null);
      setInput("");
    } catch (createError) {
      setError(getErrorMessage(createError, "Failed to create a new chat."));
    }
  }, [isStreaming, selectedModel]);

  const handleDeleteThread = useCallback(
    async (threadId: string) => {
      if (isStreaming) {
        return;
      }

      try {
        await api.deleteChatThread(threadId);

        const remainingThreads = threads.filter((thread) => thread.id !== threadId);
        if (remainingThreads.length === 0) {
          const replacementThread = await api.createChatThread({
            model_id: selectedModel ?? undefined,
          });
          setThreads([replacementThread]);
          setActiveThreadId(replacementThread.id);
          setMessages([]);
          setExpandedThoughts({});
          setStats(null);
          return;
        }

        setThreads(remainingThreads);
        if (activeThreadId === threadId) {
          setActiveThreadId(remainingThreads[0].id);
        }
      } catch (deleteError) {
        setError(getErrorMessage(deleteError, "Failed to delete this chat."));
      }
    },
    [activeThreadId, isStreaming, selectedModel, threads],
  );

  const sendMessage = () => {
    const text = input.trim();
    if (!text || isStreaming) {
      return;
    }

    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (!activeThreadId) {
      setError("No active chat thread. Create a new chat and try again.");
      return;
    }

    setError(null);
    setStats(null);

    const timestamp = Date.now();
    const targetThreadId = activeThreadId;
    const userTempId = `tmp-user-${timestamp}`;
    const assistantTempId = `tmp-assistant-${timestamp}`;

    const optimisticUserMessage: ChatThreadMessageRecord = {
      id: userTempId,
      thread_id: targetThreadId,
      role: "user",
      content: text,
      created_at: timestamp,
      tokens_generated: null,
      generation_time_ms: null,
    };

    const optimisticAssistantMessage: ChatThreadMessageRecord = {
      id: assistantTempId,
      thread_id: targetThreadId,
      role: "assistant",
      content: "",
      created_at: timestamp + 1,
      tokens_generated: null,
      generation_time_ms: null,
    };

    const systemPrompt = supportsThinking
      ? THINKING_SYSTEM_PROMPT.content
      : DEFAULT_SYSTEM_PROMPT.content;

    setMessages((previous) => [
      ...previous,
      optimisticUserMessage,
      optimisticAssistantMessage,
    ]);
    setInput("");
    setIsStreaming(true);
    setStreamingThreadId(targetThreadId);

    streamAbortRef.current = api.sendChatThreadMessageStream(
      targetThreadId,
      {
        model_id: selectedModel,
        content: text,
        system_prompt: systemPrompt,
      },
      {
        onStart: ({ userMessage }) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === userTempId ? userMessage : message,
            ),
          );
        },
        onDelta: (delta) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantTempId
                ? { ...message, content: `${message.content}${delta}` }
                : message,
            ),
          );
        },
        onDone: ({ assistantMessage, stats: streamStats }) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantTempId ? assistantMessage : message,
            ),
          );
          setStats(streamStats);
        },
        onError: (message) => {
          setError(message);
          setMessages((previous) =>
            previous.filter(
              (entry) =>
                !(entry.id === assistantTempId && entry.content.trim().length === 0),
            ),
          );
        },
        onClose: () => {
          setIsStreaming(false);
          setStreamingThreadId(null);
          streamAbortRef.current = null;
          void refreshThreadList(targetThreadId);
          void api
            .getChatThread(targetThreadId)
            .then((detail) => {
              if (activeThreadIdRef.current !== targetThreadId) {
                return;
              }
              setMessages(detail.messages);
              setStats(extractLatestStats(detail.messages));
            })
            .catch(() => {
              // Ignore follow-up sync failures after stream close.
            });
        },
      },
    );
  };

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager();
  };

  const renderModelSelector = () => (
    <div
      className="relative z-40 inline-block w-[300px] max-w-[80vw]"
      ref={modelMenuRef}
    >
      <button
        onClick={() => setIsModelMenuOpen((previous) => !previous)}
        className={clsx(
          "chat-model-selector-btn h-9 w-full px-3 rounded-xl border inline-flex items-center justify-between gap-2 text-xs transition-colors",
          selectedOption?.isReady
            ? "chat-model-selector-btn-ready"
            : "chat-model-selector-btn-idle",
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
            className="chat-model-menu absolute left-0 right-0 bottom-11 rounded-2xl border p-2 shadow-2xl z-[90]"
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
                    "chat-model-option w-full text-left rounded-xl px-2.5 py-2 transition-colors border",
                    selectedOption?.value === option.value
                      ? "chat-model-option-active"
                      : "chat-model-option-idle",
                  )}
                >
                  <div className="chat-model-option-label text-xs truncate">
                    {option.label}
                  </div>
                  <span
                    className={clsx(
                      "chat-model-status mt-1 inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px]",
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
        "chat-composer-shell relative rounded-[32px] border shadow-[0_20px_60px_rgba(0,0,0,0.45)]",
        centered ? "chat-composer-shell-centered" : "chat-composer-shell-docked",
      )}
    >
      <div className="chat-composer-body rounded-[32px] overflow-visible">
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
            "chat-composer-input w-full bg-transparent px-5 pt-5 pb-3 text-sm resize-none focus:outline-none",
            centered ? "min-h-[132px]" : "min-h-[96px]",
          )}
          disabled={isStreaming}
        />

        <div className="flex items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={handleOpenModels}
              className="chat-models-button inline-flex items-center gap-1.5 h-9 px-3 rounded-xl text-xs border transition-colors"
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
              className="chat-send-button h-9 px-3 rounded-xl text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
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
    <div className="relative flex flex-col lg:flex-row gap-4 h-[calc(100dvh-9rem)] lg:h-[calc(100dvh-6.5rem)]">
      <aside className="chat-thread-panel w-full lg:w-72 lg:min-w-72 rounded-2xl border flex flex-col overflow-hidden">
        <div className="chat-thread-panel-header px-3 py-3 border-b flex items-center justify-between gap-3">
          <div>
            <h2 className="chat-thread-panel-title text-sm font-semibold">Chats</h2>
            <p className="chat-thread-panel-subtitle text-xs">
              {threads.length} {threads.length === 1 ? "thread" : "threads"}
            </p>
          </div>
          <button
            onClick={handleCreateThread}
            disabled={isStreaming}
            className="chat-thread-create-btn inline-flex items-center gap-1.5 h-8 px-2.5 rounded-lg text-xs border transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="w-3.5 h-3.5" />
            New
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1.5 max-h-56 lg:max-h-none">
          {threadsLoading ? (
            <div className="chat-thread-empty p-3 rounded-lg text-xs">Loading chats...</div>
          ) : threads.length === 0 ? (
            <div className="chat-thread-empty p-3 rounded-lg text-xs">
              No chats yet. Create one to begin.
            </div>
          ) : (
            threads.map((thread) => {
              const isActive = thread.id === activeThreadId;
              const preview = thread.last_message_preview?.trim() || "No messages yet";

              return (
                <div
                  key={thread.id}
                  className={clsx(
                    "chat-thread-row rounded-xl border",
                    isActive ? "chat-thread-row-active" : "chat-thread-row-idle",
                  )}
                >
                  <button
                    onClick={() => {
                      if (isStreaming) {
                        return;
                      }
                      setActiveThreadId(thread.id);
                      setError(null);
                    }}
                    disabled={isStreaming}
                    className="chat-thread-main w-full text-left px-2.5 py-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="chat-thread-title truncate text-xs font-medium">
                        {thread.title}
                      </p>
                      <span className="chat-thread-time shrink-0 text-[10px]">
                        {formatThreadTimestamp(thread.updated_at)}
                      </span>
                    </div>
                    <p className="chat-thread-preview mt-1 text-[11px] line-clamp-2">
                      {preview}
                    </p>
                  </button>

                  <div className="chat-thread-actions px-2 pb-2 flex justify-end">
                    <button
                      onClick={() => {
                        void handleDeleteThread(thread.id);
                      }}
                      disabled={isStreaming}
                      className="chat-thread-delete-btn h-7 w-7 rounded-md inline-flex items-center justify-center transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Delete chat"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </aside>

      <div className="relative flex-1 min-h-0 flex flex-col">
        {!hasConversation ? (
          <div className="relative flex-1 flex items-center justify-center px-1 sm:px-4">
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
                    <span className="chat-meta-ready">
                      {modelLabel || selectedModel} is loaded and ready.
                    </span>
                  ) : (
                    <span className="chat-meta-muted">
                      {modelLabel || selectedModel} is selected but not loaded.
                    </span>
                  )
                ) : (
                  <span className="chat-meta-muted">No model selected.</span>
                )}
              </div>

              <div className="mt-2 text-center text-xs min-h-[18px]">
                {activeThread ? (
                  <span className="chat-meta-muted">Active chat: {activeThread.title}</span>
                ) : (
                  <span className="chat-meta-muted">No active chat thread.</span>
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
          <div className="relative flex-1 min-h-0 flex flex-col overflow-hidden">
            <div className="chat-conversation-header px-4 sm:px-6 pb-3 border-b flex items-center justify-between gap-3">
              <div>
                <h2 className="chat-conversation-title text-sm font-medium">
                  {activeThread?.title || "Conversation"}
                </h2>
                <p className="chat-conversation-subtitle text-xs mt-1">
                  {selectedModelReady
                    ? `Using ${modelLabel || selectedModel}`
                    : "Model not loaded"}
                </p>
              </div>
              <div className="chat-conversation-subtitle text-xs inline-flex items-center gap-1.5">
                <MessageSquare className="w-3.5 h-3.5" />
                {visibleMessages.length} messages
              </div>
            </div>

            <div className="relative flex-1 min-h-0">
              <div className="h-full overflow-y-auto px-4 sm:px-6 pb-64 pt-4">
                {messagesLoading ? (
                  <div className="chat-thread-empty max-w-4xl mx-auto p-3 rounded-lg text-xs inline-flex items-center gap-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Loading conversation...
                  </div>
                ) : (
                  <div className="max-w-4xl mx-auto space-y-3">
                    {visibleMessages.map((message, index) => {
                      const isUser = message.role === "user";
                      const isLastAssistant =
                        !isUser &&
                        index === visibleMessages.length - 1 &&
                        isStreaming &&
                        streamingThreadId === activeThreadId;
                      const parsed = isUser
                        ? null
                        : supportsThinking
                          ? parseAssistantContent(message.content || "")
                          : null;
                      const messageKey = message.id;
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
                            <div className="chat-assistant-avatar w-7 h-7 rounded-lg border flex items-center justify-center flex-shrink-0">
                              <Bot className="w-4 h-4" />
                            </div>
                          )}

                          <div
                            className={clsx(
                              "chat-message-bubble max-w-[85%] rounded-lg px-3 py-2.5 text-sm break-words",
                              isUser
                                ? "chat-message-bubble-user"
                                : "chat-message-bubble-assistant",
                            )}
                          >
                            {isUser ? (
                              <MarkdownContent
                                content={message.content}
                                className="chat-markdown-user"
                              />
                            ) : (
                              <>
                                {showStreamingThinking && parsed && (
                                  <div className="chat-thinking-panel mb-2 rounded border px-2.5 py-2 text-xs">
                                    <div className="chat-thinking-title mb-1.5 flex items-center gap-1.5 uppercase tracking-wide text-[10px]">
                                      <Loader2 className="w-3 h-3 animate-spin" />
                                      Thinking
                                    </div>
                                    <div className="chat-thinking-body whitespace-pre-wrap">
                                      {parsed.thinking}
                                    </div>
                                  </div>
                                )}

                                {parsed && parsed.answer.length > 0 ? (
                                  <MarkdownContent content={parsed.answer} />
                                ) : parsed && parsed.hasThink ? (
                                  <div className="chat-thinking-placeholder italic">
                                    {isLastAssistant
                                      ? "Thinking..."
                                      : "No final answer was generated."}
                                  </div>
                                ) : (
                                  <MarkdownContent content={message.content} />
                                )}

                                {parsed && parsed.hasThink && !showStreamingThinking && (
                                  <div className="mt-2">
                                    <button
                                      onClick={() =>
                                        setExpandedThoughts((previous) => ({
                                          ...previous,
                                          [messageKey]: !previous[messageKey],
                                        }))
                                      }
                                      className="chat-thinking-toggle inline-flex items-center gap-1 text-xs transition-colors"
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
                                    <div className="chat-thinking-panel chat-thinking-body mt-2 rounded border px-2.5 py-2 text-xs whitespace-pre-wrap">
                                      {parsed.thinking}
                                    </div>
                                  )}

                                {isLastAssistant &&
                                  ((parsed && parsed.answer.length > 0) ||
                                    !showAnswerOnly) && (
                                    <span className="inline-flex items-center ml-1">
                                      <Loader2 className="chat-inline-spinner w-3 h-3 animate-spin" />
                                    </span>
                                  )}
                              </>
                            )}
                          </div>

                          {isUser && (
                            <div className="chat-user-avatar w-7 h-7 rounded-lg border flex items-center justify-center flex-shrink-0">
                              <User className="w-4 h-4" />
                            </div>
                          )}
                        </motion.div>
                      );
                    })}
                    <div ref={listEndRef} />
                  </div>
                )}
              </div>

              <div className="absolute z-30 bottom-0 left-0 right-0 px-4 sm:px-6 pb-4 pt-2 bg-gradient-to-t from-black/30 to-transparent pointer-events-none">
                <div className="max-w-4xl mx-auto pointer-events-auto">
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
                    <div className="chat-stats-line mb-2 text-xs text-center">
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
    </div>
  );
}
