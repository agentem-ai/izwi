import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Brain,
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
import { useSearchParams } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
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
    "You are a helpful assistant. Always reason inside <think>...</think> before giving the final answer. Keep thinking concise, always close </think>, then provide a clear final answer outside the tags.",
};

const DEFAULT_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful assistant. Provide only the final answer and do not output <think> tags or internal reasoning.",
};

const DEFAULT_THREAD_TITLE = "New chat";
const MAX_THREAD_TITLE_CHARS = 80;

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

function truncateText(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxChars - 3)).trim()}...`;
}

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function stripThinkingArtifacts(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, " ")
    .replace(/<think>[\s\S]*$/gi, " ")
    .replace(/<think>/gi, " ")
    .replace(/<\/think>/gi, " ");
}

function displayThreadTitle(rawTitle: string | null | undefined): string {
  const cleaned = normalizeWhitespace(stripThinkingArtifacts(rawTitle ?? ""));
  if (!cleaned) {
    return DEFAULT_THREAD_TITLE;
  }
  return truncateText(cleaned, MAX_THREAD_TITLE_CHARS);
}

function threadPreviewFromContent(content: string | null | undefined): string {
  const normalized = normalizeWhitespace(stripThinkingArtifacts(content ?? ""));
  if (!normalized) {
    return "No messages yet";
  }
  return truncateText(normalized, 120);
}

function normalizeGeneratedThreadTitle(raw: string): string | null {
  const compact = normalizeWhitespace(
    stripThinkingArtifacts(raw)
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/<\/?[^>]+>/g, " "),
  );
  if (!compact) {
    return null;
  }

  let title = compact.replace(/^title\s*[:\-]\s*/i, "").trim();
  title = normalizeWhitespace(title.replace(/^['"`]+|['"`]+$/g, "").trim());

  if (!title || /^user\s*:/i.test(title) || /^assistant\s*:/i.test(title)) {
    return null;
  }

  return displayThreadTitle(title);
}

function fallbackThreadTitleFromUserMessage(content: string): string {
  const normalized = normalizeWhitespace(stripThinkingArtifacts(content));
  if (!normalized) {
    return DEFAULT_THREAD_TITLE;
  }
  return truncateText(normalized, MAX_THREAD_TITLE_CHARS);
}

interface GenerateTitleArgs {
  threadId: string;
  userContent: string;
  assistantContent: string;
  modelId: string | null;
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
  const [searchParams, setSearchParams] = useSearchParams();
  const activeThreadId = searchParams.get("threadId");

  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [messages, setMessages] = useState<ChatThreadMessageRecord[]>([]);
  const [expandedThoughts, setExpandedThoughts] = useState<
    Record<string, boolean>
  >({});
  const [input, setInput] = useState("");
  const [isThinkingEnabled, setIsThinkingEnabled] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPreparingThread, setIsPreparingThread] = useState(false);
  const [streamingThreadId, setStreamingThreadId] = useState<string | null>(
    null,
  );
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
  const isStreamingRef = useRef(false);
  const streamingThreadIdRef = useRef<string | null>(null);
  const threadsRef = useRef<ChatThread[]>([]);
  const titleGenerationInFlightRef = useRef<Set<string>>(new Set());
  const streamAbortRef = useRef<AbortController | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return (
      modelOptions.find((option) => option.value === selectedModel) || null
    );
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
    !!activeThreadId &&
    (visibleMessages.length > 0 || isStreaming || messagesLoading);
  const thinkingEnabledForModel = supportsThinking && isThinkingEnabled;

  const setActiveThreadInUrl = useCallback(
    (threadId: string | null, replace = false) => {
      const nextSearchParams = new URLSearchParams(searchParams);
      if (threadId) {
        nextSearchParams.set("threadId", threadId);
      } else {
        nextSearchParams.delete("threadId");
      }
      setSearchParams(nextSearchParams, { replace });
    },
    [searchParams, setSearchParams],
  );

  const refreshThreadList = useCallback(
    async (preferredThreadId?: string | null) => {
      try {
        const listedThreads = await api.listChatThreads();
        setThreads(listedThreads);

        const resolvedThreadId = preferredThreadId ?? activeThreadIdRef.current;
        if (
          resolvedThreadId &&
          !listedThreads.some((thread) => thread.id === resolvedThreadId)
        ) {
          setActiveThreadInUrl(null, true);
        }
      } catch {
        // Keep current thread state on refresh failures.
      }
    },
    [setActiveThreadInUrl],
  );

  const maybeGenerateThreadTitle = useCallback(
    async (args: GenerateTitleArgs) => {
      const { threadId, userContent, assistantContent, modelId } = args;

      const existingThread = threadsRef.current.find(
        (thread) => thread.id === threadId,
      );
      if (
        existingThread &&
        existingThread.title !== DEFAULT_THREAD_TITLE &&
        displayThreadTitle(existingThread.title) !== DEFAULT_THREAD_TITLE
      ) {
        return;
      }

      if (titleGenerationInFlightRef.current.has(threadId)) {
        return;
      }

      titleGenerationInFlightRef.current.add(threadId);

      let nextTitle: string | null = null;

      if (modelId) {
        try {
          const titleResponse = await api.createResponse({
            model_id: modelId,
            instructions:
              "Generate a concise chat title (max 8 words) that summarizes the conversation topic. Return only the title text with no quotes, punctuation suffix, or commentary.",
            input: `User: ${userContent}\nAssistant: ${assistantContent}`,
            max_output_tokens: 24,
            store: false,
          });

          nextTitle = normalizeGeneratedThreadTitle(titleResponse.output_text);
        } catch {
          // Fall back to deterministic title below.
        }
      }

      if (!nextTitle) {
        nextTitle = fallbackThreadTitleFromUserMessage(userContent);
      }

      if (!nextTitle || nextTitle === DEFAULT_THREAD_TITLE) {
        titleGenerationInFlightRef.current.delete(threadId);
        return;
      }

      try {
        const updatedThread = await api.updateChatThread(threadId, {
          title: nextTitle,
        });

        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === updatedThread.id ? updatedThread : thread,
          ),
        );
      } catch {
        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === threadId
              ? { ...thread, title: nextTitle ?? thread.title }
              : thread,
          ),
        );
      } finally {
        titleGenerationInFlightRef.current.delete(threadId);
      }
    },
    [],
  );

  useEffect(() => {
    activeThreadIdRef.current = activeThreadId;
  }, [activeThreadId]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    streamingThreadIdRef.current = streamingThreadId;
  }, [streamingThreadId]);

  useEffect(() => {
    threadsRef.current = threads;
  }, [threads]);

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
    if (initializedRef.current) {
      return;
    }
    initializedRef.current = true;

    const initializeThreads = async () => {
      setThreadsLoading(true);
      try {
        const listedThreads = await api.listChatThreads();
        setThreads(listedThreads);

        if (
          activeThreadIdRef.current &&
          !listedThreads.some(
            (thread) => thread.id === activeThreadIdRef.current,
          )
        ) {
          setActiveThreadInUrl(null, true);
        }
      } catch (loadError) {
        setError(getErrorMessage(loadError, "Failed to load chat threads."));
      } finally {
        setThreadsLoading(false);
      }
    };

    void initializeThreads();
  }, [setActiveThreadInUrl]);

  useEffect(() => {
    if (threadsLoading || !activeThreadId) {
      return;
    }

    if (!threads.some((thread) => thread.id === activeThreadId)) {
      setActiveThreadInUrl(null, true);
    }
  }, [activeThreadId, setActiveThreadInUrl, threads, threadsLoading]);

  useEffect(() => {
    if (!activeThreadId) {
      setMessages([]);
      setExpandedThoughts({});
      setStats(null);
      setMessagesLoading(false);
      return;
    }

    const requestedThreadId = activeThreadId;
    if (
      isStreamingRef.current &&
      streamingThreadIdRef.current === requestedThreadId
    ) {
      return;
    }

    let cancelled = false;

    const loadThread = async () => {
      setMessagesLoading(true);
      try {
        const detail = await api.getChatThread(requestedThreadId);
        if (cancelled || activeThreadIdRef.current !== requestedThreadId) {
          return;
        }
        if (
          isStreamingRef.current &&
          streamingThreadIdRef.current === requestedThreadId
        ) {
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
          setError(
            getErrorMessage(loadError, "Failed to load this conversation."),
          );
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

    const activeId = activeThreadIdRef.current;
    if (activeId) {
      void api
        .getChatThread(activeId)
        .then((detail) => {
          if (detail.thread.id !== activeThreadIdRef.current) {
            return;
          }
          setMessages(detail.messages);
          setStats(extractLatestStats(detail.messages));
        })
        .catch(() => {
          // Ignore follow-up sync failures after cancel.
        });
      void refreshThreadList(activeId);
    }
  }, [refreshThreadList]);

  const handleCreateThread = useCallback(async () => {
    if (isStreaming) {
      return;
    }

    try {
      const thread = await api.createChatThread({
        model_id: selectedModel ?? undefined,
      });
      setThreads((previous) => [thread, ...previous]);
      setActiveThreadInUrl(thread.id);
      setMessages([]);
      setExpandedThoughts({});
      setStats(null);
      setError(null);
      setInput("");
    } catch (createError) {
      setError(getErrorMessage(createError, "Failed to create a new chat."));
    }
  }, [isStreaming, selectedModel, setActiveThreadInUrl]);

  const handleDeleteThread = useCallback(
    async (threadId: string) => {
      if (isStreaming) {
        return;
      }

      try {
        await api.deleteChatThread(threadId);
        setThreads((previous) =>
          previous.filter((thread) => thread.id !== threadId),
        );

        if (activeThreadIdRef.current === threadId) {
          setActiveThreadInUrl(null, true);
          setMessages([]);
          setExpandedThoughts({});
          setStats(null);
        }
      } catch (deleteError) {
        setError(getErrorMessage(deleteError, "Failed to delete this chat."));
      }
    },
    [isStreaming, setActiveThreadInUrl],
  );

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isStreaming || isPreparingThread) {
      return;
    }

    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    let targetThreadId = activeThreadId;
    if (!targetThreadId) {
      setIsPreparingThread(true);
      try {
        const createdThread = await api.createChatThread({
          model_id: selectedModel ?? undefined,
        });
        setThreads((previous) => [createdThread, ...previous]);
        setActiveThreadInUrl(createdThread.id);
        setMessages([]);
        setExpandedThoughts({});
        setStats(null);
        targetThreadId = createdThread.id;
      } catch (threadError) {
        setError(getErrorMessage(threadError, "Failed to create a new chat."));
        setIsPreparingThread(false);
        return;
      }
      setIsPreparingThread(false);
    }

    if (!targetThreadId) {
      return;
    }

    setError(null);
    setStats(null);

    const isFirstTurn =
      targetThreadId === activeThreadId
        ? messages.filter((message) => message.role === "user").length === 0
        : true;

    const timestamp = Date.now();
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

    const systemPrompt = thinkingEnabledForModel
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
          setMessages((previous) => {
            let replaced = false;
            const updated = previous.map((message) => {
              if (message.id === userTempId) {
                replaced = true;
                return userMessage;
              }
              return message;
            });

            if (
              replaced ||
              updated.some((message) => message.id === userMessage.id)
            ) {
              return updated;
            }

            return [...updated, userMessage];
          });
        },
        onDelta: (delta) => {
          setMessages((previous) => {
            let updatedAssistant = false;
            const updated = previous.map((message) => {
              if (message.id === assistantTempId) {
                updatedAssistant = true;
                return { ...message, content: `${message.content}${delta}` };
              }
              return message;
            });

            if (updatedAssistant) {
              return updated;
            }

            if (
              updated.some(
                (message) => message.id === optimisticAssistantMessage.id,
              )
            ) {
              return updated;
            }

            return [
              ...updated,
              { ...optimisticAssistantMessage, content: delta },
            ];
          });
        },
        onDone: ({ assistantMessage, stats: streamStats, modelId }) => {
          setMessages((previous) => {
            let replaced = false;
            const updated = previous.map((message) => {
              if (message.id === assistantTempId) {
                replaced = true;
                return assistantMessage;
              }
              return message;
            });

            if (
              replaced ||
              updated.some((message) => message.id === assistantMessage.id)
            ) {
              return updated;
            }

            return [...updated, assistantMessage];
          });
          setStats(streamStats);

          if (isFirstTurn) {
            void maybeGenerateThreadTitle({
              threadId: targetThreadId,
              userContent: text,
              assistantContent: assistantMessage.content,
              modelId,
            });
          }
        },
        onError: (message) => {
          setError(message);
          setMessages((previous) =>
            previous.filter(
              (entry) =>
                !(
                  entry.id === assistantTempId &&
                  entry.content.trim().length === 0
                ),
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
      className={cn(
        "relative z-40 inline-block w-[240px] sm:w-[300px] max-w-[calc(100vw-9rem)]",
      )}
      ref={modelMenuRef}
    >
      <Button
        variant="outline"
        onClick={() => setIsModelMenuOpen((previous) => !previous)}
        className={cn(
          "w-full justify-between font-normal h-9",
          selectedOption?.isReady ? "border-primary/20 bg-primary/5" : "",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown className="w-3.5 h-3.5 shrink-0 opacity-50" />
      </Button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className="absolute left-0 right-0 bottom-11 rounded-md border bg-popover text-popover-foreground p-1 shadow-md z-[90]"
          >
            <div className="max-h-64 overflow-y-auto">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={cn(
                    "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 px-2 text-sm outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                    selectedOption?.value === option.value &&
                      "bg-accent text-accent-foreground",
                  )}
                >
                  <div className="flex flex-col items-start min-w-0">
                    <span className="truncate w-full text-left font-medium">
                      {option.label}
                    </span>
                    <span
                      className={cn(
                        "mt-1 text-[10px] uppercase tracking-wider font-semibold",
                        option.isReady
                          ? "text-green-500"
                          : "text-muted-foreground",
                      )}
                    >
                      {option.statusLabel}
                    </span>
                  </div>
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
      className={cn(
        "relative rounded-xl border border-[var(--border-muted)] bg-background shadow-sm overflow-visible",
        centered && "max-w-3xl mx-auto shadow-md",
      )}
    >
      <div className="bg-background">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void sendMessage();
            }
          }}
          placeholder={
            !activeThreadId
              ? "Ask anything..."
              : !selectedModel
                ? "Choose a model and ask anything..."
                : !selectedModelReady
                  ? "Model selected but not loaded. Open Models to load it."
                  : "Ask anything..."
          }
          className={cn(
            "w-full bg-transparent px-4 pt-4 pb-3 text-sm resize-none focus:outline-none placeholder:text-muted-foreground",
            centered ? "min-h-[132px]" : "min-h-[96px]",
          )}
          disabled={isStreaming || isPreparingThread}
        />

        <div className="flex flex-wrap items-center justify-between gap-2 px-3 pb-3 pt-1">
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleOpenModels}
              className="h-8 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
            >
              <Settings2 className="w-3.5 h-3.5" />
              Models
            </Button>
            {supportsThinking && (
              <Button
                variant={thinkingEnabledForModel ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setIsThinkingEnabled((previous) => !previous)}
                disabled={isStreaming || isPreparingThread}
                className={cn(
                  "h-8 gap-1.5 text-xs",
                  !thinkingEnabledForModel &&
                    "text-muted-foreground hover:text-foreground",
                )}
                title={
                  thinkingEnabledForModel
                    ? "Thinking mode is enabled"
                    : "Thinking mode is disabled"
                }
              >
                <Brain className="w-3.5 h-3.5" />
                Thinking {thinkingEnabledForModel ? "On" : "Off"}
              </Button>
            )}
          </div>

          <div className="flex items-center gap-2 w-full sm:w-auto justify-end flex-wrap sm:flex-nowrap">
            {renderModelSelector()}

            <Button
              onClick={isStreaming ? stopStreaming : () => void sendMessage()}
              disabled={isPreparingThread || (!isStreaming && !input.trim())}
              variant={isStreaming ? "destructive" : "default"}
              size="sm"
              className="h-9 gap-1.5 font-medium px-4"
            >
              {isStreaming ? (
                <Square className="w-3.5 h-3.5" />
              ) : isPreparingThread ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              {isStreaming
                ? "Cancel"
                : isPreparingThread
                  ? "Starting..."
                  : "Send"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="relative flex flex-col lg:flex-row gap-4 h-[calc(100dvh-12rem)] lg:h-[calc(100dvh-11.75rem)]">
      <aside className="w-full lg:w-80 lg:min-w-[20rem] max-h-[38dvh] lg:max-h-none shrink-0 rounded-xl border border-[var(--border-muted)] bg-card text-card-foreground flex flex-col overflow-hidden shadow-sm">
        <div className="px-4 py-3 border-b border-[var(--border-muted)] flex items-center justify-between gap-3 bg-muted/30">
          <div>
            <h2 className="text-sm font-semibold tracking-tight">Chats</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              {threads.length} {threads.length === 1 ? "thread" : "threads"}
            </p>
          </div>
          <Button
            onClick={handleCreateThread}
            disabled={isStreaming || isPreparingThread}
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            New
          </Button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-1 scrollbar-thin">
          {threadsLoading ? (
            <div className="p-4 text-center text-xs text-muted-foreground flex items-center justify-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading chats...
            </div>
          ) : threads.length === 0 ? (
            <div className="p-4 text-center text-xs text-muted-foreground border border-[var(--border-muted)] border-dashed rounded-lg m-2 bg-muted/20">
              No chats yet. Create one to begin.
            </div>
          ) : (
            threads.map((thread) => {
              const isActive = thread.id === activeThreadId;
              const preview = threadPreviewFromContent(
                thread.last_message_preview,
              );

              return (
                <div
                  key={thread.id}
                  className={cn(
                    "group relative rounded-md border border-transparent transition-colors hover:bg-muted/50",
                    isActive && "bg-accent/80 hover:bg-accent/80",
                  )}
                >
                  <button
                    onClick={() => {
                      if (isStreaming) {
                        return;
                      }
                      setActiveThreadInUrl(thread.id);
                      setError(null);
                    }}
                    disabled={isStreaming || isPreparingThread}
                    className="block w-full text-left px-3 py-2.5 pr-10 outline-none"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p
                        className={cn(
                          "truncate text-sm font-medium leading-none",
                          isActive ? "text-foreground" : "text-foreground/80",
                        )}
                      >
                        {displayThreadTitle(thread.title)}
                      </p>
                      <span className="shrink-0 text-[10px] text-muted-foreground">
                        {formatThreadTimestamp(thread.updated_at)}
                      </span>
                    </div>
                    <p className="mt-1.5 text-xs text-muted-foreground line-clamp-2 leading-snug">
                      {preview}
                    </p>
                  </button>

                  <Button
                    onClick={(e) => {
                      e.stopPropagation();
                      void handleDeleteThread(thread.id);
                    }}
                    disabled={isStreaming || isPreparingThread}
                    variant="ghost"
                    size="icon"
                    className="absolute right-1 top-1.5 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive hover:text-destructive-foreground"
                    title="Delete chat"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
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

              <div className="mt-4 text-center text-xs text-muted-foreground min-h-[18px]">
                {!activeThreadId ? (
                  <span>
                    No active chat selected. Start typing and send to create a
                    new chat.
                  </span>
                ) : selectedModel ? (
                  selectedModelReady ? (
                    <span className="text-foreground/80 font-medium">
                      {modelLabel || selectedModel} is loaded and ready.
                    </span>
                  ) : (
                    <span>
                      {modelLabel || selectedModel} is selected but not loaded.
                    </span>
                  )
                ) : (
                  <span>No model selected.</span>
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
          <div className="relative flex-1 min-h-0 flex flex-col overflow-hidden bg-card border border-[var(--border-muted)] rounded-xl shadow-sm">
            <div className="px-4 sm:px-6 py-4 border-b border-[var(--border-muted)] flex items-center justify-between gap-3 bg-muted/20">
              <div>
                <h2 className="text-sm font-semibold tracking-tight">
                  {activeThread
                    ? displayThreadTitle(activeThread.title)
                    : "Conversation"}
                </h2>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {selectedModelReady
                    ? `Using ${modelLabel || selectedModel}`
                    : "Model not loaded"}
                </p>
              </div>
              <div className="text-xs text-muted-foreground inline-flex items-center gap-1.5 font-medium bg-muted px-2 py-1 rounded-md">
                <MessageSquare className="w-3.5 h-3.5" />
                {visibleMessages.length} messages
              </div>
            </div>

            <div className="relative flex-1 min-h-0 bg-background/50">
              <div className="h-full overflow-y-auto px-4 sm:px-6 pb-64 pt-6 scrollbar-thin">
                {messagesLoading ? (
                  <div className="max-w-4xl mx-auto p-4 text-center text-xs text-muted-foreground flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading conversation...
                  </div>
                ) : (
                  <div className="max-w-4xl mx-auto space-y-6">
                    {visibleMessages.map((message, index) => {
                      const isUser = message.role === "user";
                      const isLastAssistant =
                        !isUser &&
                        index === visibleMessages.length - 1 &&
                        isStreaming &&
                        streamingThreadId === activeThreadId;
                      const assistantDisplayContent = isUser
                        ? message.content
                        : thinkingEnabledForModel
                          ? message.content
                          : stripThinkingArtifacts(message.content || "");
                      const parsed = isUser
                        ? null
                        : thinkingEnabledForModel
                          ? parseAssistantContent(message.content || "")
                          : null;
                      const messageKey = message.id;
                      const isThoughtExpanded = !!expandedThoughts[messageKey];
                      const showStreamingThinking =
                        !isUser &&
                        !!parsed &&
                        isLastAssistant &&
                        parsed.thinking.length > 0 &&
                        (parsed.hasIncompleteThink ||
                          parsed.answer.length === 0);
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
                          className={cn(
                            "flex gap-4",
                            isUser && "flex-row-reverse",
                          )}
                        >
                          <div
                            className={cn(
                              "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 border shadow-sm",
                              isUser
                                ? "bg-primary text-primary-foreground border-primary/20"
                                : "bg-muted text-muted-foreground border-[var(--border-muted)]",
                            )}
                          >
                            {isUser ? (
                              <User className="w-4 h-4" />
                            ) : (
                              <Bot className="w-4 h-4" />
                            )}
                          </div>

                          <div
                            className={cn(
                              "max-w-[85%] text-sm break-words flex flex-col",
                              isUser ? "items-end" : "items-start",
                            )}
                          >
                            <div
                              className={cn(
                                "rounded-2xl px-4 py-2.5 shadow-sm",
                                isUser
                                  ? "bg-primary text-primary-foreground rounded-tr-sm"
                                  : "bg-card border border-[var(--border-muted)] text-card-foreground rounded-tl-sm",
                              )}
                            >
                              {isUser ? (
                                <MarkdownContent
                                  content={message.content}
                                  className="prose-p:leading-relaxed prose-pre:bg-black/10 dark:prose-pre:bg-white/10 prose-pre:border-none prose-a:text-primary-foreground underline"
                                />
                              ) : (
                                <>
                                  {showStreamingThinking && parsed && (
                                    <div className="mb-3 rounded-lg bg-muted/50 border border-[var(--border-muted)] px-3 py-2 text-xs text-muted-foreground">
                                      <div className="mb-2 flex items-center gap-1.5 uppercase tracking-wider text-[10px] font-semibold">
                                        <Loader2 className="w-3 h-3 animate-spin text-primary" />
                                        Thinking
                                      </div>
                                      <div className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed">
                                        {parsed.thinking}
                                      </div>
                                    </div>
                                  )}

                                  {parsed && parsed.answer.length > 0 ? (
                                    <MarkdownContent content={parsed.answer} />
                                  ) : parsed && parsed.hasThink ? (
                                    <div className="italic text-muted-foreground opacity-70">
                                      {isLastAssistant
                                        ? "Thinking..."
                                        : "No final answer was generated."}
                                    </div>
                                  ) : (
                                    <MarkdownContent
                                      content={assistantDisplayContent}
                                    />
                                  )}

                                  {parsed &&
                                    parsed.hasThink &&
                                    !showStreamingThinking && (
                                      <div className="mt-3 border-t pt-2">
                                        <button
                                          onClick={() =>
                                            setExpandedThoughts((previous) => ({
                                              ...previous,
                                              [messageKey]:
                                                !previous[messageKey],
                                            }))
                                          }
                                          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground font-medium transition-colors"
                                        >
                                          {isThoughtExpanded ? (
                                            <ChevronDown className="w-3 h-3" />
                                          ) : (
                                            <ChevronRight className="w-3 h-3" />
                                          )}
                                          {isThoughtExpanded
                                            ? "Hide thought process"
                                            : "Show thought process"}
                                        </button>
                                      </div>
                                    )}

                                  {parsed &&
                                    parsed.hasThink &&
                                    !showStreamingThinking &&
                                    isThoughtExpanded && (
                                      <div className="mt-2 rounded-lg bg-muted/30 border border-[var(--border-muted)] px-3 py-2 text-xs whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-muted-foreground">
                                        {parsed.thinking}
                                      </div>
                                    )}

                                  {isLastAssistant &&
                                    ((parsed && parsed.answer.length > 0) ||
                                      !showAnswerOnly) && (
                                      <span className="inline-flex items-center ml-2 align-middle">
                                        <span className="w-1.5 h-4 bg-primary animate-pulse inline-block" />
                                      </span>
                                    )}
                                </>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                    <div ref={listEndRef} />
                  </div>
                )}
              </div>

              <div className="absolute z-30 bottom-0 left-0 right-0 px-4 sm:px-6 pb-6 pt-4 bg-gradient-to-t from-background via-background/95 to-transparent pointer-events-none">
                <div className="max-w-4xl mx-auto pointer-events-auto">
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, height: 0, y: 10 }}
                        animate={{ opacity: 1, height: "auto", y: 0 }}
                        exit={{ opacity: 0, height: 0, y: 10 }}
                        className="mb-4 p-3 rounded-lg bg-destructive text-destructive-foreground shadow-lg text-sm font-medium flex items-center gap-2"
                      >
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {stats && !isStreaming && (
                    <div className="mb-3 text-[11px] font-medium text-muted-foreground flex items-center justify-center gap-3">
                      <span className="bg-muted px-2 py-0.5 rounded-md border border-[var(--border-muted)] shadow-sm">
                        {stats.tokens_generated} tokens
                      </span>
                      <span className="bg-muted px-2 py-0.5 rounded-md border border-[var(--border-muted)] shadow-sm">
                        {Math.round(stats.generation_time_ms)} ms
                      </span>
                      <span className="bg-muted px-2 py-0.5 rounded-md border border-[var(--border-muted)] shadow-sm">
                        {Math.round(
                          stats.tokens_generated /
                            (stats.generation_time_ms / 1000),
                        )}{" "}
                        t/s
                      </span>
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
