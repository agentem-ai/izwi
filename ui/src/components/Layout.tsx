import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Users,
  Wand2,
  FileText,
  MessageSquare,
  AudioLines,
  Box,
  Github,
  Waves,
  AlertCircle,
  X,
  Menu,
} from "lucide-react";
import clsx from "clsx";

interface LayoutProps {
  error: string | null;
  onErrorDismiss: () => void;
  readyModelsCount: number;
}

interface NavItem {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  path: string;
}

const NAV_ITEMS: NavItem[] = [
  {
    id: "chat",
    label: "Chat",
    description: "Text-to-text conversation",
    icon: MessageSquare,
    path: "/chat",
  },
  {
    id: "voice",
    label: "Voice",
    description: "Realtime voice assistant",
    icon: AudioLines,
    path: "/voice",
  },
  {
    id: "text-to-speech",
    label: "Text to Speech",
    description: "Generate speech from text",
    icon: Mic,
    path: "/text-to-speech",
  },
  {
    id: "voice-cloning",
    label: "Voice Cloning",
    description: "Clone any voice from audio",
    icon: Users,
    path: "/voice-cloning",
  },
  {
    id: "voice-design",
    label: "Voice Design",
    description: "Create voices from descriptions",
    icon: Wand2,
    path: "/voice-design",
  },
  {
    id: "transcription",
    label: "Transcription",
    description: "Speech-to-text conversion",
    icon: FileText,
    path: "/transcription",
  },
];

const BOTTOM_NAV_ITEMS: NavItem[] = [
  {
    id: "my-models",
    label: "My Models",
    description: "Manage your downloaded models",
    icon: Box,
    path: "/my-models",
  },
];

export function Layout({
  error,
  onErrorDismiss,
  readyModelsCount,
}: LayoutProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex bg-[#0d0d0d]">
      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-[#0d0d0d] border-b border-[#2a2a2a]">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center">
              <Waves className="w-4 h-4 text-black" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white">Izwi Audio</h1>
              <p className="text-xs text-gray-500">Audio Toolkit</p>
            </div>
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg hover:bg-[#1a1a1a] transition-colors"
          >
            <Menu className="w-5 h-5 text-white" />
          </button>
        </div>
      </div>

      {/* Mobile menu overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setMobileMenuOpen(false)}
            className="lg:hidden fixed inset-0 bg-black/50 z-40"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside
        className={clsx(
          "w-64 border-r border-[#2a2a2a] flex flex-col fixed h-full z-50 bg-[#0d0d0d] transition-transform duration-300",
          "lg:translate-x-0",
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Logo - hidden on mobile since it's in the header */}
        <div className="hidden lg:block p-4 border-b border-[#2a2a2a]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-white rounded-lg flex items-center justify-center">
              <Waves className="w-5 h-5 text-black" />
            </div>
            <div>
              <h1 className="text-base font-semibold text-white">Izwi Audio</h1>
              <p className="text-xs text-gray-500">Audio Toolkit</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 overflow-y-auto flex flex-col">
          <div className="space-y-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors group",
                    isActive
                      ? "bg-[#1a1a1a] border border-[#2a2a2a]"
                      : "hover:bg-[#161616] border border-transparent",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "p-2 rounded-lg transition-colors",
                        isActive
                          ? "bg-white text-black"
                          : "bg-[#1f1f1f] text-gray-400 group-hover:text-gray-300",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className={clsx(
                          "text-sm font-medium transition-colors truncate",
                          isActive
                            ? "text-white"
                            : "text-gray-400 group-hover:text-gray-300",
                        )}
                      >
                        {item.label}
                      </div>
                      <div className="text-xs text-gray-600 truncate">
                        {item.description}
                      </div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </div>

          {/* Bottom navigation section */}
          <div className="mt-auto pt-3 space-y-1 border-t border-[#2a2a2a]">
            {BOTTOM_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors group",
                    isActive
                      ? "bg-[#1a1a1a] border border-[#2a2a2a]"
                      : "hover:bg-[#161616] border border-transparent",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "p-2 rounded-lg transition-colors",
                        isActive
                          ? "bg-white text-black"
                          : "bg-[#1f1f1f] text-gray-400 group-hover:text-gray-300",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className={clsx(
                          "text-sm font-medium transition-colors truncate",
                          isActive
                            ? "text-white"
                            : "text-gray-400 group-hover:text-gray-300",
                        )}
                      >
                        {item.label}
                      </div>
                      <div className="text-xs text-gray-600 truncate">
                        {item.description}
                      </div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-[#2a2a2a]">
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-500">
              {readyModelsCount > 0 ? (
                <span className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                  {readyModelsCount} model{readyModelsCount !== 1 ? "s" : ""}{" "}
                  loaded
                </span>
              ) : (
                <span className="text-gray-600">No models loaded</span>
              )}
            </div>
            <a
              href="https://github.com/QwenLM/Qwen3-TTS"
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 rounded hover:bg-[#1a1a1a] transition-colors"
            >
              <Github className="w-4 h-4 text-gray-500 hover:text-white" />
            </a>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 lg:ml-64 pt-16 lg:pt-0">
        {/* Error toast */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="fixed top-4 left-1/2 -translate-x-1/2 z-50"
            >
              <div className="flex items-center gap-3 px-4 py-2.5 rounded bg-[#1a1a1a] border border-red-900/50">
                <AlertCircle className="w-4 h-4 text-red-400" />
                <span className="text-sm text-red-200">{error}</span>
                <button
                  onClick={onErrorDismiss}
                  className="p-1 rounded hover:bg-[#2a2a2a] transition-colors"
                >
                  <X className="w-3.5 h-3.5 text-gray-500" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Page content */}
        <main className="p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
