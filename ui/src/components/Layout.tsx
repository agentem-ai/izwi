import { useEffect, useState } from "react";
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

const TOP_NAV_ITEMS: NavItem[] = [
  {
    id: "voice",
    label: "Voice",
    description: "Flagship realtime interaction",
    icon: AudioLines,
    path: "/voice",
  },
  {
    id: "chat",
    label: "Chat",
    description: "Standard AI interaction hub",
    icon: MessageSquare,
    path: "/chat",
  },
  {
    id: "transcription",
    label: "Transcription",
    description: "Input utility for audio workflows",
    icon: FileText,
    path: "/transcription",
  },
];

const CREATION_NAV_ITEMS: NavItem[] = [
  {
    id: "text-to-speech",
    label: "Text to Speech",
    description: "Output speech from text",
    icon: Mic,
    path: "/text-to-speech",
  },
  {
    id: "voice-cloning",
    label: "Voice Cloning",
    description: "Identity personalization from reference audio",
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
];

const BOTTOM_NAV_ITEMS: NavItem[] = [
  {
    id: "models",
    label: "Models",
    description: "Manage your downloaded models",
    icon: Box,
    path: "/models",
  },
];

export function Layout({
  error,
  onErrorDismiss,
  readyModelsCount,
}: LayoutProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return window.localStorage.getItem("izwi.sidebar.collapsed") === "1";
  });

  useEffect(() => {
    window.localStorage.setItem(
      "izwi.sidebar.collapsed",
      isSidebarCollapsed ? "1" : "0",
    );
  }, [isSidebarCollapsed]);

  const loadedText =
    readyModelsCount > 0
      ? `${readyModelsCount} model${readyModelsCount !== 1 ? "s" : ""} loaded`
      : "No models loaded";

  return (
    <div className="min-h-screen flex bg-[#0d0d0d]">
      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-[#0a0a0a] border-b border-[#262626]">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg overflow-hidden border border-[#2a2a2a] bg-[#111111]">
              <img
                src="/app-icon.png"
                srcSet="/app-icon-128.png 1x, /app-icon-256.png 2x"
                alt="Izwi logo"
                className="w-full h-full object-cover"
              />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white">Izwi</h1>
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
          "w-64 border-r border-[#262626] flex flex-col fixed h-full z-50 bg-[#0a0a0a] transition-all duration-300",
          "lg:translate-x-0",
          isSidebarCollapsed ? "lg:w-[76px]" : "lg:w-64",
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Logo - hidden on mobile since it's in the header */}
        <div
          className={clsx(
            "hidden lg:flex border-b border-[#262626]",
            isSidebarCollapsed
              ? "flex-col items-center gap-2 px-2 py-3"
              : "items-center justify-between p-4",
          )}
          >
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg overflow-hidden border border-[#2a2a2a] bg-[#111111] flex-shrink-0">
              <img
                src="/app-icon.png"
                srcSet="/app-icon-128.png 1x, /app-icon-256.png 2x"
                alt="Izwi logo"
                className="w-full h-full object-cover"
              />
            </div>
            <div className={clsx(isSidebarCollapsed && "hidden")}>
              <h1 className="text-base font-semibold text-white">Izwi</h1>
            </div>
          </div>
          <button
            onClick={() => setIsSidebarCollapsed((collapsed) => !collapsed)}
            className={clsx(
              "rounded-md border border-[#2f2f2f] bg-[#141414] hover:bg-[#1c1c1c] transition-colors text-gray-400 hover:text-white",
              isSidebarCollapsed ? "p-1.5" : "p-1.5",
            )}
            title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <Menu className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 overflow-y-auto flex flex-col">
          <div className="space-y-1">
            {TOP_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center rounded-lg transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "bg-[#1a1a1a] border border-[#333333]"
                      : "hover:bg-[#161616] border border-transparent",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "p-2 rounded-lg transition-all",
                        isActive
                          ? "bg-white text-black"
                          : "bg-[#1f1f1f] text-gray-400 group-hover:text-gray-300 group-hover:bg-[#262626]",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div
                      className={clsx(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
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

          <div className="mt-3 pt-3 border-t border-[#262626]/80 space-y-1">
            {CREATION_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center rounded-lg transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "bg-[#1a1a1a] border border-[#333333]"
                      : "hover:bg-[#161616] border border-transparent",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "p-2 rounded-lg transition-all",
                        isActive
                          ? "bg-white text-black"
                          : "bg-[#1f1f1f] text-gray-400 group-hover:text-gray-300 group-hover:bg-[#262626]",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div
                      className={clsx(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
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
          <div className="mt-auto pt-3 space-y-1 border-t border-[#262626]">
            {BOTTOM_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center rounded-lg transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "bg-[#1a1a1a] border border-[#333333]"
                      : "hover:bg-[#161616] border border-transparent",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "p-2 rounded-lg transition-all",
                        isActive
                          ? "bg-white text-black"
                          : "bg-[#1f1f1f] text-gray-400 group-hover:text-gray-300 group-hover:bg-[#262626]",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div
                      className={clsx(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
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
        <div
          className={clsx(
            "border-t border-[#262626]",
            isSidebarCollapsed ? "p-2.5" : "p-4",
          )}
        >
          <div
            className={clsx(
              "flex items-center",
              isSidebarCollapsed ? "flex-col gap-2" : "justify-between",
            )}
          >
            <div
              className={clsx(
                "text-xs text-gray-500",
                isSidebarCollapsed && "text-center",
              )}
              title={loadedText}
            >
              {isSidebarCollapsed ? (
                <span
                  className={clsx(
                    "inline-flex w-2 h-2 rounded-full",
                    readyModelsCount > 0 ? "bg-white" : "bg-gray-600",
                  )}
                />
              ) : readyModelsCount > 0 ? (
                <span className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-white" />
                  {loadedText}
                </span>
              ) : (
                <span className="text-gray-600">{loadedText}</span>
              )}
            </div>
            <a
              href="https://github.com/agentem-ai/izwi"
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 rounded hover:bg-[#1a1a1a] transition-colors"
              title="Izwi on GitHub"
            >
              <Github className="w-4 h-4 text-gray-500 hover:text-white" />
            </a>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div
        className={clsx(
          "flex-1 pt-16 lg:pt-0 transition-all duration-300",
          isSidebarCollapsed ? "lg:ml-[76px]" : "lg:ml-64",
        )}
      >
        {/* Error toast */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="fixed top-4 left-1/2 -translate-x-1/2 z-50"
            >
              <div className="flex items-center gap-3 px-4 py-2.5 rounded bg-[#1a1a1a] border border-[#333333]">
                <AlertCircle className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-300">{error}</span>
                <button
                  onClick={onErrorDismiss}
                  className="p-1 rounded hover:bg-[#262626] transition-colors"
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
