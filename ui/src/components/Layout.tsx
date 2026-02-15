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
  Sun,
  Moon,
} from "lucide-react";
import clsx from "clsx";

const appIconUrl = `/app-icon.png?v=${Date.now()}`;

interface LayoutProps {
  error: string | null;
  onErrorDismiss: () => void;
  readyModelsCount: number;
  resolvedTheme: "light" | "dark";
  themePreference: "system" | "light" | "dark";
  onThemePreferenceChange: (preference: "system" | "light" | "dark") => void;
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
  {
    id: "diarization",
    label: "Diarization",
    description: "Speaker segmentation with timestamps",
    icon: Users,
    path: "/diarization",
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
  resolvedTheme,
  themePreference,
  onThemePreferenceChange,
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

  const switchTheme = () => {
    onThemePreferenceChange(resolvedTheme === "dark" ? "light" : "dark");
  };

  return (
    <div className="min-h-screen flex bg-[#0d0d0d]">
      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-[#0a0a0a] border-b border-[#262626]">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="relative w-8 h-8 rounded-lg overflow-hidden border border-white/20 bg-[#161616] shadow-[inset_0_1px_0_rgba(255,255,255,0.14),0_0_0_1px_rgba(255,255,255,0.05)]">
              <img
                src={appIconUrl}
                alt="Izwi logo"
                className="w-full h-full object-cover p-0.5 brightness-125 contrast-125"
              />
              <span className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.18),transparent_56%)]" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white">Izwi</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={switchTheme}
              className="p-2 rounded-lg border border-[#2a2a2a] bg-[#141414] hover:bg-[#1a1a1a] transition-colors"
              title={
                resolvedTheme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
            >
              {resolvedTheme === "dark" ? (
                <Sun className="w-4 h-4 text-gray-200" />
              ) : (
                <Moon className="w-4 h-4 text-gray-700" />
              )}
            </button>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg hover:bg-[#1a1a1a] transition-colors"
            >
              <Menu className="w-5 h-5 text-white" />
            </button>
          </div>
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
            <div className="relative w-9 h-9 rounded-lg overflow-hidden border border-white/20 bg-[#161616] shadow-[inset_0_1px_0_rgba(255,255,255,0.14),0_0_0_1px_rgba(255,255,255,0.05)] flex-shrink-0">
              <img
                src={appIconUrl}
                alt="Izwi logo"
                className="w-full h-full object-cover p-0.5 brightness-125 contrast-125"
              />
              <span className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.18),transparent_56%)]" />
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
                    "sidebar-link flex items-center rounded-lg transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "sidebar-link-active"
                      : "sidebar-link-idle",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "sidebar-link-icon p-2 rounded-lg transition-all",
                        isActive
                          ? "sidebar-link-icon-active"
                          : "sidebar-link-icon-idle",
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
                          "sidebar-link-title text-sm font-medium transition-colors truncate",
                          isActive
                            ? "sidebar-link-title-active"
                            : "sidebar-link-title-idle",
                        )}
                      >
                        {item.label}
                      </div>
                      <div className="sidebar-link-description text-xs truncate">
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
                    "sidebar-link flex items-center rounded-lg transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "sidebar-link-active"
                      : "sidebar-link-idle",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "sidebar-link-icon p-2 rounded-lg transition-all",
                        isActive
                          ? "sidebar-link-icon-active"
                          : "sidebar-link-icon-idle",
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
                          "sidebar-link-title text-sm font-medium transition-colors truncate",
                          isActive
                            ? "sidebar-link-title-active"
                            : "sidebar-link-title-idle",
                        )}
                      >
                        {item.label}
                      </div>
                      <div className="sidebar-link-description text-xs truncate">
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
                    "sidebar-link flex items-center rounded-lg transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "sidebar-link-active"
                      : "sidebar-link-idle",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={clsx(
                        "sidebar-link-icon p-2 rounded-lg transition-all",
                        isActive
                          ? "sidebar-link-icon-active"
                          : "sidebar-link-icon-idle",
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
                          "sidebar-link-title text-sm font-medium transition-colors truncate",
                          isActive
                            ? "sidebar-link-title-active"
                            : "sidebar-link-title-idle",
                        )}
                      >
                        {item.label}
                      </div>
                      <div className="sidebar-link-description text-xs truncate">
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
        <div className="hidden lg:flex justify-end px-6 lg:px-8 pt-4">
          <div className="flex flex-col items-end">
            <button
              onClick={switchTheme}
              className="flex items-center gap-2 px-3 py-2 rounded-lg border border-[#2a2a2a] bg-[#141414] hover:bg-[#1a1a1a] transition-colors"
              title={
                resolvedTheme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
            >
              {resolvedTheme === "dark" ? (
                <>
                  <Sun className="w-4 h-4 text-gray-200" />
                  <span className="text-xs text-gray-300">Light</span>
                </>
              ) : (
                <>
                  <Moon className="w-4 h-4 text-gray-700" />
                  <span className="text-xs text-gray-700">Dark</span>
                </>
              )}
            </button>
            {themePreference === "system" && (
              <div className="mt-1 text-[10px] text-right text-gray-500">
                System
              </div>
            )}
          </div>
        </div>

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
        <main className="p-6 lg:px-8 lg:pb-8 lg:pt-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
