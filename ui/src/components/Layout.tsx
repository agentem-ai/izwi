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
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const appIconUrl = `/app-icon.png?v=${Date.now()}`;
const APP_VERSION = `v${__APP_VERSION__}`;

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

  const handleNavClick = (path: string) => {
    setMobileMenuOpen(false);
    if (
      path === "/chat" &&
      typeof window !== "undefined" &&
      window.innerWidth >= 1024
    ) {
      setIsSidebarCollapsed(true);
    }
  };

  return (
    <div className="min-h-screen flex bg-background text-foreground selection:bg-primary selection:text-primary-foreground">
      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-background/80 backdrop-blur-xl border-b border-border">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="relative w-8 h-8 rounded-lg overflow-hidden border border-border bg-card shadow-sm">
              <img
                src={appIconUrl}
                alt="Izwi logo"
                className="w-full h-full object-cover p-0.5 brightness-125 contrast-125"
              />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-foreground">Izwi</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={switchTheme}
              title={
                resolvedTheme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
            >
              {resolvedTheme === "dark" ? (
                <Sun className="w-4 h-4 text-foreground" />
              ) : (
                <Moon className="w-4 h-4 text-foreground" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <Menu className="w-5 h-5 text-foreground" />
            </Button>
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
            className="lg:hidden fixed inset-0 bg-background/80 backdrop-blur-sm z-40"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside
        className={cn(
          "w-64 border-r border-border flex flex-col fixed h-full z-50 bg-card transition-all duration-300 shadow-sm",
          "lg:translate-x-0",
          isSidebarCollapsed ? "lg:w-[76px]" : "lg:w-64",
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Logo - hidden on mobile since it's in the header */}
        <div
          className={cn(
            "hidden lg:flex border-b border-border",
            isSidebarCollapsed
              ? "flex-col items-center gap-2 px-2 py-3"
              : "items-center justify-between p-4",
          )}
        >
          <div className="flex items-center gap-3">
            <div className="relative w-9 h-9 rounded-lg overflow-hidden border border-border shadow-sm flex-shrink-0 bg-background">
              <img
                src={appIconUrl}
                alt="Izwi logo"
                className="w-full h-full object-cover p-0.5 brightness-125 contrast-125"
              />
            </div>
            <div className={cn(isSidebarCollapsed && "hidden")}>
              <h1 className="text-base font-semibold text-foreground tracking-tight">
                Izwi
              </h1>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsSidebarCollapsed((collapsed) => !collapsed)}
            className={cn(
              "text-muted-foreground hover:text-foreground",
              isSidebarCollapsed ? "h-8 w-8" : "h-8 w-8",
            )}
            title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <Menu className="w-4 h-4" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 overflow-y-auto flex flex-col scrollbar-thin">
          <div className="space-y-1">
            {TOP_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => handleNavClick(item.path)}
                className={({ isActive }) =>
                  cn(
                    "flex items-center rounded-md transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "bg-secondary text-secondary-foreground font-medium shadow-sm"
                      : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={cn(
                        "p-1 rounded-md transition-all flex items-center justify-center",
                        isActive
                          ? "text-secondary-foreground"
                          : "text-muted-foreground group-hover:text-foreground",
                      )}
                    >
                      <item.icon className="w-[18px] h-[18px]" />
                    </div>
                    <div
                      className={cn(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
                      <div className="text-sm truncate leading-none mb-1">
                        {item.label}
                      </div>
                      <div className="text-[11px] text-muted-foreground truncate leading-none">
                        {item.description}
                      </div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-border space-y-1">
            <h4
              className={cn(
                "px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2",
                isSidebarCollapsed && "sr-only",
              )}
            >
              Creation
            </h4>
            {CREATION_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => handleNavClick(item.path)}
                className={({ isActive }) =>
                  cn(
                    "flex items-center rounded-md transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "bg-secondary text-secondary-foreground font-medium shadow-sm"
                      : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={cn(
                        "p-1 rounded-md transition-all flex items-center justify-center",
                        isActive
                          ? "text-secondary-foreground"
                          : "text-muted-foreground group-hover:text-foreground",
                      )}
                    >
                      <item.icon className="w-[18px] h-[18px]" />
                    </div>
                    <div
                      className={cn(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
                      <div className="text-sm truncate leading-none mb-1">
                        {item.label}
                      </div>
                      <div className="text-[11px] text-muted-foreground truncate leading-none">
                        {item.description}
                      </div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </div>

          {/* Bottom navigation section */}
          <div className="mt-auto pt-4 space-y-1">
            {BOTTOM_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => handleNavClick(item.path)}
                className={({ isActive }) =>
                  cn(
                    "flex items-center rounded-md transition-all group",
                    isSidebarCollapsed
                      ? "justify-center px-2 py-2.5"
                      : "gap-3 px-3 py-2.5",
                    isActive
                      ? "bg-secondary text-secondary-foreground font-medium shadow-sm"
                      : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={cn(
                        "p-1 rounded-md transition-all flex items-center justify-center",
                        isActive
                          ? "text-secondary-foreground"
                          : "text-muted-foreground group-hover:text-foreground",
                      )}
                    >
                      <item.icon className="w-[18px] h-[18px]" />
                    </div>
                    <div
                      className={cn(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
                      <div className="text-sm truncate leading-none mb-1">
                        {item.label}
                      </div>
                      <div className="text-[11px] text-muted-foreground truncate leading-none">
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
          className={cn(
            "border-t border-border bg-card",
            isSidebarCollapsed ? "p-3" : "p-4",
          )}
        >
          <div
            className={cn(
              "flex items-center",
              isSidebarCollapsed
                ? "flex-col items-center gap-3"
                : "justify-between",
            )}
          >
            <div
              className={cn(
                "flex flex-col",
                isSidebarCollapsed ? "items-center gap-1.5" : "min-w-0 gap-1",
              )}
            >
              <div
                className={cn(
                  "text-xs font-medium",
                  readyModelsCount > 0
                    ? "text-foreground"
                    : "text-muted-foreground",
                  isSidebarCollapsed && "text-center",
                )}
                title={loadedText}
              >
                {isSidebarCollapsed ? (
                  <span
                    className={cn(
                      "inline-flex w-2.5 h-2.5 rounded-full shadow-sm",
                      readyModelsCount > 0
                        ? "bg-green-500"
                        : "bg-muted-foreground/30",
                    )}
                  />
                ) : readyModelsCount > 0 ? (
                  <span className="flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                    </span>
                    {loadedText}
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-muted-foreground/30" />
                    {loadedText}
                  </span>
                )}
              </div>
            </div>
            <div
              className={cn(
                "flex items-center",
                isSidebarCollapsed ? "flex-col gap-2" : "gap-3",
              )}
            >
              <div
                className={cn(
                  "text-[10px] font-medium text-muted-foreground tracking-wider",
                  isSidebarCollapsed && "text-center",
                )}
                title={`App version ${APP_VERSION}`}
              >
                {APP_VERSION}
              </div>
              <a
                href="https://github.com/agentem-ai/izwi"
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-foreground transition-colors"
                title="Izwi on GitHub"
              >
                <Github className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div
        className={cn(
          "flex-1 pt-16 lg:pt-0 transition-all duration-300 min-w-0 bg-background",
          isSidebarCollapsed ? "lg:ml-[76px]" : "lg:ml-64",
        )}
      >
        <div className="hidden lg:flex justify-end px-6 lg:px-8 pt-4">
          <div className="flex flex-col items-end">
            <Button
              variant="outline"
              size="sm"
              onClick={switchTheme}
              className="gap-2 rounded-full px-4"
              title={
                resolvedTheme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
            >
              {resolvedTheme === "dark" ? (
                <>
                  <Sun className="w-4 h-4" />
                  <span className="text-xs font-medium">Light</span>
                </>
              ) : (
                <>
                  <Moon className="w-4 h-4" />
                  <span className="text-xs font-medium">Dark</span>
                </>
              )}
            </Button>
            {themePreference === "system" && (
              <div className="mt-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider pr-2">
                System
              </div>
            )}
          </div>
        </div>

        {/* Error toast */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              className="fixed top-4 left-1/2 -translate-x-1/2 z-50"
            >
              <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-destructive text-destructive-foreground shadow-lg font-medium text-sm">
                <AlertCircle className="w-4 h-4" />
                <span>{error}</span>
                <button
                  onClick={onErrorDismiss}
                  className="p-1 rounded-md hover:bg-white/20 transition-colors ml-2"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Page content */}
        <main className="p-6 lg:px-10 lg:pb-10 lg:pt-6 max-w-7xl mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
