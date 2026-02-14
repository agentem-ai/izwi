import { motion } from "framer-motion";

interface VoiceOrbProps {
  isActive?: boolean;
  isGenerating?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizes = {
  sm: "w-16 h-16",
  md: "w-24 h-24",
  lg: "w-32 h-32",
};

export function VoiceOrb({
  isActive = false,
  isGenerating = false,
  size = "md",
  className = "",
}: VoiceOrbProps) {
  const active = isActive || isGenerating;

  return (
    <div className={`relative ${sizes[size]} ${className}`}>
      {/* Outer glow rings */}
      {active && (
        <>
          <motion.div
            className="absolute inset-0 rounded-full bg-gradient-to-r from-[var(--accent-solid)]/20 to-[var(--status-positive-solid)]/20"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.5, 0, 0.5],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <motion.div
            className="absolute inset-0 rounded-full bg-gradient-to-r from-[var(--accent-solid)]/25 to-[var(--status-positive-solid)]/25"
            animate={{
              scale: [1, 1.3, 1],
              opacity: [0.6, 0, 0.6],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.5,
            }}
          />
        </>
      )}

      {/* Main orb */}
      <motion.div
        className={`absolute inset-0 rounded-full ${
          active
            ? "bg-gradient-to-br from-[var(--status-positive-solid)] via-[var(--accent-solid)] to-[var(--bg-surface-4)]"
            : "bg-gradient-to-br from-gray-700 to-gray-800"
        }`}
        animate={
          active
            ? {
                scale: [1, 1.05, 1],
                boxShadow: [
                  "0 0 20px rgba(255, 255, 255, 0.2)",
                  "0 0 36px rgba(255, 255, 255, 0.35)",
                  "0 0 20px rgba(255, 255, 255, 0.2)",
                ],
              }
            : {}
        }
        transition={{
          duration: 1.5,
          repeat: active ? Infinity : 0,
          ease: "easeInOut",
        }}
      >
        {/* Inner highlight */}
        <div className="absolute inset-2 rounded-full bg-gradient-to-br from-white/20 to-transparent" />

        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          {isGenerating ? (
            <motion.div
              className="w-1/3 h-1/3 rounded-full bg-white/80"
              animate={{
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: 0.5,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          ) : (
            <svg
              className={`w-1/3 h-1/3 ${active ? "text-white" : "text-gray-500"}`}
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z" />
            </svg>
          )}
        </div>
      </motion.div>
    </div>
  );
}
