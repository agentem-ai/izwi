import { motion } from "framer-motion";

interface WaveformProps {
  isPlaying?: boolean;
  isGenerating?: boolean;
  className?: string;
}

export function Waveform({
  isPlaying = false,
  isGenerating = false,
  className = "",
}: WaveformProps) {
  const bars = 24;
  const active = isPlaying || isGenerating;

  return (
    <div
      className={`flex items-center justify-center gap-[3px] h-12 ${className}`}
    >
      {Array.from({ length: bars }).map((_, i) => (
        <motion.div
          key={i}
          className={`w-1 rounded-full ${
            active
              ? "bg-gradient-to-t from-[var(--status-positive-solid)] to-[var(--accent-solid)]"
              : "bg-white/20"
          }`}
          initial={{ height: 8 }}
          animate={{
            height: active ? [8, Math.random() * 32 + 16, 8] : 8,
          }}
          transition={{
            duration: isGenerating ? 0.4 : 0.5,
            repeat: active ? Infinity : 0,
            delay: i * 0.05,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

export function MiniWaveform({ isActive = false }: { isActive?: boolean }) {
  return (
    <div className="flex items-center gap-[2px] h-4">
      {[0.6, 1, 0.7, 0.9, 0.5].map((scale, i) => (
        <motion.div
          key={i}
          className={`w-0.5 rounded-full ${isActive ? "bg-[var(--status-positive-solid)]" : "bg-white/30"}`}
          initial={{ height: 4 }}
          animate={{
            height: isActive ? [4, 12 * scale, 4] : 4,
          }}
          transition={{
            duration: 0.6,
            repeat: isActive ? Infinity : 0,
            delay: i * 0.1,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
