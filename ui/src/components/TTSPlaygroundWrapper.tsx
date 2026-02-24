import { CustomVoicePlayground } from "./CustomVoicePlayground";

interface TTSPlaygroundWrapperProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

export function TTSPlaygroundWrapper({
  selectedModel,
  onModelRequired,
}: TTSPlaygroundWrapperProps) {
  return (
    <div className="space-y-4">
      {/* Engine description */}
      <p className="text-xs text-gray-500">
        Built-in voice TTS with Qwen3-TTS, Kokoro-82M, and LFM2
      </p>

      {/* Qwen3-TTS Playground */}
      <CustomVoicePlayground
        selectedModel={selectedModel}
        onModelRequired={onModelRequired}
      />
    </div>
  );
}
