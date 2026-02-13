export type ViewMode =
  | "custom-voice"
  | "voice-clone"
  | "voice-design"
  | "transcription"
  | "chat";

export interface ViewConfig {
  id: ViewMode;
  label: string;
  description: string;
  icon: string;
  modelFilter: (variant: string) => boolean;
  emptyStateTitle: string;
  emptyStateDescription: string;
  disabled?: boolean;
}

export const VIEW_CONFIGS: Record<ViewMode, ViewConfig> = {
  "custom-voice": {
    id: "custom-voice",
    label: "Text to Speech",
    description: "Generate speech with built-in voice profiles",
    icon: "Volume2",
    modelFilter: (variant) => variant.includes("CustomVoice"),
    emptyStateTitle: "No TTS Model Loaded",
    emptyStateDescription: "Load a CustomVoice model to generate speech",
  },
  "voice-clone": {
    id: "voice-clone",
    label: "Voice Cloning",
    description: "Clone any voice with a reference audio sample",
    icon: "Users",
    modelFilter: (variant) =>
      variant.includes("Base") && !variant.includes("Tokenizer"),
    emptyStateTitle: "No Base Model Loaded",
    emptyStateDescription:
      "Load a Base model to clone voices from reference audio",
  },
  "voice-design": {
    id: "voice-design",
    label: "Voice Design",
    description: "Create unique voices from text descriptions",
    icon: "Wand2",
    modelFilter: (variant) => variant.includes("VoiceDesign"),
    emptyStateTitle: "No Voice Design Model Loaded",
    emptyStateDescription:
      "Load the VoiceDesign model to create voices from descriptions",
  },
  transcription: {
    id: "transcription",
    label: "Transcription",
    description:
      "Speech-to-text with Qwen3-ASR, Parakeet-TDT, and Voxtral models",
    icon: "FileText",
    modelFilter: (variant) =>
      variant.includes("Qwen3-ASR") ||
      variant.includes("Parakeet-TDT") ||
      variant.includes("Voxtral"),
    emptyStateTitle: "No ASR Model Loaded",
    emptyStateDescription:
      "Download and load a Qwen3-ASR, Parakeet-TDT, or Voxtral model for speech transcription",
  },
  chat: {
    id: "chat",
    label: "Chat",
    description: "Text-to-text chat with Qwen3 and Gemma 3",
    icon: "MessageSquare",
    modelFilter: (variant) =>
      variant === "Qwen3-0.6B-4bit" ||
      variant === "Gemma-3-1b-it" ||
      variant === "Gemma-3-4b-it",
    emptyStateTitle: "No Chat Model Loaded",
    emptyStateDescription:
      "Load Qwen3-0.6B-4bit or Gemma-3-1b-it to start chatting",
  },
};

export const SPEAKERS = [
  {
    id: "Vivian",
    name: "Vivian",
    language: "Chinese",
    description: "Warm and expressive female voice",
  },
  {
    id: "Serena",
    name: "Serena",
    language: "English",
    description: "Clear and professional female voice",
  },
  {
    id: "Ryan",
    name: "Ryan",
    language: "English",
    description: "Confident and friendly male voice",
  },
  {
    id: "Aiden",
    name: "Aiden",
    language: "English",
    description: "Young and energetic male voice",
  },
  {
    id: "Dylan",
    name: "Dylan",
    language: "English",
    description: "Deep and authoritative male voice",
  },
  {
    id: "Eric",
    name: "Eric",
    language: "English",
    description: "Calm and measured male voice",
  },
  {
    id: "Sohee",
    name: "Sohee",
    language: "Korean",
    description: "Gentle and melodic female voice",
  },
  {
    id: "Ono_anna",
    name: "Anna",
    language: "Japanese",
    description: "Soft and pleasant female voice",
  },
  {
    id: "Uncle_fu",
    name: "Uncle Fu",
    language: "Chinese",
    description: "Mature and wise male voice",
  },
];

export const LANGUAGES = [
  { id: "Auto", name: "Auto Detect" },
  { id: "Chinese", name: "Chinese" },
  { id: "English", name: "English" },
  { id: "Japanese", name: "Japanese" },
  { id: "Korean", name: "Korean" },
  { id: "German", name: "German" },
  { id: "French", name: "French" },
  { id: "Russian", name: "Russian" },
  { id: "Portuguese", name: "Portuguese" },
  { id: "Spanish", name: "Spanish" },
  { id: "Italian", name: "Italian" },
];

export const SAMPLE_TEXTS = {
  english: [
    "Hello! Welcome to Izwi, a text-to-speech engine powered by Qwen3-TTS.",
    "The quick brown fox jumps over the lazy dog.",
    "In a world where technology evolves rapidly, the ability to generate natural-sounding speech has become increasingly important.",
  ],
  chinese: [
    "你好！欢迎使用Izwi，一个由Qwen3-TTS驱动的文本转语音引擎。",
    "今天天气真好，我们一起去公园散步吧。",
    "人工智能正在改变我们的生活方式。",
  ],
};

export const VOICE_DESIGN_PRESETS = [
  {
    name: "Professional Female",
    description:
      "A clear, confident, professional female voice with neutral accent. Suitable for business presentations and narration.",
  },
  {
    name: "Warm Storyteller",
    description:
      "A warm, gentle male voice with a storytelling quality. Perfect for audiobooks and bedtime stories.",
  },
  {
    name: "Energetic Youth",
    description:
      "A young, energetic voice full of enthusiasm. Great for advertisements and exciting content.",
  },
  {
    name: "Wise Elder",
    description:
      "A mature, thoughtful voice conveying wisdom and experience. Ideal for documentaries and educational content.",
  },
  {
    name: "Playful Character",
    description:
      "A playful, animated voice with expressive range. Perfect for character voices and entertainment.",
  },
  {
    name: "Calm Meditation",
    description:
      "A soft, soothing voice that promotes relaxation. Ideal for meditation guides and ASMR content.",
  },
];
