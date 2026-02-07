export function withQwen3Prefix(name: string, variant: string): string {
  const trimmed = name.trim();
  const isQwen3Variant = variant.toLowerCase().startsWith("qwen3");

  if (!isQwen3Variant) {
    return trimmed || variant;
  }

  if (!trimmed) {
    return "Qwen3";
  }

  if (/^qwen3\b/i.test(trimmed)) {
    return trimmed;
  }

  return `Qwen3 ${trimmed}`;
}
