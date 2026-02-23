export async function blobToBase64Payload(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onloadend = () => {
      const value = typeof reader.result === "string" ? reader.result : "";
      const payload = value.split(",")[1] ?? "";
      if (!payload) {
        reject(new Error("Failed to encode audio blob as base64"));
        return;
      }
      resolve(payload);
    };

    reader.onerror = () => {
      reject(new Error("Failed to read audio blob"));
    };

    reader.readAsDataURL(blob);
  });
}
