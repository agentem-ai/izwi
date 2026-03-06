import { APP_ICON_URL } from "@/shared/config/runtime";

function setLinkHref(rel: string, href: string) {
  const link =
    document.querySelector<HTMLLinkElement>(`link[rel='${rel}']`) ??
    document.createElement("link");

  link.rel = rel;
  link.href = href;

  if (!link.parentElement) {
    document.head.appendChild(link);
  }
}

export function bootstrapDocumentIcons() {
  setLinkHref("icon", APP_ICON_URL);
  setLinkHref("apple-touch-icon", APP_ICON_URL);
}
