import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

const appIconUrl = `/app-icon.png?v=${Date.now()}`;
const THEME_STORAGE_KEY = "izwi.theme.preference";

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

setLinkHref("icon", appIconUrl);
setLinkHref("apple-touch-icon", appIconUrl);

const storedThemePreference = window.localStorage.getItem(THEME_STORAGE_KEY);
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const resolvedTheme =
  storedThemePreference === "light" || storedThemePreference === "dark"
    ? storedThemePreference
    : prefersDark
      ? "dark"
      : "light";
document.documentElement.classList.remove("theme-light", "theme-dark");
document.documentElement.classList.add(
  resolvedTheme === "dark" ? "theme-dark" : "theme-light",
);
document.documentElement.style.colorScheme = resolvedTheme;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
