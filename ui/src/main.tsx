import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import appIconUrl from "./assets/app-icon.png";

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

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
