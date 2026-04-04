import { useEffect } from "react";
import { BrowserRouter, useNavigate } from "react-router-dom";
import { useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { AppRoutes } from "@/app/router/AppRoutes";

interface TrayRouteDetail {
  path?: unknown;
}

interface TrayApiUrlDetail {
  url?: unknown;
}

async function copyTextToClipboard(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function DesktopTrayBridge() {
  const navigate = useNavigate();
  const { checkForUpdates } = useAppUpdates();

  useEffect(() => {
    const onTrayRoute = (event: Event) => {
      const routeEvent = event as CustomEvent<TrayRouteDetail>;
      const path = routeEvent.detail?.path;
      if (typeof path !== "string" || !path.startsWith("/")) {
        return;
      }
      navigate(path);
    };
    const onTrayCheckUpdates = () => {
      void checkForUpdates(true);
    };
    const onTrayCopyApiUrl = (event: Event) => {
      const apiUrlEvent = event as CustomEvent<TrayApiUrlDetail>;
      const url = apiUrlEvent.detail?.url;
      if (typeof url !== "string" || !url.trim()) {
        return;
      }
      void copyTextToClipboard(url.trim());
    };

    window.addEventListener("izwi:tray-route", onTrayRoute as EventListener);
    window.addEventListener("izwi:tray-check-updates", onTrayCheckUpdates);
    window.addEventListener("izwi:tray-copy-api-url", onTrayCopyApiUrl as EventListener);

    return () => {
      window.removeEventListener("izwi:tray-route", onTrayRoute as EventListener);
      window.removeEventListener("izwi:tray-check-updates", onTrayCheckUpdates);
      window.removeEventListener(
        "izwi:tray-copy-api-url",
        onTrayCopyApiUrl as EventListener,
      );
    };
  }, [checkForUpdates, navigate]);

  return null;
}

export function AppRouter() {
  return (
    <BrowserRouter>
      <DesktopTrayBridge />
      <AppRoutes />
    </BrowserRouter>
  );
}
