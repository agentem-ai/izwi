import { useEffect, type ReactNode } from "react";

import { api } from "@/api";
import { setAnalyticsEnabled } from "@/app/analytics/client";
import {
  routeIdFromPathname,
  trackAppOpened,
  trackRouteViewed,
} from "@/app/analytics/events";

interface AnalyticsBootstrapProviderProps {
  children: ReactNode;
}

export function AnalyticsBootstrapProvider({
  children,
}: AnalyticsBootstrapProviderProps) {
  useEffect(() => {
    let active = true;

    api
      .getPreferences()
      .then((preferences) => {
        if (!active) {
          return;
        }

        setAnalyticsEnabled(preferences.analytics_opt_in);

        if (preferences.analytics_opt_in) {
          void trackAppOpened();
          const routeId = routeIdFromPathname(window.location.pathname);
          if (routeId) {
            void trackRouteViewed(routeId);
          }
        }
      })
      .catch((error) => {
        console.error("Failed to bootstrap analytics preference:", error);
        if (!active) {
          return;
        }
        setAnalyticsEnabled(false);
      });

    return () => {
      active = false;
    };
  }, []);

  return children;
}
