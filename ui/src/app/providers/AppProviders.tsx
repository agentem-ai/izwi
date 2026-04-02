import type { ReactNode } from "react";
import { AnalyticsBootstrapProvider } from "@/app/analytics/AnalyticsBootstrapProvider";
import { ModelCatalogProvider } from "@/app/providers/ModelCatalogProvider";
import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { ThemeProvider } from "@/app/providers/ThemeProvider";

interface AppProvidersProps {
  children: ReactNode;
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <AnalyticsBootstrapProvider>
      <ThemeProvider>
        <NotificationProvider>
          <ModelCatalogProvider>{children}</ModelCatalogProvider>
        </NotificationProvider>
      </ThemeProvider>
    </AnalyticsBootstrapProvider>
  );
}
