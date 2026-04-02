import { ApiHttpClient } from "@/shared/api/http";

export interface UserPreferencesResponse {
  analytics_opt_in: boolean;
}

export interface UpdateAnalyticsPreferenceRequest {
  opt_in: boolean;
}

export class PreferencesApiClient {
  constructor(private readonly http: ApiHttpClient) {}

  async getPreferences(): Promise<UserPreferencesResponse> {
    return this.http.request("/preferences");
  }

  async updateAnalyticsPreference(
    request: UpdateAnalyticsPreferenceRequest,
  ): Promise<UserPreferencesResponse> {
    return this.http.request("/preferences/analytics", {
      method: "PUT",
      body: JSON.stringify(request),
    });
  }
}
