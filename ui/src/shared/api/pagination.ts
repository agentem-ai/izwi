export interface CursorPaginationMeta {
  next_cursor: string | null;
  has_more: boolean;
  limit: number;
}

export interface CursorPaginationQuery {
  limit?: number;
  cursor?: string | null;
}

export interface CursorPageResult<T> {
  items: T[];
  pagination: CursorPaginationMeta;
}

export function buildCursorQueryString(
  query?: CursorPaginationQuery,
): string {
  if (!query) {
    return "";
  }

  const params = new URLSearchParams();
  if (query.limit && Number.isFinite(query.limit) && query.limit > 0) {
    params.set("limit", String(Math.floor(query.limit)));
  }
  if (query.cursor && query.cursor.trim().length > 0) {
    params.set("cursor", query.cursor.trim());
  }

  const serialized = params.toString();
  return serialized ? `?${serialized}` : "";
}

export function normalizeCursorPaginationMeta(
  value: Partial<CursorPaginationMeta> | null | undefined,
  fallbackLimit: number,
): CursorPaginationMeta {
  return {
    next_cursor: value?.next_cursor ?? null,
    has_more: Boolean(value?.has_more),
    limit:
      typeof value?.limit === "number" && Number.isFinite(value.limit)
        ? Math.max(1, Math.floor(value.limit))
        : Math.max(1, Math.floor(fallbackLimit)),
  };
}
