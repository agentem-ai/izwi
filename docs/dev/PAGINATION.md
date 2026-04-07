# Cursor Pagination Contract

Izwi list endpoints that can grow large use cursor pagination with keyset ordering.

## Query Parameters

- `limit` (optional): positive integer page size, clamped server-side.
- `cursor` (optional): opaque base64url token returned by the previous page.

## Response Shape

List responses include their original array field plus a `pagination` object:

```json
{
  "records": [],
  "pagination": {
    "next_cursor": "opaque-token-or-null",
    "has_more": true,
    "limit": 25
  }
}
```

- `next_cursor` is `null` at the end of the result set.
- `has_more` indicates whether another page is available.
- Clients should pass `next_cursor` back as `cursor` for the next page.

## Paginated Endpoints

- `GET /v1/transcriptions`
- `GET /v1/diarizations`
- `GET /v1/text-to-speech-generations`
- `GET /v1/voice-design-generations`
- `GET /v1/voice-clone-generations`
- `GET /v1/voices`
- `GET /v1/studio/projects`

## Frontend Compatibility Behavior

Frontend API helpers normalize pagination metadata and safely handle missing pagination fields by treating the result as a single page (`has_more=false`, `next_cursor=null`).
