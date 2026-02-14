use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone, Debug)]
pub struct RequestContext {
    pub correlation_id: String,
}

pub async fn attach_request_context(mut req: Request, next: Next) -> Response {
    let correlation_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|h| h.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    req.extensions_mut().insert(RequestContext {
        correlation_id: correlation_id.clone(),
    });

    let mut response = next.run(req).await;
    if let Ok(value) = correlation_id.parse() {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}
