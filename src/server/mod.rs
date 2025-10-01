use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::post,
    Router,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, error};

use crate::error::{SearchError, SearchResult};
use crate::types::{SearchRequest, SearchResponse};

/// Main search server structure
pub struct SearchServer {
    app: Router,
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    // TODO: Add ML service, cache manager, etc. in future tasks
}

impl SearchServer {
    /// Create a new search server instance
    pub async fn new() -> SearchResult<Self> {
        let state = Arc::new(AppState {});

        let app = Router::new()
            .route("/semantic-search", post(semantic_search_handler))
            .route("/health", axum::routing::get(health_handler))
            .with_state(state);

        Ok(SearchServer { app })
    }

    /// Run the server
    pub async fn run(self) -> SearchResult<()> {
        let listener = TcpListener::bind("0.0.0.0:8080")
            .await
            .map_err(|e| SearchError::ConfigError(format!("Failed to bind to port 8080: {}", e)))?;

        info!("Server listening on 0.0.0.0:8080");

        axum::serve(listener, self.app)
            .await
            .map_err(|e| SearchError::Internal(format!("Server error: {}", e)))?;

        Ok(())
    }
}

/// Handler for semantic search endpoint
async fn semantic_search_handler(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<Vec<SearchResponse>>, (StatusCode, Json<ErrorResponse>)> {
    // Validate request
    if let Err(validation_error) = request.validate() {
        error!("Invalid request: {}", validation_error);
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid request".to_string(),
                message: validation_error,
            }),
        ));
    }

    info!("Processing search request for query: {}", request.query);

    // TODO: Implement actual search logic in future tasks
    // For now, return empty results
    Ok(Json(vec![]))
}

/// Handler for health check endpoint
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now(),
    })
}

/// Error response structure
#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

/// Health check response structure
#[derive(serde::Serialize)]
struct HealthResponse {
    status: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}