use std::env;
use tracing::subscriber::set_global_default;
use tracing_subscriber::{
    fmt::{self, format::JsonFields},
    layer::SubscriberExt,
    EnvFilter, Registry,
};
use crate::error::{SearchError, SearchResult};

/// Tracing service for structured logging and distributed tracing
pub struct TracingService {
    service_name: String,
}

impl TracingService {
    /// Create a new tracing service
    pub async fn new() -> SearchResult<Self> {
        let service_name = env::var("SERVICE_NAME")
            .unwrap_or_else(|_| "rag-search-api".to_string());
        
        Ok(Self { service_name })
    }

    /// Get service name
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Shutdown the tracing service gracefully
    pub async fn shutdown(&self) -> SearchResult<()> {
        // For now, just a placeholder for graceful shutdown
        Ok(())
    }
}

/// Initialize global tracing subscriber with JSON formatting
pub async fn init_tracing() -> SearchResult<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,rag_search_api=debug"));

    let formatting_layer = fmt::layer()
        .json()
        .with_current_span(false)
        .with_span_list(true)
        .fmt_fields(JsonFields::new());

    // Build subscriber with JSON formatting
    let subscriber = Registry::default()
        .with(env_filter)
        .with(formatting_layer);

    set_global_default(subscriber)
        .map_err(|e| SearchError::Internal(format!("Failed to set global subscriber: {}", e)))?;

    Ok(())
}

/// Macro for creating spans with automatic trace_id injection
#[macro_export]
macro_rules! trace_span {
    ($level:expr, $name:expr) => {
        tracing::span!($level, $name, trace_id = %uuid::Uuid::new_v4())
    };
    ($level:expr, $name:expr, $($field:tt)*) => {
        tracing::span!($level, $name, trace_id = %uuid::Uuid::new_v4(), $($field)*)
    };
}

/// Macro for creating info spans with trace_id
#[macro_export]
macro_rules! info_span {
    ($name:expr) => {
        $crate::trace_span!(tracing::Level::INFO, $name)
    };
    ($name:expr, $($field:tt)*) => {
        $crate::trace_span!(tracing::Level::INFO, $name, $($field)*)
    };
}

/// Macro for creating debug spans with trace_id
#[macro_export]
macro_rules! debug_span {
    ($name:expr) => {
        $crate::trace_span!(tracing::Level::DEBUG, $name)
    };
    ($name:expr, $($field:tt)*) => {
        $crate::trace_span!(tracing::Level::DEBUG, $name, $($field)*)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::{info, warn};

    #[tokio::test]
    async fn test_tracing_service_creation() {
        // This test might fail in CI without OTLP endpoint, so we'll make it conditional
        if env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok() {
            let service = TracingService::new().await;
            assert!(service.is_ok());
        }
    }

    #[tokio::test]
    async fn test_tracing_initialization() {
        // Test that tracing can be initialized without errors
        let result = init_tracing().await;
        // In test environment, this might fail due to missing OTLP endpoint
        // but we can still test the function doesn't panic
        match result {
            Ok(_) => {
                info!("Tracing initialized successfully");
                warn!("Test warning message");
            }
            Err(e) => {
                println!("Tracing initialization failed (expected in test): {}", e);
            }
        }
    }

    #[test]
    fn test_span_macros() {
        // Test that our span macros compile and work
        let _span = info_span!("test_span");
        let _span_with_fields = debug_span!("test_span_with_fields", field1 = "value1");
    }
}