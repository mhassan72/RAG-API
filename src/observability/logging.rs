use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use tracing::{event, Level};
use uuid::Uuid;
use crate::error::{SearchError, SearchResult};

/// Structured JSON logging service with trace_id injection
#[derive(Clone)]
pub struct LoggingService {
    service_name: String,
    service_version: String,
    environment: String,
}

impl LoggingService {
    /// Create a new logging service
    pub fn new() -> Self {
        let service_name = env::var("SERVICE_NAME")
            .unwrap_or_else(|_| "rag-search-api".to_string());
        
        let service_version = env::var("SERVICE_VERSION")
            .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string());
        
        let environment = env::var("ENVIRONMENT")
            .unwrap_or_else(|_| "development".to_string());

        Self {
            service_name,
            service_version,
            environment,
        }
    }

    /// Log a structured message with trace_id
    pub fn log_structured(&self, level: Level, message: &str, fields: Option<HashMap<String, Value>>) {
        let trace_id = Uuid::new_v4();
        
        let mut log_entry = json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "level": level.to_string().to_uppercase(),
            "message": message,
            "trace_id": trace_id.to_string(),
            "service": {
                "name": self.service_name,
                "version": self.service_version,
                "environment": self.environment
            }
        });

        // Add custom fields if provided
        if let Some(fields) = fields {
            if let Value::Object(ref mut map) = log_entry {
                for (key, value) in fields {
                    map.insert(key, value);
                }
            }
        }

        // Emit the structured log using tracing
        match level {
            Level::ERROR => event!(Level::ERROR, %trace_id, "{}", log_entry),
            Level::WARN => event!(Level::WARN, %trace_id, "{}", log_entry),
            Level::INFO => event!(Level::INFO, %trace_id, "{}", log_entry),
            Level::DEBUG => event!(Level::DEBUG, %trace_id, "{}", log_entry),
            Level::TRACE => event!(Level::TRACE, %trace_id, "{}", log_entry),
        }
    }

    /// Log search request with sanitized query
    pub fn log_search_request(&self, query: &str, k: u32, filters: Option<&str>, _trace_id: Uuid) {
        let sanitized_query = self.sanitize_query(query);
        
        let fields = HashMap::from([
            ("query_length".to_string(), json!(query.len())),
            ("k".to_string(), json!(k)),
            ("filters".to_string(), json!(filters.unwrap_or("none"))),
            ("sanitized_query".to_string(), json!(sanitized_query)),
        ]);

        self.log_structured(
            Level::INFO,
            "Search request received",
            Some(fields),
        );
    }

    /// Log search response with performance metrics
    pub fn log_search_response(
        &self,
        _trace_id: Uuid,
        duration_ms: f64,
        result_count: usize,
        cache_hit: bool,
        redis_used: bool,
        postgres_used: bool,
    ) {
        let fields = HashMap::from([
            ("duration_ms".to_string(), json!(duration_ms)),
            ("result_count".to_string(), json!(result_count)),
            ("cache_hit".to_string(), json!(cache_hit)),
            ("redis_used".to_string(), json!(redis_used)),
            ("postgres_used".to_string(), json!(postgres_used)),
        ]);

        self.log_structured(
            Level::INFO,
            "Search request completed",
            Some(fields),
        );
    }

    /// Log error with context
    pub fn log_error(&self, error: &SearchError, context: Option<HashMap<String, Value>>) {
        let mut fields = HashMap::from([
            ("error_type".to_string(), json!(error.error_type())),
            ("error_message".to_string(), json!(error.to_string())),
        ]);

        if let Some(context) = context {
            fields.extend(context);
        }

        self.log_structured(
            Level::ERROR,
            "Error occurred",
            Some(fields),
        );
    }

    /// Log cache operation
    pub fn log_cache_operation(
        &self,
        operation: &str,
        cache_type: &str,
        key: &str,
        hit: bool,
        duration_ms: Option<f64>,
    ) {
        let sanitized_key = self.sanitize_cache_key(key);
        
        let mut fields = HashMap::from([
            ("operation".to_string(), json!(operation)),
            ("cache_type".to_string(), json!(cache_type)),
            ("sanitized_key".to_string(), json!(sanitized_key)),
            ("hit".to_string(), json!(hit)),
        ]);

        if let Some(duration) = duration_ms {
            fields.insert("duration_ms".to_string(), json!(duration));
        }

        self.log_structured(
            Level::DEBUG,
            "Cache operation",
            Some(fields),
        );
    }

    /// Log database operation
    pub fn log_database_operation(
        &self,
        operation: &str,
        table: &str,
        duration_ms: f64,
        rows_affected: Option<usize>,
    ) {
        let mut fields = HashMap::from([
            ("operation".to_string(), json!(operation)),
            ("table".to_string(), json!(table)),
            ("duration_ms".to_string(), json!(duration_ms)),
        ]);

        if let Some(rows) = rows_affected {
            fields.insert("rows_affected".to_string(), json!(rows));
        }

        self.log_structured(
            Level::DEBUG,
            "Database operation",
            Some(fields),
        );
    }

    /// Log model inference
    pub fn log_model_inference(
        &self,
        model_type: &str,
        input_tokens: usize,
        duration_ms: f64,
        success: bool,
    ) {
        let fields = HashMap::from([
            ("model_type".to_string(), json!(model_type)),
            ("input_tokens".to_string(), json!(input_tokens)),
            ("duration_ms".to_string(), json!(duration_ms)),
            ("success".to_string(), json!(success)),
        ]);

        let level = if success { Level::DEBUG } else { Level::WARN };
        let message = if success {
            "Model inference completed"
        } else {
            "Model inference failed"
        };

        self.log_structured(level, message, Some(fields));
    }

    /// Log circuit breaker state change
    pub fn log_circuit_breaker_state(&self, component: &str, old_state: &str, new_state: &str) {
        let fields = HashMap::from([
            ("component".to_string(), json!(component)),
            ("old_state".to_string(), json!(old_state)),
            ("new_state".to_string(), json!(new_state)),
        ]);

        self.log_structured(
            Level::WARN,
            "Circuit breaker state changed",
            Some(fields),
        );
    }

    /// Log GDPR deletion operation
    pub fn log_gdpr_deletion(&self, post_id: &str, operation: &str, success: bool) {
        let fields = HashMap::from([
            ("post_id".to_string(), json!(post_id)),
            ("operation".to_string(), json!(operation)),
            ("success".to_string(), json!(success)),
            ("audit_timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339())),
        ]);

        let level = if success { Level::INFO } else { Level::ERROR };
        let message = format!("GDPR deletion {}", if success { "completed" } else { "failed" });

        self.log_structured(level, &message, Some(fields));
    }

    /// Sanitize query for logging (remove PII, truncate)
    fn sanitize_query(&self, query: &str) -> String {
        let mut sanitized = query.to_string();
        
        // Remove potential email addresses
        sanitized = regex::Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
            .unwrap_or_else(|_| regex::Regex::new(r"").unwrap())
            .replace_all(&sanitized, "[EMAIL]")
            .to_string();
        
        // Remove potential phone numbers
        sanitized = regex::Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
            .unwrap_or_else(|_| regex::Regex::new(r"").unwrap())
            .replace_all(&sanitized, "[PHONE]")
            .to_string();
        
        // Truncate if too long
        if sanitized.len() > 200 {
            sanitized.truncate(197);
            sanitized.push_str("...");
        }
        
        sanitized
    }

    /// Sanitize cache key for logging
    fn sanitize_cache_key(&self, key: &str) -> String {
        // For cache keys, we typically want to show the pattern but not the full key
        if key.len() > 50 {
            format!("{}...{}", &key[..20], &key[key.len()-10..])
        } else {
            key.to_string()
        }
    }
}

impl SearchError {
    /// Get error type as string for logging
    pub fn error_type(&self) -> &'static str {
        match self {
            SearchError::InvalidRequest(_) => "invalid_request",
            SearchError::RateLimitExceeded => "rate_limit_exceeded",
            SearchError::Timeout => "timeout",
            SearchError::RedisError(_) => "redis_error",
            SearchError::DatabaseError(_) => "database_error",
            SearchError::ModelError(_) => "model_error",
            SearchError::CacheError(_) => "cache_error",
            SearchError::ConfigError(_) => "config_error",
            SearchError::IoError(_) => "io_error",
            SearchError::SerializationError(_) => "serialization_error",
            SearchError::Internal(_) => "internal_error",
        }
    }
}

/// Initialize global logging configuration
pub fn init_logging() -> SearchResult<()> {
    // Logging is initialized through tracing subscriber in tracing.rs
    // This function exists for consistency and future logging-specific setup
    Ok(())
}

/// Macro for structured info logging with trace_id
#[macro_export]
macro_rules! log_info {
    ($logger:expr, $message:expr) => {
        $logger.log_structured(tracing::Level::INFO, $message, None)
    };
    ($logger:expr, $message:expr, $($field:tt)*) => {
        {
            let mut fields = std::collections::HashMap::new();
            $(
                fields.insert(stringify!($field).to_string(), serde_json::json!($field));
            )*
            $logger.log_structured(tracing::Level::INFO, $message, Some(fields))
        }
    };
}

/// Macro for structured error logging with trace_id
#[macro_export]
macro_rules! log_error {
    ($logger:expr, $message:expr) => {
        $logger.log_structured(tracing::Level::ERROR, $message, None)
    };
    ($logger:expr, $error:expr, $context:expr) => {
        $logger.log_error($error, Some($context))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_logging_service_creation() {
        let service = LoggingService::new();
        assert_eq!(service.service_name, "rag-search-api");
        assert!(!service.service_version.is_empty());
    }

    #[test]
    fn test_query_sanitization() {
        let service = LoggingService::new();
        
        let query_with_email = "Find posts by john.doe@example.com about rust";
        let sanitized = service.sanitize_query(query_with_email);
        assert!(sanitized.contains("[EMAIL]"));
        assert!(!sanitized.contains("john.doe@example.com"));
        
        let query_with_phone = "Contact me at 555-123-4567 for more info";
        let sanitized = service.sanitize_query(query_with_phone);
        assert!(sanitized.contains("[PHONE]"));
        assert!(!sanitized.contains("555-123-4567"));
    }

    #[test]
    fn test_cache_key_sanitization() {
        let service = LoggingService::new();
        
        let short_key = "search:topk:12345";
        let sanitized = service.sanitize_cache_key(short_key);
        assert_eq!(sanitized, short_key);
        
        let long_key = "search:topk:very_long_hash_key_that_should_be_truncated_for_logging_purposes";
        let sanitized = service.sanitize_cache_key(long_key);
        assert!(sanitized.len() < long_key.len());
        assert!(sanitized.contains("..."));
    }

    #[test]
    fn test_structured_logging() {
        let service = LoggingService::new();
        
        let mut fields = HashMap::new();
        fields.insert("test_field".to_string(), json!("test_value"));
        fields.insert("numeric_field".to_string(), json!(42));
        
        // This should not panic
        service.log_structured(Level::INFO, "Test message", Some(fields));
    }

    #[test]
    fn test_search_logging_methods() {
        let service = LoggingService::new();
        let trace_id = Uuid::new_v4();
        
        // Test search request logging
        service.log_search_request("test query", 10, Some("language:en"), trace_id);
        
        // Test search response logging
        service.log_search_response(trace_id, 45.5, 8, true, true, false);
        
        // Test model inference logging
        service.log_model_inference("bi-encoder", 256, 1.2, true);
        
        // Test cache operation logging
        service.log_cache_operation("GET", "topk", "search:topk:hash123", true, Some(0.5));
        
        // Test database operation logging
        service.log_database_operation("SELECT", "posts", 12.3, Some(25));
    }
}