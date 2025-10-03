use super::*;
use crate::error::SearchError;
use axum::http::StatusCode;
use axum_test::TestServer;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[tokio::test]
async fn test_observability_service_initialization() {
    let observability = ObservabilityService::new().await;
    assert!(observability.is_ok());
    
    let obs = observability.unwrap();
    assert!(obs.init_global().await.is_ok());
    assert!(obs.shutdown().await.is_ok());
}

#[tokio::test]
async fn test_metrics_collection_integration() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Simulate search operations
    registry.metrics.search_total.inc();
    registry.metrics.search_duration_seconds.observe(0.045);
    registry.metrics.inflight_requests.set(5.0);
    registry.metrics.redis_hit_topk_ratio.set(0.85);
    registry.metrics.pg_tuples_returned.observe(25.0);
    registry.metrics.model_inference_seconds.observe(0.002);
    
    let output = registry.gather().unwrap();
    
    // Verify all required metrics are present
    assert!(output.contains("search_total"));
    assert!(output.contains("search_duration_seconds"));
    assert!(output.contains("redis_hit_topk_ratio"));
    assert!(output.contains("pg_tuples_returned"));
    assert!(output.contains("inflight_requests"));
    assert!(output.contains("model_inference_seconds"));
}

#[tokio::test]
async fn test_structured_logging_integration() {
    let logger = LoggingService::new();
    let trace_id = Uuid::new_v4();
    
    // Test various logging scenarios
    logger.log_search_request("test semantic search query", 10, Some("language:en"), trace_id);
    logger.log_search_response(trace_id, 45.2, 8, true, true, false);
    
    let mut context = HashMap::new();
    context.insert("component".to_string(), serde_json::json!("redis"));
    context.insert("operation".to_string(), serde_json::json!("vector_search"));
    
    let error = SearchError::RedisError("Connection timeout".to_string());
    logger.log_error(&error, Some(context));
    
    logger.log_model_inference("bi-encoder", 256, 1.5, true);
    logger.log_cache_operation("GET", "topk", "search:topk:hash123", true, Some(0.8));
    logger.log_database_operation("SELECT", "posts", 12.3, Some(25));
    logger.log_circuit_breaker_state("redis", "closed", "open");
    logger.log_gdpr_deletion("post_123", "cache_purge", true);
}

#[tokio::test]
async fn test_health_check_endpoints() {
    let health_service = HealthService::new();
    let app = health_routes().with_state(health_service.clone());
    let server = TestServer::new(app).unwrap();
    
    // Test liveness endpoint
    let response = server.get("/health/live").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    
    let json: serde_json::Value = response.json();
    assert_eq!(json["status"], "alive");
    assert!(json["timestamp"].is_string());
    
    // Test readiness endpoint
    let response = server.get("/health/ready").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    
    let health: health::HealthResponse = response.json();
    assert_eq!(health.status, HealthStatus::Healthy);
    assert_eq!(health.service.name, "rag-search-api");
    
    // Test detailed health endpoint
    let response = server.get("/health").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    
    let detailed_health: health::DetailedHealthResponse = response.json();
    assert_eq!(detailed_health.status, HealthStatus::Healthy);
    assert!(detailed_health.components.contains_key("redis"));
    assert!(detailed_health.components.contains_key("postgres"));
    assert!(detailed_health.components.contains_key("ml_models"));
}

#[tokio::test]
async fn test_health_service_component_updates() {
    let health_service = HealthService::new();
    
    // Update component health statuses
    health_service.update_component_health(
        "redis",
        HealthStatus::Healthy,
        None,
        Some(1.2),
    ).await;
    
    health_service.update_component_health(
        "postgres",
        HealthStatus::Degraded,
        Some("High latency detected".to_string()),
        Some(150.0),
    ).await;
    
    health_service.update_component_health(
        "ml_models",
        HealthStatus::Unhealthy,
        Some("Model inference failing".to_string()),
        None,
    ).await;
    
    let health = health_service.comprehensive_health_check().await;
    
    // Should be unhealthy due to ml_models being unhealthy
    assert_eq!(health.status, HealthStatus::Unhealthy);
    
    let redis_health = health.components.get("redis").unwrap();
    assert_eq!(redis_health.status, HealthStatus::Healthy);
    assert_eq!(redis_health.response_time_ms, Some(1.2));
    
    let postgres_health = health.components.get("postgres").unwrap();
    assert_eq!(postgres_health.status, HealthStatus::Degraded);
    assert_eq!(postgres_health.message, Some("High latency detected".to_string()));
    
    let ml_health = health.components.get("ml_models").unwrap();
    assert_eq!(ml_health.status, HealthStatus::Unhealthy);
    assert!(ml_health.message.is_some());
}

#[tokio::test]
async fn test_timer_functionality() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Test timer with actual operation
    let timer = Timer::new(registry.metrics.search_duration_seconds.clone());
    
    // Simulate some work
    sleep(Duration::from_millis(10)).await;
    
    timer.observe();
    
    // Verify the histogram recorded a value
    let output = registry.gather().unwrap();
    assert!(output.contains("search_duration_seconds"));
    
    // The bucket count should be > 0
    assert!(output.contains("search_duration_seconds_bucket"));
}

#[tokio::test]
async fn test_metrics_with_labels() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Test that metrics have proper labels
    registry.metrics.search_total.inc();
    registry.metrics.search_errors_total.inc();
    
    let output = registry.gather().unwrap();
    
    // Verify service label is present
    assert!(output.contains("service=\"rag-search-api\""));
}

#[tokio::test]
async fn test_logging_sanitization() {
    let logger = LoggingService::new();
    
    // Test email sanitization
    let query_with_email = "Find posts by john.doe@example.com about rust programming";
    logger.log_search_request(query_with_email, 10, None, Uuid::new_v4());
    
    // Test phone number sanitization
    let query_with_phone = "Contact support at 555-123-4567 for help";
    logger.log_search_request(query_with_phone, 5, None, Uuid::new_v4());
    
    // Test long query truncation
    let long_query = "a".repeat(300);
    logger.log_search_request(&long_query, 20, None, Uuid::new_v4());
}

#[tokio::test]
async fn test_circuit_breaker_metrics() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Simulate circuit breaker state changes
    registry.metrics.circuit_breaker_state.set(0.0); // Closed
    registry.metrics.circuit_breaker_failures_total.inc();
    
    registry.metrics.circuit_breaker_state.set(1.0); // Open
    registry.metrics.circuit_breaker_failures_total.inc();
    
    registry.metrics.circuit_breaker_state.set(2.0); // Half-open
    
    let output = registry.gather().unwrap();
    assert!(output.contains("circuit_breaker_state"));
    assert!(output.contains("circuit_breaker_failures_total"));
}

#[tokio::test]
async fn test_cache_metrics() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Simulate cache operations
    registry.metrics.cache_hits_total.inc();
    registry.metrics.cache_hits_total.inc();
    registry.metrics.cache_misses_total.inc();
    
    // Calculate hit ratio
    let hit_ratio = 2.0 / 3.0; // 2 hits out of 3 total
    registry.metrics.redis_hit_topk_ratio.set(hit_ratio);
    
    let output = registry.gather().unwrap();
    assert!(output.contains("cache_hits_total"));
    assert!(output.contains("cache_misses_total"));
    assert!(output.contains("redis_hit_topk_ratio"));
}

#[tokio::test]
async fn test_database_metrics() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Simulate database operations
    registry.metrics.pg_tuples_returned.observe(25.0);
    registry.metrics.pg_tuples_returned.observe(100.0);
    registry.metrics.pg_tuples_returned.observe(5.0);
    
    registry.metrics.pg_connections_active.set(8.0);
    registry.metrics.pg_query_duration_seconds.observe(0.025);
    
    let output = registry.gather().unwrap();
    assert!(output.contains("pg_tuples_returned"));
    assert!(output.contains("pg_connections_active"));
    assert!(output.contains("pg_query_duration_seconds"));
}

#[tokio::test]
async fn test_ml_inference_metrics() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Simulate model inference operations
    registry.metrics.model_inference_total.inc();
    registry.metrics.model_inference_seconds.observe(0.001); // 1ms inference
    
    registry.metrics.model_inference_total.inc();
    registry.metrics.model_inference_seconds.observe(0.005); // 5ms inference
    
    registry.metrics.model_inference_errors_total.inc();
    
    let output = registry.gather().unwrap();
    assert!(output.contains("model_inference_total"));
    assert!(output.contains("model_inference_seconds"));
    assert!(output.contains("model_inference_errors_total"));
}

#[tokio::test]
async fn test_http_metrics() {
    let registry = MetricsRegistry::new().unwrap();
    
    // Simulate HTTP requests
    registry.metrics.http_requests_total.inc();
    registry.metrics.http_request_duration_seconds.observe(0.045);
    registry.metrics.inflight_requests.inc();
    
    // Complete request
    registry.metrics.inflight_requests.dec();
    
    let output = registry.gather().unwrap();
    assert!(output.contains("http_requests_total"));
    assert!(output.contains("http_request_duration_seconds"));
    assert!(output.contains("inflight_requests"));
}

#[test]
fn test_error_type_classification() {
    let errors = vec![
        SearchError::InvalidRequest("Bad query".to_string()),
        SearchError::RateLimitExceeded,
        SearchError::Timeout,
        SearchError::RedisError("Connection failed".to_string()),
        SearchError::DatabaseError("Query failed".to_string()),
        SearchError::ModelError("Inference failed".to_string()),
        SearchError::Internal("Internal error".to_string()),
    ];
    
    let expected_types = vec![
        "invalid_request",
        "rate_limit_exceeded",
        "timeout",
        "redis_error",
        "database_error",
        "model_error",
        "internal_error",
    ];
    
    for (error, expected_type) in errors.iter().zip(expected_types.iter()) {
        assert_eq!(error.error_type(), *expected_type);
    }
}