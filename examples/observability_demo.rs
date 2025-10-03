use rag_search_api::observability::{
    ObservabilityService, MetricsRegistry, LoggingService, HealthService, HealthStatus
};
use rag_search_api::error::SearchError;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RAG Search API Observability Demo");
    
    // Initialize observability service
    println!("📊 Initializing observability service...");
    let observability = ObservabilityService::new().await?;
    observability.init_global().await?;
    
    // Demonstrate metrics collection
    println!("📈 Demonstrating metrics collection...");
    demonstrate_metrics(&observability.metrics).await;
    
    // Demonstrate structured logging
    println!("📝 Demonstrating structured logging...");
    demonstrate_logging(&observability.logging).await;
    
    // Demonstrate health checks
    println!("🏥 Demonstrating health checks...");
    demonstrate_health_checks(&observability.health).await;
    
    // Show metrics output
    println!("📊 Current metrics:");
    let metrics_output = observability.metrics.gather()?;
    println!("{}", metrics_output);
    
    // Graceful shutdown
    println!("🛑 Shutting down observability service...");
    observability.shutdown().await?;
    
    println!("✅ Observability demo completed successfully!");
    Ok(())
}

async fn demonstrate_metrics(registry: &MetricsRegistry) {
    println!("  • Recording search metrics...");
    
    // Simulate search operations
    for i in 0..5 {
        registry.metrics.search_total.inc();
        registry.metrics.search_duration_seconds.observe(0.045 + (i as f64 * 0.01));
        registry.metrics.inflight_requests.inc();
        
        // Simulate some processing time
        sleep(Duration::from_millis(10)).await;
        
        registry.metrics.inflight_requests.dec();
    }
    
    // Simulate cache operations
    registry.metrics.cache_hits_total.inc_by(3.0);
    registry.metrics.cache_misses_total.inc();
    registry.metrics.redis_hit_topk_ratio.set(0.75);
    
    // Simulate database operations
    registry.metrics.pg_tuples_returned.observe(25.0);
    registry.metrics.pg_connections_active.set(8.0);
    registry.metrics.pg_query_duration_seconds.observe(0.025);
    
    // Simulate ML inference
    registry.metrics.model_inference_total.inc_by(5.0);
    registry.metrics.model_inference_seconds.observe(0.002);
    
    // Simulate HTTP requests
    registry.metrics.http_requests_total.inc_by(5.0);
    registry.metrics.http_request_duration_seconds.observe(0.048);
    
    println!("  ✓ Metrics recorded successfully");
}

async fn demonstrate_logging(logger: &LoggingService) {
    println!("  • Demonstrating structured logging...");
    
    let trace_id = Uuid::new_v4();
    
    // Log search request
    logger.log_search_request(
        "semantic search for rust programming tutorials",
        10,
        Some("language:en"),
        trace_id,
    );
    
    // Log search response
    logger.log_search_response(
        trace_id,
        45.2,
        8,
        true,  // cache_hit
        true,  // redis_used
        false, // postgres_used
    );
    
    // Log model inference
    logger.log_model_inference("bi-encoder", 256, 1.5, true);
    
    // Log cache operation
    logger.log_cache_operation("GET", "topk", "search:topk:hash123", true, Some(0.8));
    
    // Log database operation
    logger.log_database_operation("SELECT", "posts", 12.3, Some(25));
    
    // Log circuit breaker state change
    logger.log_circuit_breaker_state("redis", "closed", "open");
    
    // Log GDPR deletion
    logger.log_gdpr_deletion("post_123", "cache_purge", true);
    
    // Log error
    let error = SearchError::RedisError("Connection timeout".to_string());
    logger.log_error(&error, None);
    
    println!("  ✓ Structured logs generated successfully");
}

async fn demonstrate_health_checks(health_service: &HealthService) {
    println!("  • Performing health checks...");
    
    // Update component health statuses
    health_service.update_component_health(
        "redis",
        HealthStatus::Healthy,
        None,
        Some(1.2),
    ).await;
    
    health_service.update_component_health(
        "postgres",
        HealthStatus::Healthy,
        None,
        Some(5.8),
    ).await;
    
    health_service.update_component_health(
        "ml_models",
        HealthStatus::Healthy,
        Some("All models loaded successfully".to_string()),
        Some(2.1),
    ).await;
    
    // Perform comprehensive health check
    let health = health_service.comprehensive_health_check().await;
    
    println!("  • Overall health status: {:?}", health.status);
    println!("  • Service: {} v{}", health.service.name, health.service.version);
    println!("  • Components checked: {}", health.components.len());
    
    for (component, health) in &health.components {
        println!(
            "    - {}: {:?} ({}ms)",
            component,
            health.status,
            health.response_time_ms.unwrap_or(0.0)
        );
    }
    
    println!("  ✓ Health checks completed successfully");
}