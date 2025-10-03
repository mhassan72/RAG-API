use prometheus::{
    Counter, Histogram, Gauge, Registry, Encoder, TextEncoder,
    HistogramOpts, Opts,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use crate::error::{SearchError, SearchResult};

/// Prometheus metrics registry and collectors
#[derive(Clone)]
pub struct MetricsRegistry {
    registry: Arc<Registry>,
    pub metrics: Arc<Metrics>,
}

/// All application metrics
pub struct Metrics {
    // Search metrics
    pub search_total: Counter,
    pub search_duration_seconds: Histogram,
    pub search_errors_total: Counter,
    
    // Cache metrics
    pub redis_hit_topk_ratio: Gauge,
    pub cache_hits_total: Counter,
    pub cache_misses_total: Counter,
    
    // Database metrics
    pub pg_tuples_returned: Histogram,
    pub pg_connections_active: Gauge,
    pub pg_query_duration_seconds: Histogram,
    
    // System metrics
    pub inflight_requests: Gauge,
    pub http_requests_total: Counter,
    pub http_request_duration_seconds: Histogram,
    
    // ML metrics
    pub model_inference_seconds: Histogram,
    pub model_inference_total: Counter,
    pub model_inference_errors_total: Counter,
    
    // Circuit breaker metrics
    pub circuit_breaker_state: Gauge,
    pub circuit_breaker_failures_total: Counter,
    
    // Health metrics
    pub health_check_duration_seconds: Histogram,
    pub component_health_status: Gauge,
}

impl MetricsRegistry {
    /// Create a new metrics registry with all collectors
    pub fn new() -> SearchResult<Self> {
        let registry = Arc::new(Registry::new());
        let metrics = Arc::new(Metrics::new(&registry)?);
        
        Ok(Self {
            registry,
            metrics,
        })
    }

    /// Get metrics in Prometheus text format
    pub fn gather(&self) -> SearchResult<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)
            .map_err(|e| SearchError::Internal(format!("Failed to encode metrics: {}", e)))?;
        
        String::from_utf8(buffer)
            .map_err(|e| SearchError::Internal(format!("Failed to convert metrics to string: {}", e)))
    }

    /// Get the underlying registry for middleware integration
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
}

impl Metrics {
    fn new(registry: &Registry) -> SearchResult<Self> {
        // Search metrics
        let search_total = Counter::new("search_total", "Total number of search requests processed")
            .map_err(|e| SearchError::Internal(format!("Failed to create search_total metric: {}", e)))?;
        
        let search_duration_seconds = Histogram::with_opts(HistogramOpts::new(
            "search_duration_seconds",
            "Duration of search requests in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]))
        .map_err(|e| SearchError::Internal(format!("Failed to create search_duration_seconds metric: {}", e)))?;
        
        let search_errors_total = Counter::new("search_errors_total", "Total number of search errors")
            .map_err(|e| SearchError::Internal(format!("Failed to create search_errors_total metric: {}", e)))?;

        // Cache metrics
        let redis_hit_topk_ratio = Gauge::new("redis_hit_topk_ratio", "Ratio of Redis top-k cache hits")
            .map_err(|e| SearchError::Internal(format!("Failed to create redis_hit_topk_ratio metric: {}", e)))?;
        
        let cache_hits_total = Counter::new("cache_hits_total", "Total number of cache hits")
            .map_err(|e| SearchError::Internal(format!("Failed to create cache_hits_total metric: {}", e)))?;
        
        let cache_misses_total = Counter::new("cache_misses_total", "Total number of cache misses")
            .map_err(|e| SearchError::Internal(format!("Failed to create cache_misses_total metric: {}", e)))?;

        // Database metrics
        let pg_tuples_returned = Histogram::with_opts(HistogramOpts::new(
            "pg_tuples_returned",
            "Number of tuples returned by PostgreSQL queries"
        ).buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]))
        .map_err(|e| SearchError::Internal(format!("Failed to create pg_tuples_returned metric: {}", e)))?;
        
        let pg_connections_active = Gauge::new("pg_connections_active", "Number of active PostgreSQL connections")
            .map_err(|e| SearchError::Internal(format!("Failed to create pg_connections_active metric: {}", e)))?;
        
        let pg_query_duration_seconds = Histogram::with_opts(HistogramOpts::new(
            "pg_query_duration_seconds",
            "Duration of PostgreSQL queries in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]))
        .map_err(|e| SearchError::Internal(format!("Failed to create pg_query_duration_seconds metric: {}", e)))?;

        // System metrics
        let inflight_requests = Gauge::new("inflight_requests", "Number of requests currently being processed")
            .map_err(|e| SearchError::Internal(format!("Failed to create inflight_requests metric: {}", e)))?;
        
        let http_requests_total = Counter::new("http_requests_total", "Total number of HTTP requests")
            .map_err(|e| SearchError::Internal(format!("Failed to create http_requests_total metric: {}", e)))?;
        
        let http_request_duration_seconds = Histogram::with_opts(HistogramOpts::new(
            "http_request_duration_seconds",
            "Duration of HTTP requests in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]))
        .map_err(|e| SearchError::Internal(format!("Failed to create http_request_duration_seconds metric: {}", e)))?;

        // ML metrics
        let model_inference_seconds = Histogram::with_opts(HistogramOpts::new(
            "model_inference_seconds",
            "Duration of ML model inference in seconds"
        ).buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]))
        .map_err(|e| SearchError::Internal(format!("Failed to create model_inference_seconds metric: {}", e)))?;
        
        let model_inference_total = Counter::new("model_inference_total", "Total number of model inferences")
            .map_err(|e| SearchError::Internal(format!("Failed to create model_inference_total metric: {}", e)))?;
        
        let model_inference_errors_total = Counter::new("model_inference_errors_total", "Total number of model inference errors")
            .map_err(|e| SearchError::Internal(format!("Failed to create model_inference_errors_total metric: {}", e)))?;

        // Circuit breaker metrics
        let circuit_breaker_state = Gauge::new("circuit_breaker_state", "Circuit breaker state (0=closed, 1=open, 2=half-open)")
            .map_err(|e| SearchError::Internal(format!("Failed to create circuit_breaker_state metric: {}", e)))?;
        
        let circuit_breaker_failures_total = Counter::new("circuit_breaker_failures_total", "Total number of circuit breaker failures")
            .map_err(|e| SearchError::Internal(format!("Failed to create circuit_breaker_failures_total metric: {}", e)))?;

        // Health metrics
        let health_check_duration_seconds = Histogram::with_opts(HistogramOpts::new(
            "health_check_duration_seconds",
            "Duration of health checks in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]))
        .map_err(|e| SearchError::Internal(format!("Failed to create health_check_duration_seconds metric: {}", e)))?;
        
        let component_health_status = Gauge::new("component_health_status", "Health status of components (1=healthy, 0=unhealthy)")
            .map_err(|e| SearchError::Internal(format!("Failed to create component_health_status metric: {}", e)))?;

        // Register all metrics
        registry.register(Box::new(search_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register search_total: {}", e)))?;
        registry.register(Box::new(search_duration_seconds.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register search_duration_seconds: {}", e)))?;
        registry.register(Box::new(search_errors_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register search_errors_total: {}", e)))?;
        registry.register(Box::new(redis_hit_topk_ratio.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register redis_hit_topk_ratio: {}", e)))?;
        registry.register(Box::new(cache_hits_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register cache_hits_total: {}", e)))?;
        registry.register(Box::new(cache_misses_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register cache_misses_total: {}", e)))?;
        registry.register(Box::new(pg_tuples_returned.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register pg_tuples_returned: {}", e)))?;
        registry.register(Box::new(pg_connections_active.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register pg_connections_active: {}", e)))?;
        registry.register(Box::new(pg_query_duration_seconds.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register pg_query_duration_seconds: {}", e)))?;
        registry.register(Box::new(inflight_requests.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register inflight_requests: {}", e)))?;
        registry.register(Box::new(http_requests_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register http_requests_total: {}", e)))?;
        registry.register(Box::new(http_request_duration_seconds.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register http_request_duration_seconds: {}", e)))?;
        registry.register(Box::new(model_inference_seconds.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register model_inference_seconds: {}", e)))?;
        registry.register(Box::new(model_inference_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register model_inference_total: {}", e)))?;
        registry.register(Box::new(model_inference_errors_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register model_inference_errors_total: {}", e)))?;
        registry.register(Box::new(circuit_breaker_state.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register circuit_breaker_state: {}", e)))?;
        registry.register(Box::new(circuit_breaker_failures_total.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register circuit_breaker_failures_total: {}", e)))?;
        registry.register(Box::new(health_check_duration_seconds.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register health_check_duration_seconds: {}", e)))?;
        registry.register(Box::new(component_health_status.clone()))
            .map_err(|e| SearchError::Internal(format!("Failed to register component_health_status: {}", e)))?;

        Ok(Self {
            search_total,
            search_duration_seconds,
            search_errors_total,
            redis_hit_topk_ratio,
            cache_hits_total,
            cache_misses_total,
            pg_tuples_returned,
            pg_connections_active,
            pg_query_duration_seconds,
            inflight_requests,
            http_requests_total,
            http_request_duration_seconds,
            model_inference_seconds,
            model_inference_total,
            model_inference_errors_total,
            circuit_breaker_state,
            circuit_breaker_failures_total,
            health_check_duration_seconds,
            component_health_status,
        })
    }
}

/// Timer helper for measuring durations
pub struct Timer {
    start: Instant,
    histogram: Histogram,
}

impl Timer {
    pub fn new(histogram: Histogram) -> Self {
        Self {
            start: Instant::now(),
            histogram,
        }
    }

    pub fn observe(self) {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}

/// Macro for timing operations
#[macro_export]
macro_rules! time_operation {
    ($histogram:expr, $operation:expr) => {{
        let timer = $crate::observability::metrics::Timer::new($histogram.clone());
        let result = $operation;
        timer.observe();
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registry_creation() {
        let registry = MetricsRegistry::new().unwrap();
        assert!(registry.gather().is_ok());
    }

    #[test]
    fn test_metrics_collection() {
        let registry = MetricsRegistry::new().unwrap();
        
        // Increment some counters
        registry.metrics.search_total.inc();
        registry.metrics.cache_hits_total.inc();
        
        // Set some gauges
        registry.metrics.inflight_requests.set(5.0);
        registry.metrics.redis_hit_topk_ratio.set(0.85);
        
        // Record some histograms
        registry.metrics.search_duration_seconds.observe(0.05);
        registry.metrics.model_inference_seconds.observe(0.001);
        
        let output = registry.gather().unwrap();
        assert!(output.contains("search_total"));
        assert!(output.contains("cache_hits_total"));
        assert!(output.contains("inflight_requests"));
        assert!(output.contains("redis_hit_topk_ratio"));
    }

    #[test]
    fn test_timer_functionality() {
        let registry = MetricsRegistry::new().unwrap();
        let timer = Timer::new(registry.metrics.search_duration_seconds.clone());
        
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        timer.observe();
        
        // Verify the histogram recorded a value
        let output = registry.gather().unwrap();
        assert!(output.contains("search_duration_seconds"));
    }
}