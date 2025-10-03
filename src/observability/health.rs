use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use crate::error::{SearchError, SearchResult};

/// Health check service for Cloud Run readiness/liveness probes
#[derive(Clone)]
pub struct HealthService {
    components: Arc<RwLock<HashMap<String, ComponentHealth>>>,
}

/// Health status of individual components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub message: Option<String>,
    pub response_time_ms: Option<f64>,
}

/// Overall health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health check response
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub service: ServiceInfo,
    pub components: HashMap<String, ComponentHealth>,
    pub uptime_seconds: u64,
}

/// Service information
#[derive(Serialize, Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub version: String,
    pub environment: String,
}

/// Detailed health check response
#[derive(Serialize, Deserialize)]
pub struct DetailedHealthResponse {
    pub status: HealthStatus,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub service: ServiceInfo,
    pub components: HashMap<String, ComponentHealth>,
    pub uptime_seconds: u64,
    pub system: SystemHealth,
}

/// System health metrics
#[derive(Serialize, Deserialize)]
pub struct SystemHealth {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: Option<f64>,
    pub active_connections: u32,
    pub request_rate_per_second: f64,
}

impl HealthService {
    /// Create a new health service
    pub fn new() -> Self {
        Self {
            components: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update component health status
    pub async fn update_component_health(
        &self,
        component: &str,
        status: HealthStatus,
        message: Option<String>,
        response_time_ms: Option<f64>,
    ) {
        let health = ComponentHealth {
            status,
            last_check: chrono::Utc::now(),
            message,
            response_time_ms,
        };

        let mut components = self.components.write().await;
        components.insert(component.to_string(), health);
    }

    /// Check Redis health
    pub async fn check_redis_health(&self) -> (HealthStatus, Option<String>, Option<f64>) {
        let start = Instant::now();
        
        // This would be implemented with actual Redis client
        // For now, we'll simulate the check
        match self.simulate_redis_check().await {
            Ok(_) => {
                let duration = start.elapsed().as_secs_f64() * 1000.0;
                (HealthStatus::Healthy, None, Some(duration))
            }
            Err(e) => (
                HealthStatus::Unhealthy,
                Some(format!("Redis connection failed: {}", e)),
                None,
            ),
        }
    }

    /// Check PostgreSQL health
    pub async fn check_postgres_health(&self) -> (HealthStatus, Option<String>, Option<f64>) {
        let start = Instant::now();
        
        // This would be implemented with actual Postgres client
        // For now, we'll simulate the check
        match self.simulate_postgres_check().await {
            Ok(_) => {
                let duration = start.elapsed().as_secs_f64() * 1000.0;
                (HealthStatus::Healthy, None, Some(duration))
            }
            Err(e) => (
                HealthStatus::Unhealthy,
                Some(format!("PostgreSQL connection failed: {}", e)),
                None,
            ),
        }
    }

    /// Check ML model health
    pub async fn check_model_health(&self) -> (HealthStatus, Option<String>, Option<f64>) {
        let start = Instant::now();
        
        // This would be implemented with actual model inference
        // For now, we'll simulate the check
        match self.simulate_model_check().await {
            Ok(_) => {
                let duration = start.elapsed().as_secs_f64() * 1000.0;
                (HealthStatus::Healthy, None, Some(duration))
            }
            Err(e) => (
                HealthStatus::Degraded, // Models can be degraded but service still works
                Some(format!("Model inference slow/failed: {}", e)),
                None,
            ),
        }
    }

    /// Perform comprehensive health check
    pub async fn comprehensive_health_check(&self) -> HealthResponse {
        let start_time = Instant::now();

        // Check all components in parallel
        let (redis_result, postgres_result, model_result) = tokio::join!(
            self.check_redis_health(),
            self.check_postgres_health(),
            self.check_model_health()
        );

        // Update component health
        self.update_component_health(
            "redis",
            redis_result.0,
            redis_result.1,
            redis_result.2,
        ).await;

        self.update_component_health(
            "postgres",
            postgres_result.0,
            postgres_result.1,
            postgres_result.2,
        ).await;

        self.update_component_health(
            "ml_models",
            model_result.0,
            model_result.1,
            model_result.2,
        ).await;

        // Determine overall status
        let components = self.components.read().await;
        let overall_status = self.calculate_overall_status(&components);

        HealthResponse {
            status: overall_status,
            timestamp: chrono::Utc::now(),
            service: ServiceInfo {
                name: "rag-search-api".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                environment: std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()),
            },
            components: components.clone(),
            uptime_seconds: start_time.elapsed().as_secs(), // This would be actual uptime
        }
    }

    /// Calculate overall health status based on component health
    fn calculate_overall_status(&self, components: &HashMap<String, ComponentHealth>) -> HealthStatus {
        let mut has_unhealthy = false;
        let mut has_degraded = false;

        for health in components.values() {
            match health.status {
                HealthStatus::Unhealthy => {
                    // Critical components (Redis, Postgres) being unhealthy makes service unhealthy
                    has_unhealthy = true;
                }
                HealthStatus::Degraded => {
                    has_degraded = true;
                }
                HealthStatus::Healthy => {}
            }
        }

        if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    /// Get system health metrics
    async fn get_system_health(&self) -> SystemHealth {
        SystemHealth {
            memory_usage_mb: self.get_memory_usage(),
            cpu_usage_percent: None, // Would be implemented with system metrics
            active_connections: 0,   // Would be tracked from connection pools
            request_rate_per_second: 0.0, // Would be calculated from metrics
        }
    }

    /// Get memory usage in MB
    fn get_memory_usage(&self) -> f64 {
        // This would be implemented with actual system metrics
        // For now, return a placeholder
        0.0
    }

    // Simulation methods for testing (would be replaced with actual implementations)
    async fn simulate_redis_check(&self) -> Result<(), String> {
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(())
    }

    async fn simulate_postgres_check(&self) -> Result<(), String> {
        tokio::time::sleep(Duration::from_millis(2)).await;
        Ok(())
    }

    async fn simulate_model_check(&self) -> Result<(), String> {
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }
}

/// Liveness probe handler - basic check that service is running
pub async fn liveness_handler() -> Result<Json<serde_json::Value>, StatusCode> {
    Ok(Json(serde_json::json!({
        "status": "alive",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

/// Readiness probe handler - comprehensive health check
pub async fn readiness_handler(
    State(health_service): State<HealthService>,
) -> Result<Json<HealthResponse>, StatusCode> {
    let health = health_service.comprehensive_health_check().await;
    
    let status_code = match health.status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK, // Still ready to serve traffic
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    if status_code == StatusCode::OK {
        Ok(Json(health))
    } else {
        Err(status_code)
    }
}

/// Detailed health handler for monitoring/debugging
pub async fn health_handler(
    State(health_service): State<HealthService>,
) -> Result<Json<DetailedHealthResponse>, StatusCode> {
    let basic_health = health_service.comprehensive_health_check().await;
    let system_health = health_service.get_system_health().await;
    
    let detailed_health = DetailedHealthResponse {
        status: basic_health.status,
        timestamp: basic_health.timestamp,
        service: basic_health.service,
        components: basic_health.components,
        uptime_seconds: basic_health.uptime_seconds,
        system: system_health,
    };

    Ok(Json(detailed_health))
}

/// Create health check routes
pub fn health_routes() -> Router<HealthService> {
    Router::new()
        .route("/health/live", get(liveness_handler))
        .route("/health/ready", get(readiness_handler))
        .route("/health", get(health_handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_service_creation() {
        let service = HealthService::new();
        let components = service.components.read().await;
        assert!(components.is_empty());
    }

    #[tokio::test]
    async fn test_component_health_update() {
        let service = HealthService::new();
        
        service.update_component_health(
            "test_component",
            HealthStatus::Healthy,
            Some("All good".to_string()),
            Some(10.5),
        ).await;

        let components = service.components.read().await;
        let health = components.get("test_component").unwrap();
        
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.message, Some("All good".to_string()));
        assert_eq!(health.response_time_ms, Some(10.5));
    }

    #[tokio::test]
    async fn test_redis_health_check() {
        let service = HealthService::new();
        let (status, message, response_time) = service.check_redis_health().await;
        
        assert_eq!(status, HealthStatus::Healthy);
        assert!(message.is_none());
        assert!(response_time.is_some());
    }

    #[tokio::test]
    async fn test_postgres_health_check() {
        let service = HealthService::new();
        let (status, message, response_time) = service.check_postgres_health().await;
        
        assert_eq!(status, HealthStatus::Healthy);
        assert!(message.is_none());
        assert!(response_time.is_some());
    }

    #[tokio::test]
    async fn test_model_health_check() {
        let service = HealthService::new();
        let (status, message, response_time) = service.check_model_health().await;
        
        assert_eq!(status, HealthStatus::Healthy);
        assert!(message.is_none());
        assert!(response_time.is_some());
    }

    #[tokio::test]
    async fn test_comprehensive_health_check() {
        let service = HealthService::new();
        let health = service.comprehensive_health_check().await;
        
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.service.name, "rag-search-api");
        assert!(health.components.contains_key("redis"));
        assert!(health.components.contains_key("postgres"));
        assert!(health.components.contains_key("ml_models"));
    }

    #[test]
    fn test_overall_status_calculation() {
        let service = HealthService::new();
        
        // All healthy
        let mut components = HashMap::new();
        components.insert("redis".to_string(), ComponentHealth {
            status: HealthStatus::Healthy,
            last_check: chrono::Utc::now(),
            message: None,
            response_time_ms: Some(1.0),
        });
        components.insert("postgres".to_string(), ComponentHealth {
            status: HealthStatus::Healthy,
            last_check: chrono::Utc::now(),
            message: None,
            response_time_ms: Some(2.0),
        });
        
        assert_eq!(service.calculate_overall_status(&components), HealthStatus::Healthy);
        
        // One degraded
        components.insert("ml_models".to_string(), ComponentHealth {
            status: HealthStatus::Degraded,
            last_check: chrono::Utc::now(),
            message: Some("Slow inference".to_string()),
            response_time_ms: Some(100.0),
        });
        
        assert_eq!(service.calculate_overall_status(&components), HealthStatus::Degraded);
        
        // One unhealthy
        components.insert("redis".to_string(), ComponentHealth {
            status: HealthStatus::Unhealthy,
            last_check: chrono::Utc::now(),
            message: Some("Connection failed".to_string()),
            response_time_ms: None,
        });
        
        assert_eq!(service.calculate_overall_status(&components), HealthStatus::Unhealthy);
    }
}