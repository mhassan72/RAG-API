pub mod metrics;
pub mod tracing;
pub mod logging;
pub mod health;

#[cfg(test)]
mod tests;

pub use metrics::{Metrics, MetricsRegistry, Timer};
pub use tracing::{TracingService, init_tracing};
pub use logging::{LoggingService, init_logging};
pub use health::{HealthService, HealthStatus, ComponentHealth, health_routes};

use crate::error::SearchResult;

/// Comprehensive observability service that combines metrics, tracing, and logging
pub struct ObservabilityService {
    pub metrics: MetricsRegistry,
    pub tracing: TracingService,
    pub logging: LoggingService,
    pub health: HealthService,
}

impl ObservabilityService {
    /// Initialize all observability components
    pub async fn new() -> SearchResult<Self> {
        let metrics = MetricsRegistry::new()?;
        let tracing = TracingService::new().await?;
        let logging = LoggingService::new();
        let health = HealthService::new();

        Ok(Self {
            metrics,
            tracing,
            logging,
            health,
        })
    }

    /// Initialize global observability (tracing subscriber, etc.)
    pub async fn init_global(&self) -> SearchResult<()> {
        init_tracing().await?;
        init_logging()?;
        Ok(())
    }

    /// Shutdown observability services gracefully
    pub async fn shutdown(&self) -> SearchResult<()> {
        self.tracing.shutdown().await?;
        Ok(())
    }
}