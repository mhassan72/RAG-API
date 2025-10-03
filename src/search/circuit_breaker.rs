/// Circuit breaker implementation for Redis failure tracking and state management
/// 
/// This module implements the circuit breaker pattern to handle Redis failures gracefully
/// and provide automatic fallback to Postgres-only search when Redis is unavailable.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed = 0,   // Normal operation
    Open = 1,     // Circuit is open, failing fast
    HalfOpen = 2, // Testing if service has recovered
}

impl From<u8> for CircuitState {
    fn from(value: u8) -> Self {
        match value {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit
    pub failure_threshold: u32,
    /// Time to wait before transitioning from Open to HalfOpen
    pub recovery_timeout: Duration,
    /// Number of successful requests needed to close the circuit from HalfOpen
    pub success_threshold: u32,
    /// Time window for counting failures
    pub failure_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,                    // Open after 5 failures
            recovery_timeout: Duration::from_secs(30), // Wait 30s before trying again
            success_threshold: 3,                    // Need 3 successes to close
            failure_window: Duration::from_secs(60), // Count failures in 60s window
        }
    }
}

/// Circuit breaker for Redis operations
pub struct CircuitBreaker {
    /// Current circuit state
    state: AtomicU8,
    /// Redis failure count
    redis_failures: AtomicU32,
    /// Postgres failure count  
    postgres_failures: AtomicU32,
    /// Success count in HalfOpen state
    success_count: AtomicU32,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Last state change timestamp
    last_state_change: Arc<RwLock<Instant>>,
    /// Failure timestamps for windowing
    failure_timestamps: Arc<RwLock<Vec<Instant>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default configuration
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }

    /// Create a new circuit breaker with custom configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            redis_failures: AtomicU32::new(0),
            postgres_failures: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            config,
            last_state_change: Arc::new(RwLock::new(Instant::now())),
            failure_timestamps: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        CircuitState::from(self.state.load(Ordering::Acquire))
    }

    /// Check if Redis circuit is open (should use fallback)
    pub async fn is_redis_circuit_open(&self) -> bool {
        let current_state = self.state();
        
        match current_state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                // Check if we should transition to HalfOpen
                let last_change = *self.last_state_change.read().await;
                if last_change.elapsed() >= self.config.recovery_timeout {
                    self.transition_to_half_open().await;
                    false // Allow one request to test
                } else {
                    true // Still open
                }
            }
            CircuitState::HalfOpen => false, // Allow requests to test recovery
        }
    }

    /// Record a Redis operation success
    pub async fn record_redis_success(&self) {
        let current_state = self.state();
        
        match current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.redis_failures.store(0, Ordering::Release);
                self.clear_old_failures().await;
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::AcqRel) + 1;
                info!("Circuit breaker: Redis success in HalfOpen state ({}/{})", 
                      success_count, self.config.success_threshold);
                
                if success_count >= self.config.success_threshold {
                    self.transition_to_closed().await;
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but reset if it does
                warn!("Received success while circuit is Open - this shouldn't happen");
            }
        }
    }

    /// Record a Redis operation failure
    pub async fn record_redis_failure(&self) {
        let current_state = self.state();
        
        // Add failure timestamp
        {
            let mut timestamps = self.failure_timestamps.write().await;
            timestamps.push(Instant::now());
        }
        
        let failure_count = self.redis_failures.fetch_add(1, Ordering::AcqRel) + 1;
        
        match current_state {
            CircuitState::Closed => {
                // Clean old failures and check if we should open
                self.clear_old_failures().await;
                let recent_failures = self.count_recent_failures().await;
                
                warn!("Circuit breaker: Redis failure recorded ({} recent failures)", recent_failures);
                
                if recent_failures >= self.config.failure_threshold {
                    self.transition_to_open().await;
                }
            }
            CircuitState::HalfOpen => {
                warn!("Circuit breaker: Redis failure in HalfOpen state, reopening circuit");
                self.transition_to_open().await;
            }
            CircuitState::Open => {
                debug!("Circuit breaker: Additional failure while Open (total: {})", failure_count);
            }
        }
    }

    /// Record a Postgres operation success
    pub async fn record_postgres_success(&self) {
        self.postgres_failures.store(0, Ordering::Release);
        debug!("Circuit breaker: Postgres success recorded");
    }

    /// Record a Postgres operation failure
    pub async fn record_postgres_failure(&self) {
        let failure_count = self.postgres_failures.fetch_add(1, Ordering::AcqRel) + 1;
        warn!("Circuit breaker: Postgres failure recorded (total: {})", failure_count);
    }

    /// Get failure statistics
    pub async fn get_stats(&self) -> CircuitBreakerStats {
        let recent_failures = self.count_recent_failures().await;
        
        CircuitBreakerStats {
            state: self.state(),
            redis_failures: self.redis_failures.load(Ordering::Acquire),
            postgres_failures: self.postgres_failures.load(Ordering::Acquire),
            recent_failures,
            success_count: self.success_count.load(Ordering::Acquire),
        }
    }

    /// Transition to Open state
    async fn transition_to_open(&self) {
        let old_state = CircuitState::from(
            self.state.swap(CircuitState::Open as u8, Ordering::AcqRel)
        );
        
        if old_state != CircuitState::Open {
            *self.last_state_change.write().await = Instant::now();
            self.success_count.store(0, Ordering::Release);
            warn!("Circuit breaker: Transitioned from {:?} to Open", old_state);
        }
    }

    /// Transition to HalfOpen state
    async fn transition_to_half_open(&self) {
        let old_state = CircuitState::from(
            self.state.swap(CircuitState::HalfOpen as u8, Ordering::AcqRel)
        );
        
        if old_state != CircuitState::HalfOpen {
            *self.last_state_change.write().await = Instant::now();
            self.success_count.store(0, Ordering::Release);
            info!("Circuit breaker: Transitioned from {:?} to HalfOpen", old_state);
        }
    }

    /// Transition to Closed state
    async fn transition_to_closed(&self) {
        let old_state = CircuitState::from(
            self.state.swap(CircuitState::Closed as u8, Ordering::AcqRel)
        );
        
        if old_state != CircuitState::Closed {
            *self.last_state_change.write().await = Instant::now();
            self.redis_failures.store(0, Ordering::Release);
            self.success_count.store(0, Ordering::Release);
            
            // Clear old failure timestamps
            self.failure_timestamps.write().await.clear();
            
            info!("Circuit breaker: Transitioned from {:?} to Closed", old_state);
        }
    }

    /// Count failures within the failure window
    async fn count_recent_failures(&self) -> u32 {
        let now = Instant::now();
        let cutoff = now - self.config.failure_window;
        
        let timestamps = self.failure_timestamps.read().await;
        timestamps.iter()
            .filter(|&&timestamp| timestamp > cutoff)
            .count() as u32
    }

    /// Remove old failure timestamps outside the window
    async fn clear_old_failures(&self) {
        let now = Instant::now();
        let cutoff = now - self.config.failure_window;
        
        let mut timestamps = self.failure_timestamps.write().await;
        timestamps.retain(|&timestamp| timestamp > cutoff);
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker statistics for monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub redis_failures: u32,
    pub postgres_failures: u32,
    pub recent_failures: u32,
    pub success_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(!cb.is_redis_circuit_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config(config);

        // Record failures
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Closed);
        
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Closed);
        
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.is_redis_circuit_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config(config);

        // Open the circuit
        cb.record_redis_failure().await;
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for recovery timeout
        sleep(Duration::from_millis(150)).await;
        
        // Should transition to HalfOpen on next check
        assert!(!cb.is_redis_circuit_open().await);
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_after_successes() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            success_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config(config);

        // Open the circuit
        cb.record_redis_failure().await;
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait and transition to HalfOpen
        sleep(Duration::from_millis(100)).await;
        cb.is_redis_circuit_open().await; // Triggers transition
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Record successes
        cb.record_redis_success().await;
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        
        cb.record_redis_success().await;
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reopens_on_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config(config);

        // Open the circuit
        cb.record_redis_failure().await;
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait and transition to HalfOpen
        sleep(Duration::from_millis(100)).await;
        cb.is_redis_circuit_open().await;
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Failure in HalfOpen should reopen
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_failure_window_cleanup() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            failure_window: Duration::from_millis(100),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config(config);

        // Record failures
        cb.record_redis_failure().await;
        cb.record_redis_failure().await;
        
        // Wait for failures to age out
        sleep(Duration::from_millis(150)).await;
        
        // Should not open circuit as old failures are cleaned up
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_postgres_failure_tracking() {
        let cb = CircuitBreaker::new();
        
        cb.record_postgres_failure().await;
        let stats = cb.get_stats().await;
        assert_eq!(stats.postgres_failures, 1);
        
        cb.record_postgres_success().await;
        let stats = cb.get_stats().await;
        assert_eq!(stats.postgres_failures, 0);
    }
}