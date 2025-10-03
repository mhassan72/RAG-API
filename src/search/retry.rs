/// Retry logic with exponential backoff for search operations
/// 
/// This module implements retry strategies with exponential backoff and jitter
/// to handle transient failures gracefully while avoiding thundering herd problems.

use crate::error::{SearchError, SearchResult};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};
use rand::Rng;

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Base delay for exponential backoff
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Jitter factor (0.0 to 1.0) to add randomness
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100), // 100ms, 200ms, 400ms
            max_delay: Duration::from_millis(1000),
            jitter_factor: 0.1, // 10% jitter
        }
    }
}

/// Retry strategy for different types of operations
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// Exponential backoff with jitter
    ExponentialBackoff(RetryConfig),
    /// Fixed delay between retries
    FixedDelay(Duration, u32),
    /// No retries
    None,
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self::ExponentialBackoff(RetryConfig::default())
    }
}

/// Retry executor that handles different retry strategies
pub struct RetryExecutor {
    strategy: RetryStrategy,
}

impl RetryExecutor {
    /// Create a new retry executor with the given strategy
    pub fn new(strategy: RetryStrategy) -> Self {
        Self { strategy }
    }

    /// Create a retry executor with default exponential backoff
    pub fn with_exponential_backoff() -> Self {
        Self::new(RetryStrategy::default())
    }

    /// Create a retry executor with custom exponential backoff config
    pub fn with_config(config: RetryConfig) -> Self {
        Self::new(RetryStrategy::ExponentialBackoff(config))
    }

    /// Execute an operation with retry logic
    pub async fn execute<F, Fut, T>(&self, operation: F) -> SearchResult<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = SearchResult<T>>,
    {
        match &self.strategy {
            RetryStrategy::ExponentialBackoff(config) => {
                self.execute_with_exponential_backoff(operation, config).await
            }
            RetryStrategy::FixedDelay(delay, max_retries) => {
                self.execute_with_fixed_delay(operation, *delay, *max_retries).await
            }
            RetryStrategy::None => operation().await,
        }
    }

    /// Execute operation with exponential backoff
    async fn execute_with_exponential_backoff<F, Fut, T>(
        &self,
        operation: F,
        config: &RetryConfig,
    ) -> SearchResult<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = SearchResult<T>>,
    {
        let mut last_error = None;
        
        for attempt in 0..=config.max_retries {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        debug!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error.clone());
                    
                    // Don't retry on certain error types
                    if !self.should_retry(&error) {
                        debug!("Not retrying error: {}", error);
                        return Err(error);
                    }
                    
                    // Don't sleep after the last attempt
                    if attempt < config.max_retries {
                        let delay = self.calculate_exponential_delay(attempt, config);
                        warn!(
                            "Operation failed (attempt {}/{}), retrying in {:?}: {}",
                            attempt + 1,
                            config.max_retries + 1,
                            delay,
                            error
                        );
                        sleep(delay).await;
                    } else {
                        warn!(
                            "Operation failed after {} attempts: {}",
                            config.max_retries + 1,
                            error
                        );
                    }
                }
            }
        }
        
        // Return the last error if all retries failed
        Err(last_error.unwrap_or_else(|| {
            SearchError::Internal("Retry logic error: no attempts made".to_string())
        }))
    }

    /// Execute operation with fixed delay
    async fn execute_with_fixed_delay<F, Fut, T>(
        &self,
        operation: F,
        delay: Duration,
        max_retries: u32,
    ) -> SearchResult<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = SearchResult<T>>,
    {
        let mut last_error = None;
        
        for attempt in 0..=max_retries {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        debug!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error.clone());
                    
                    if !self.should_retry(&error) {
                        return Err(error);
                    }
                    
                    if attempt < max_retries {
                        warn!(
                            "Operation failed (attempt {}/{}), retrying in {:?}: {}",
                            attempt + 1,
                            max_retries + 1,
                            delay,
                            error
                        );
                        sleep(delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| {
            SearchError::Internal("Retry logic error: no attempts made".to_string())
        }))
    }

    /// Calculate exponential backoff delay with jitter
    fn calculate_exponential_delay(&self, attempt: u32, config: &RetryConfig) -> Duration {
        // Calculate base exponential delay: base_delay * 2^attempt
        let exponential_delay = config.base_delay.as_millis() as u64 * (1u64 << attempt);
        let exponential_delay = Duration::from_millis(exponential_delay);
        
        // Cap at max_delay
        let capped_delay = std::cmp::min(exponential_delay, config.max_delay);
        
        // Add jitter to prevent thundering herd
        if config.jitter_factor > 0.0 {
            let jitter_range = (capped_delay.as_millis() as f64 * config.jitter_factor) as u64;
            let jitter = rand::thread_rng().gen_range(0..=jitter_range);
            Duration::from_millis(capped_delay.as_millis() as u64 + jitter)
        } else {
            capped_delay
        }
    }

    /// Determine if an error should be retried
    fn should_retry(&self, error: &SearchError) -> bool {
        match error {
            // Retry on transient errors
            SearchError::RedisError(_) => true,
            SearchError::DatabaseError(_) => true,
            SearchError::Timeout => true,
            SearchError::Internal(_) => true,
            
            // Don't retry on client errors
            SearchError::InvalidRequest(_) => false,
            SearchError::RateLimitExceeded => false,
            
            // Don't retry on model errors (likely persistent)
            SearchError::ModelError(_) => false,
            
            // Don't retry on configuration errors
            SearchError::ConfigError(_) => false,
            
            // Retry on other errors
            SearchError::CacheError(_) => true,
            SearchError::IoError(_) => true,
            SearchError::SerializationError(_) => false,
        }
    }
}

/// Convenience function for retrying operations with default config
pub async fn retry_with_exponential_backoff<F, Fut, T>(operation: F) -> SearchResult<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = SearchResult<T>>,
{
    let executor = RetryExecutor::with_exponential_backoff();
    executor.execute(operation).await
}

/// Convenience function for retrying operations with custom config
pub async fn retry_with_config<F, Fut, T>(
    operation: F,
    config: RetryConfig,
) -> SearchResult<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = SearchResult<T>>,
{
    let executor = RetryExecutor::with_config(config);
    executor.execute(operation).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_on_first_attempt() {
        let executor = RetryExecutor::with_exponential_backoff();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let result = executor.execute(|| async {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok::<i32, SearchError>(42)
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1), // Fast for testing
            ..Default::default()
        };
        let executor = RetryExecutor::with_config(config);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let result = executor.execute(|| async {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count < 2 {
                Err(SearchError::RedisError("Temporary failure".to_string()))
            } else {
                Ok::<i32, SearchError>(42)
            }
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        let config = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let executor = RetryExecutor::with_config(config);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let result = executor.execute(|| async {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Err::<i32, SearchError>(SearchError::RedisError("Persistent failure".to_string()))
        }).await;
        
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_no_retry_on_client_errors() {
        let executor = RetryExecutor::with_exponential_backoff();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let result = executor.execute(|| async {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Err::<i32, SearchError>(SearchError::InvalidRequest("Bad request".to_string()))
        }).await;
        
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1); // No retries
    }

    #[tokio::test]
    async fn test_exponential_backoff_calculation() {
        let config = RetryConfig {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(1000),
            jitter_factor: 0.0, // No jitter for predictable testing
            ..Default::default()
        };
        let executor = RetryExecutor::with_config(config.clone());
        
        // Test delay calculation
        let delay0 = executor.calculate_exponential_delay(0, &config);
        let delay1 = executor.calculate_exponential_delay(1, &config);
        let delay2 = executor.calculate_exponential_delay(2, &config);
        
        assert_eq!(delay0, Duration::from_millis(100)); // 100 * 2^0
        assert_eq!(delay1, Duration::from_millis(200)); // 100 * 2^1
        assert_eq!(delay2, Duration::from_millis(400)); // 100 * 2^2
    }

    #[tokio::test]
    async fn test_max_delay_cap() {
        let config = RetryConfig {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(300),
            jitter_factor: 0.0,
            ..Default::default()
        };
        let executor = RetryExecutor::with_config(config.clone());
        
        let delay3 = executor.calculate_exponential_delay(3, &config);
        assert_eq!(delay3, Duration::from_millis(300)); // Capped at max_delay
    }

    #[tokio::test]
    async fn test_fixed_delay_strategy() {
        let executor = RetryExecutor::new(RetryStrategy::FixedDelay(
            Duration::from_millis(1),
            2,
        ));
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let result = executor.execute(|| async {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count < 1 {
                Err(SearchError::RedisError("Failure".to_string()))
            } else {
                Ok::<i32, SearchError>(42)
            }
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_no_retry_strategy() {
        let executor = RetryExecutor::new(RetryStrategy::None);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let result = executor.execute(|| async {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Err::<i32, SearchError>(SearchError::RedisError("Failure".to_string()))
        }).await;
        
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1); // No retries
    }
}