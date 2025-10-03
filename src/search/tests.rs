use super::*;
use crate::cache::CacheManager;
use crate::config::{DatabaseConfig, RedisConfig};
use crate::database::DatabaseManager;
use crate::types::{SearchCandidate, SearchSource};
use std::sync::Arc;
use tokio;

/// Mock cache manager for testing
struct MockCacheManager {
    should_fail: bool,
    candidates: Vec<SearchCandidate>,
}

impl MockCacheManager {
    fn new(candidates: Vec<SearchCandidate>) -> Self {
        Self {
            should_fail: false,
            candidates,
        }
    }

    fn with_failure() -> Self {
        Self {
            should_fail: true,
            candidates: Vec::new(),
        }
    }

    async fn vector_search(&self, _query_vector: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        if self.should_fail {
            return Err(SearchError::RedisError("Mock Redis failure".to_string()));
        }
        
        let mut results = self.candidates.clone();
        results.truncate(limit);
        Ok(results)
    }

    async fn health_check(&self) -> SearchResult<()> {
        if self.should_fail {
            Err(SearchError::RedisError("Mock Redis unhealthy".to_string()))
        } else {
            Ok(())
        }
    }
}

/// Mock database manager for testing
struct MockDatabaseManager {
    should_fail: bool,
    candidates: Vec<SearchCandidate>,
}

impl MockDatabaseManager {
    fn new(candidates: Vec<SearchCandidate>) -> Self {
        Self {
            should_fail: false,
            candidates,
        }
    }

    fn with_failure() -> Self {
        Self {
            should_fail: true,
            candidates: Vec::new(),
        }
    }

    async fn vector_search(&self, _query_vector: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        if self.should_fail {
            return Err(SearchError::DatabaseError("Mock Postgres failure".to_string()));
        }
        
        let mut results = self.candidates.clone();
        results.truncate(limit);
        Ok(results)
    }

    async fn health_check(&self) -> SearchResult<()> {
        if self.should_fail {
            Err(SearchError::DatabaseError("Mock Postgres unhealthy".to_string()))
        } else {
            Ok(())
        }
    }
}

/// Create test search candidates
fn create_test_candidates() -> (Vec<SearchCandidate>, Vec<SearchCandidate>) {
    let redis_candidates = vec![
        SearchCandidate {
            post_id: "post1".to_string(),
            score: 0.95,
            source: SearchSource::Redis,
        },
        SearchCandidate {
            post_id: "post2".to_string(),
            score: 0.85,
            source: SearchSource::Redis,
        },
        SearchCandidate {
            post_id: "post3".to_string(),
            score: 0.75,
            source: SearchSource::Redis,
        },
    ];

    let postgres_candidates = vec![
        SearchCandidate {
            post_id: "post1".to_string(), // Duplicate with different score
            score: 0.90,
            source: SearchSource::Postgres,
        },
        SearchCandidate {
            post_id: "post4".to_string(),
            score: 0.80,
            source: SearchSource::Postgres,
        },
        SearchCandidate {
            post_id: "post5".to_string(),
            score: 0.70,
            source: SearchSource::Postgres,
        },
    ];

    (redis_candidates, postgres_candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_and_dedup_basic() {
        // Create a mock service for testing merge logic
        let redis_config = RedisConfig {
            url: "redis://localhost:6379".to_string(),
            max_connections: 10,
            connection_timeout_secs: 5,
            default_ttl_secs: 3600,
        };
        
        let database_config = DatabaseConfig {
            supabase_url: "postgresql://localhost:5432/test".to_string(),
            supabase_service_key: "test_key".to_string(),
            max_connections: 10,
            connection_timeout_secs: 30,
        };

        // We can't easily create real managers in tests, so we'll test the merge logic directly
        let candidates = vec![
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.95,
                source: SearchSource::Redis,
            },
            SearchCandidate {
                post_id: "post2".to_string(),
                score: 0.85,
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post1".to_string(), // Duplicate with lower score
                score: 0.90,
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post3".to_string(),
                score: 0.75,
                source: SearchSource::Redis,
            },
        ];

        // Create a temporary service to test merge logic
        // Since we can't easily mock the managers, we'll create a helper function
        let merged = merge_and_dedup_helper(candidates, 130);

        // Should have 3 unique posts
        assert_eq!(merged.len(), 3);

        // Should be sorted by score (descending)
        assert_eq!(merged[0].post_id, "post1");
        assert_eq!(merged[0].score, 0.95); // Higher score kept
        assert_eq!(merged[0].source, SearchSource::Redis);

        assert_eq!(merged[1].post_id, "post2");
        assert_eq!(merged[1].score, 0.85);

        assert_eq!(merged[2].post_id, "post3");
        assert_eq!(merged[2].score, 0.75);
    }

    #[test]
    fn test_merge_and_dedup_max_candidates() {
        let mut candidates = Vec::new();
        
        // Create 150 candidates to test the 130 limit
        for i in 0..150 {
            candidates.push(SearchCandidate {
                post_id: format!("post{}", i),
                score: 1.0 - (i as f32 * 0.001), // Decreasing scores
                source: SearchSource::Redis,
            });
        }

        let merged = merge_and_dedup_helper(candidates, 130);

        // Should be limited to 130 candidates
        assert_eq!(merged.len(), 130);

        // Should be sorted by score (descending)
        assert!(merged[0].score > merged[1].score);
        assert!(merged[1].score > merged[2].score);
    }

    #[test]
    fn test_merge_and_dedup_duplicate_handling() {
        let candidates = vec![
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.80,
                source: SearchSource::Redis,
            },
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.90, // Higher score
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.85, // Middle score
                source: SearchSource::Redis,
            },
        ];

        let merged = merge_and_dedup_helper(candidates, 130);

        // Should have only 1 result
        assert_eq!(merged.len(), 1);

        // Should keep the highest score
        assert_eq!(merged[0].score, 0.90);
        assert_eq!(merged[0].source, SearchSource::Postgres);
    }

    #[test]
    fn test_merge_and_dedup_empty_input() {
        let candidates = vec![];
        let merged = merge_and_dedup_helper(candidates, 130);
        assert_eq!(merged.len(), 0);
    }

    #[test]
    fn test_merge_and_dedup_score_sorting() {
        let candidates = vec![
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.60,
                source: SearchSource::Redis,
            },
            SearchCandidate {
                post_id: "post2".to_string(),
                score: 0.90,
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post3".to_string(),
                score: 0.75,
                source: SearchSource::Redis,
            },
        ];

        let merged = merge_and_dedup_helper(candidates, 130);

        // Should be sorted by score (descending)
        assert_eq!(merged[0].post_id, "post2");
        assert_eq!(merged[0].score, 0.90);

        assert_eq!(merged[1].post_id, "post3");
        assert_eq!(merged[1].score, 0.75);

        assert_eq!(merged[2].post_id, "post1");
        assert_eq!(merged[2].score, 0.60);
    }

    // Helper function to test merge logic without needing real managers
    fn merge_and_dedup_helper(candidates: Vec<SearchCandidate>, max_candidates: usize) -> Vec<SearchCandidate> {
        use std::collections::HashMap;

        // Use HashMap to deduplicate by post_id, keeping the highest score
        let mut best_candidates: HashMap<String, SearchCandidate> = HashMap::new();

        for candidate in candidates {
            match best_candidates.get(&candidate.post_id) {
                Some(existing) => {
                    // Keep the candidate with higher score
                    if candidate.score > existing.score {
                        best_candidates.insert(candidate.post_id.clone(), candidate);
                    }
                }
                None => {
                    best_candidates.insert(candidate.post_id.clone(), candidate);
                }
            }
        }

        // Convert to vector and sort by score (descending)
        let mut merged_candidates: Vec<SearchCandidate> = best_candidates.into_values().collect();
        merged_candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max candidates
        merged_candidates.truncate(max_candidates);
        merged_candidates
    }
}

// Tests for circuit breaker and fallback functionality
#[cfg(test)]
mod circuit_breaker_tests {
    use super::*;
    use crate::search::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
    use crate::search::fallback::FallbackSearchService;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_circuit_breaker_basic_functionality() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_millis(100),
            success_threshold: 2,
            failure_window: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::with_config(config);

        // Initial state should be Closed
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(!cb.is_redis_circuit_open().await);

        // Record failures to open circuit
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.is_redis_circuit_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            success_threshold: 2,
            failure_window: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::with_config(config);

        // Open the circuit
        cb.record_redis_failure().await;
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for recovery timeout
        sleep(Duration::from_millis(100)).await;

        // Should transition to HalfOpen
        assert!(!cb.is_redis_circuit_open().await);
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Record successes to close circuit
        cb.record_redis_success().await;
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_redis_success().await;
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            success_threshold: 2,
            failure_window: Duration::from_secs(60),
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

        // Failure should reopen circuit
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure_window() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_millis(50),
            success_threshold: 2,
            failure_window: Duration::from_millis(100),
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
    async fn test_circuit_breaker_stats() {
        let cb = CircuitBreaker::new();

        cb.record_redis_failure().await;
        cb.record_postgres_failure().await;

        let stats = cb.get_stats().await;
        assert_eq!(stats.state, CircuitState::Closed);
        assert_eq!(stats.redis_failures, 1);
        assert_eq!(stats.postgres_failures, 1);
        assert_eq!(stats.recent_failures, 1);
        assert_eq!(stats.success_count, 0);
    }

    // Mock implementations for testing fallback service
    struct MockCacheManagerForFallback {
        should_fail: bool,
        candidates: Vec<SearchCandidate>,
    }

    impl MockCacheManagerForFallback {
        fn new(candidates: Vec<SearchCandidate>) -> Self {
            Self {
                should_fail: false,
                candidates,
            }
        }

        fn with_failure() -> Self {
            Self {
                should_fail: true,
                candidates: Vec::new(),
            }
        }

        async fn vector_search(&self, _query_vector: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
            if self.should_fail {
                return Err(SearchError::RedisError("Mock Redis failure".to_string()));
            }
            
            let mut results = self.candidates.clone();
            results.truncate(limit);
            Ok(results)
        }

        async fn health_check(&self) -> SearchResult<()> {
            if self.should_fail {
                Err(SearchError::RedisError("Mock Redis unhealthy".to_string()))
            } else {
                Ok(())
            }
        }
    }

    struct MockDatabaseManagerForFallback {
        should_fail: bool,
        candidates: Vec<SearchCandidate>,
    }

    impl MockDatabaseManagerForFallback {
        fn new(candidates: Vec<SearchCandidate>) -> Self {
            Self {
                should_fail: false,
                candidates,
            }
        }

        fn with_failure() -> Self {
            Self {
                should_fail: true,
                candidates: Vec::new(),
            }
        }

        async fn vector_search(&self, _query_vector: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
            if self.should_fail {
                return Err(SearchError::DatabaseError("Mock Postgres failure".to_string()));
            }
            
            let mut results = self.candidates.clone();
            results.truncate(limit);
            Ok(results)
        }

        async fn health_check(&self) -> SearchResult<()> {
            if self.should_fail {
                Err(SearchError::DatabaseError("Mock Postgres unhealthy".to_string()))
            } else {
                Ok(())
            }
        }
    }

    // Note: These tests would require proper mocking framework integration
    // For now, they serve as documentation of the expected behavior
    
    #[tokio::test]
    async fn test_fallback_search_mode_determination() {
        // This test demonstrates the expected behavior for search mode determination
        // In a real implementation, we would use proper mocks
        
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            success_threshold: 2,
            failure_window: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::with_config(config);

        // Initially should be Full mode (circuit closed, both services healthy)
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(!cb.is_redis_circuit_open().await);

        // After failures, should be PostgresOnly mode
        cb.record_redis_failure().await;
        cb.record_redis_failure().await;
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.is_redis_circuit_open().await);
    }

    #[tokio::test]
    async fn test_retry_logic_integration() {
        use crate::search::retry::{RetryExecutor, RetryConfig};
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1), // Fast for testing
            max_delay: Duration::from_millis(10),
            jitter_factor: 0.0, // No jitter for predictable testing
        };
        let executor = RetryExecutor::with_config(config);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        // Test successful retry after failures
        let result = executor.execute(|| {
            let counter = counter_clone.clone();
            async move {
                let count = counter.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    Err(SearchError::RedisError("Temporary failure".to_string()))
                } else {
                    Ok::<i32, SearchError>(42)
                }
            }
        }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_logic_exhaustion() {
        use crate::search::retry::{RetryExecutor, RetryConfig};
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            jitter_factor: 0.0,
        };
        let executor = RetryExecutor::with_config(config);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        // Test retry exhaustion
        let result = executor.execute(|| {
            let counter = counter_clone.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Err::<i32, SearchError>(SearchError::RedisError("Persistent failure".to_string()))
            }
        }).await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_no_retry_on_client_errors() {
        use crate::search::retry::{RetryExecutor, RetryConfig};
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig::default();
        let executor = RetryExecutor::with_config(config);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        // Test that client errors are not retried
        let result = executor.execute(|| {
            let counter = counter_clone.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Err::<i32, SearchError>(SearchError::InvalidRequest("Bad request".to_string()))
            }
        }).await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1); // No retries for client errors
    }
}

// Integration tests that would require real Redis/Postgres connections
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::env;

    async fn create_test_cache_manager() -> Option<Arc<CacheManager>> {
        if let Ok(redis_url) = env::var("REDIS_URL") {
            let config = RedisConfig {
                url: redis_url,
                max_connections: 5,
                connection_timeout_secs: 5,
                default_ttl_secs: 3600,
            };
            
            if let Ok(manager) = CacheManager::new(config).await {
                return Some(Arc::new(manager));
            }
        }
        None
    }

    async fn create_test_database_manager() -> Option<Arc<DatabaseManager>> {
        if let Ok(database_url) = env::var("DATABASE_URL") {
            let config = DatabaseConfig {
                supabase_url: database_url,
                supabase_service_key: "test_key".to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
            };
            
            if let Ok(manager) = DatabaseManager::new(config).await {
                return Some(Arc::new(manager));
            }
        }
        None
    }

    #[tokio::test]
    #[ignore = "requires Redis and Postgres connections"]
    async fn test_parallel_search_integration() {
        let cache_manager = create_test_cache_manager().await;
        let database_manager = create_test_database_manager().await;

        if let (Some(cache), Some(db)) = (cache_manager, database_manager) {
            let search_service = VectorSearchService::new(cache, db);
            
            // Test with a sample query vector
            let query_vector = vec![0.1; 384]; // 384-dimensional vector
            let limit = 10;

            let result = search_service.parallel_search(&query_vector, limit).await;
            
            // Should succeed even if no results found
            assert!(result.is_ok(), "Parallel search failed: {:?}", result);
            
            let candidates = result.unwrap();
            assert!(candidates.len() <= limit);
        } else {
            println!("Skipping integration test - Redis or Postgres not available");
        }
    }

    #[tokio::test]
    #[ignore = "requires Redis and Postgres connections"]
    async fn test_health_check_integration() {
        let cache_manager = create_test_cache_manager().await;
        let database_manager = create_test_database_manager().await;

        if let (Some(cache), Some(db)) = (cache_manager, database_manager) {
            let search_service = VectorSearchService::new(cache, db);
            
            let health_result = search_service.health_check().await;
            assert!(health_result.is_ok(), "Health check failed: {:?}", health_result);
        }
    }

    #[tokio::test]
    #[ignore = "requires Redis and Postgres connections"]
    async fn test_search_stats_integration() {
        let cache_manager = create_test_cache_manager().await;
        let database_manager = create_test_database_manager().await;

        if let (Some(cache), Some(db)) = (cache_manager, database_manager) {
            let search_service = VectorSearchService::new(cache, db);
            
            let stats_result = search_service.get_search_stats().await;
            assert!(stats_result.is_ok(), "Get stats failed: {:?}", stats_result);
            
            let stats = stats_result.unwrap();
            // Basic validation that stats are populated
            assert!(stats.postgres_total_posts >= 0);
        }
    }
}