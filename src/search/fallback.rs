/// Fallback service with circuit breaker and graceful degradation
/// 
/// This module implements the main search service with circuit breaker integration,
/// automatic fallback to Postgres-only search, and graceful degradation modes.

use crate::cache::CacheManager;
use crate::database::DatabaseManager;
use crate::error::{SearchError, SearchResult};
use crate::types::{SearchCandidate, SearchMode, SearchSource};
use crate::search::circuit_breaker::{CircuitBreaker, CircuitBreakerStats};
use crate::search::retry::{RetryExecutor, RetryConfig, RetryStrategy};
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, warn};

/// Search service with circuit breaker and fallback logic
pub struct FallbackSearchService {
    /// Cache manager for Redis operations
    cache_manager: Arc<CacheManager>,
    /// Database manager for Postgres operations
    database_manager: Arc<DatabaseManager>,
    /// Circuit breaker for failure tracking
    circuit_breaker: Arc<CircuitBreaker>,
    /// Retry executor for transient failures
    retry_executor: RetryExecutor,
    /// Maximum number of candidates after merging
    max_candidates: usize,
}

impl FallbackSearchService {
    /// Create a new fallback search service
    pub fn new(
        cache_manager: Arc<CacheManager>,
        database_manager: Arc<DatabaseManager>,
    ) -> Self {
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let retry_config = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(100), // 100ms, 200ms, 400ms
            max_delay: Duration::from_millis(400),
            jitter_factor: 0.1,
        };
        let retry_executor = RetryExecutor::with_config(retry_config);

        Self {
            cache_manager,
            database_manager,
            circuit_breaker,
            retry_executor,
            max_candidates: 130,
        }
    }

    /// Create a new service with custom circuit breaker
    pub fn with_circuit_breaker(
        cache_manager: Arc<CacheManager>,
        database_manager: Arc<DatabaseManager>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        let retry_config = RetryConfig::default();
        let retry_executor = RetryExecutor::with_config(retry_config);

        Self {
            cache_manager,
            database_manager,
            circuit_breaker,
            retry_executor,
            max_candidates: 130,
        }
    }

    /// Perform search with automatic fallback and circuit breaker logic
    pub async fn search_with_fallback(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<(Vec<SearchCandidate>, SearchMode)> {
        debug!("Starting search with fallback logic, limit: {}", limit);

        // Determine search mode based on circuit breaker state
        let search_mode = self.determine_search_mode().await;
        debug!("Determined search mode: {:?}", search_mode);

        match search_mode {
            SearchMode::Full => {
                self.full_search_with_retry(query_vector, limit).await
            }
            SearchMode::PostgresOnly => {
                self.postgres_only_search_with_retry(query_vector, limit).await
            }
            SearchMode::CacheOnly => {
                self.cache_only_search_with_retry(query_vector, limit).await
            }
            SearchMode::Degraded => {
                // For now, degraded mode is same as full but without reranking
                // Reranking logic will be implemented in a later task
                self.full_search_with_retry(query_vector, limit).await
            }
        }
    }

    /// Determine the appropriate search mode based on system state
    async fn determine_search_mode(&self) -> SearchMode {
        // Check if Redis circuit is open
        if self.circuit_breaker.is_redis_circuit_open().await {
            debug!("Redis circuit is open, using PostgresOnly mode");
            return SearchMode::PostgresOnly;
        }

        // Check if Postgres is available (simplified check)
        match self.database_manager.health_check().await {
            Ok(_) => {
                // Both Redis and Postgres available
                SearchMode::Full
            }
            Err(_) => {
                warn!("Postgres unavailable, falling back to CacheOnly mode");
                SearchMode::CacheOnly
            }
        }
    }

    /// Perform full search (Redis + Postgres) with retry logic
    async fn full_search_with_retry(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<(Vec<SearchCandidate>, SearchMode)> {
        let circuit_breaker = self.circuit_breaker.clone();
        let cache_manager = self.cache_manager.clone();
        let database_manager = self.database_manager.clone();
        let query_vector = query_vector.to_vec();

        let result = self.retry_executor.execute(|| {
            let circuit_breaker = circuit_breaker.clone();
            let cache_manager = cache_manager.clone();
            let database_manager = database_manager.clone();
            let query_vector = query_vector.clone();

            async move {
                self.execute_full_search(&query_vector, limit, &cache_manager, &database_manager, &circuit_breaker).await
            }
        }).await;

        match result {
            Ok(candidates) => Ok((candidates, SearchMode::Full)),
            Err(e) => {
                error!("Full search failed after retries: {}", e);
                // Try fallback to Postgres-only
                warn!("Attempting fallback to Postgres-only search");
                self.postgres_only_search_with_retry(&query_vector, limit).await
            }
        }
    }

    /// Execute full search (Redis + Postgres in parallel)
    async fn execute_full_search(
        &self,
        query_vector: &[f32],
        limit: usize,
        cache_manager: &CacheManager,
        database_manager: &DatabaseManager,
        circuit_breaker: &CircuitBreaker,
    ) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Executing full parallel search");

        // Launch both searches in parallel
        let (redis_result, postgres_result) = tokio::join!(
            self.redis_search_with_timeout(query_vector, 100, cache_manager),
            self.postgres_search_with_timeout(query_vector, 100, database_manager)
        );

        // Process Redis result
        let mut all_candidates = Vec::new();
        match redis_result {
            Ok(candidates) => {
                debug!("Redis search succeeded: {} candidates", candidates.len());
                circuit_breaker.record_redis_success().await;
                all_candidates.extend(candidates);
            }
            Err(e) => {
                warn!("Redis search failed: {}", e);
                circuit_breaker.record_redis_failure().await;
                
                // If Redis fails, we can still continue with Postgres results
                if e.is_redis_error() {
                    debug!("Continuing with Postgres-only results due to Redis failure");
                } else {
                    return Err(e);
                }
            }
        }

        // Process Postgres result
        match postgres_result {
            Ok(candidates) => {
                debug!("Postgres search succeeded: {} candidates", candidates.len());
                circuit_breaker.record_postgres_success().await;
                all_candidates.extend(candidates);
            }
            Err(e) => {
                warn!("Postgres search failed: {}", e);
                circuit_breaker.record_postgres_failure().await;
                
                // If we have Redis results, we can continue
                if all_candidates.is_empty() {
                    return Err(e);
                } else {
                    warn!("Continuing with Redis-only results due to Postgres failure");
                }
            }
        }

        if all_candidates.is_empty() {
            return Err(SearchError::Internal("No search results from any source".to_string()));
        }

        // Merge and deduplicate results
        let merged_candidates = self.merge_and_dedup(all_candidates);
        let final_candidates: Vec<SearchCandidate> = merged_candidates
            .into_iter()
            .take(limit)
            .collect();

        debug!("Full search completed: {} final candidates", final_candidates.len());
        Ok(final_candidates)
    }

    /// Perform Postgres-only search with retry logic
    async fn postgres_only_search_with_retry(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<(Vec<SearchCandidate>, SearchMode)> {
        let database_manager = self.database_manager.clone();
        let circuit_breaker = self.circuit_breaker.clone();
        let query_vector = query_vector.to_vec();

        let result = self.retry_executor.execute(|| {
            let database_manager = database_manager.clone();
            let circuit_breaker = circuit_breaker.clone();
            let query_vector = query_vector.clone();

            async move {
                self.execute_postgres_only_search(&query_vector, limit, &database_manager, &circuit_breaker).await
            }
        }).await;

        match result {
            Ok(candidates) => Ok((candidates, SearchMode::PostgresOnly)),
            Err(e) => {
                error!("Postgres-only search failed after retries: {}", e);
                Err(e)
            }
        }
    }

    /// Execute Postgres-only search
    async fn execute_postgres_only_search(
        &self,
        query_vector: &[f32],
        limit: usize,
        database_manager: &DatabaseManager,
        circuit_breaker: &CircuitBreaker,
    ) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Executing Postgres-only search");

        match self.postgres_search_with_timeout(query_vector, limit, database_manager).await {
            Ok(candidates) => {
                debug!("Postgres-only search succeeded: {} candidates", candidates.len());
                circuit_breaker.record_postgres_success().await;
                Ok(candidates)
            }
            Err(e) => {
                warn!("Postgres-only search failed: {}", e);
                circuit_breaker.record_postgres_failure().await;
                Err(e)
            }
        }
    }

    /// Perform cache-only search with retry logic
    async fn cache_only_search_with_retry(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<(Vec<SearchCandidate>, SearchMode)> {
        let cache_manager = self.cache_manager.clone();
        let circuit_breaker = self.circuit_breaker.clone();
        let query_vector = query_vector.to_vec();

        let result = self.retry_executor.execute(|| {
            let cache_manager = cache_manager.clone();
            let circuit_breaker = circuit_breaker.clone();
            let query_vector = query_vector.clone();

            async move {
                self.execute_cache_only_search(&query_vector, limit, &cache_manager, &circuit_breaker).await
            }
        }).await;

        match result {
            Ok(candidates) => Ok((candidates, SearchMode::CacheOnly)),
            Err(e) => {
                error!("Cache-only search failed after retries: {}", e);
                Err(e)
            }
        }
    }

    /// Execute cache-only search
    async fn execute_cache_only_search(
        &self,
        query_vector: &[f32],
        limit: usize,
        cache_manager: &CacheManager,
        circuit_breaker: &CircuitBreaker,
    ) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Executing cache-only search");

        match self.redis_search_with_timeout(query_vector, limit, cache_manager).await {
            Ok(candidates) => {
                debug!("Cache-only search succeeded: {} candidates", candidates.len());
                circuit_breaker.record_redis_success().await;
                Ok(candidates)
            }
            Err(e) => {
                warn!("Cache-only search failed: {}", e);
                circuit_breaker.record_redis_failure().await;
                Err(e)
            }
        }
    }

    /// Redis search with timeout
    async fn redis_search_with_timeout(
        &self,
        query_vector: &[f32],
        limit: usize,
        cache_manager: &CacheManager,
    ) -> SearchResult<Vec<SearchCandidate>> {
        let search_timeout = Duration::from_millis(400);
        
        timeout(search_timeout, cache_manager.vector_search(query_vector, limit))
            .await
            .map_err(|_| SearchError::RedisError("Redis search timeout".to_string()))?
    }

    /// Postgres search with timeout
    async fn postgres_search_with_timeout(
        &self,
        query_vector: &[f32],
        limit: usize,
        database_manager: &DatabaseManager,
    ) -> SearchResult<Vec<SearchCandidate>> {
        let search_timeout = Duration::from_millis(500);
        
        timeout(search_timeout, database_manager.vector_search(query_vector, limit))
            .await
            .map_err(|_| SearchError::DatabaseError("Postgres search timeout".to_string()))?
    }

    /// Merge and deduplicate search candidates
    fn merge_and_dedup(&self, candidates: Vec<SearchCandidate>) -> Vec<SearchCandidate> {
        use std::collections::HashMap;

        debug!("Merging and deduplicating {} candidates", candidates.len());

        let mut best_candidates: HashMap<String, SearchCandidate> = HashMap::new();

        for candidate in candidates {
            match best_candidates.get(&candidate.post_id) {
                Some(existing) => {
                    if candidate.score > existing.score {
                        debug!(
                            "Replacing candidate {} (score: {:.4} -> {:.4})",
                            candidate.post_id, existing.score, candidate.score
                        );
                        best_candidates.insert(candidate.post_id.clone(), candidate);
                    }
                }
                None => {
                    best_candidates.insert(candidate.post_id.clone(), candidate);
                }
            }
        }

        let mut merged_candidates: Vec<SearchCandidate> = best_candidates.into_values().collect();
        merged_candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        merged_candidates.truncate(self.max_candidates);
        debug!("Merge complete: {} unique candidates", merged_candidates.len());

        merged_candidates
    }

    /// Get circuit breaker statistics
    pub async fn get_circuit_breaker_stats(&self) -> CircuitBreakerStats {
        self.circuit_breaker.get_stats().await
    }

    /// Get current search mode
    pub async fn get_current_search_mode(&self) -> SearchMode {
        self.determine_search_mode().await
    }

    /// Perform health check
    pub async fn health_check(&self) -> SearchResult<FallbackHealthStatus> {
        let (redis_health, postgres_health) = tokio::join!(
            self.cache_manager.health_check(),
            self.database_manager.health_check()
        );

        let circuit_stats = self.circuit_breaker.get_stats().await;
        let search_mode = self.determine_search_mode().await;

        Ok(FallbackHealthStatus {
            redis_healthy: redis_health.is_ok(),
            postgres_healthy: postgres_health.is_ok(),
            circuit_breaker_stats: circuit_stats,
            current_search_mode: search_mode,
            redis_error: redis_health.err().map(|e| e.to_string()),
            postgres_error: postgres_health.err().map(|e| e.to_string()),
        })
    }
}

/// Health status for the fallback search service
#[derive(Debug, Clone)]
pub struct FallbackHealthStatus {
    pub redis_healthy: bool,
    pub postgres_healthy: bool,
    pub circuit_breaker_stats: CircuitBreakerStats,
    pub current_search_mode: SearchMode,
    pub redis_error: Option<String>,
    pub postgres_error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::circuit_breaker::CircuitBreakerConfig;
    use std::time::Duration;

    // Mock implementations would go here for testing
    // For now, we'll focus on the integration tests that can be run
    // with the actual cache and database managers

    #[tokio::test]
    async fn test_search_mode_determination() {
        // This test would require mock implementations
        // Will be implemented when we have proper mocking setup
    }

    #[tokio::test]
    async fn test_circuit_breaker_integration() {
        // This test would verify circuit breaker behavior
        // Will be implemented with proper mocking
    }
}