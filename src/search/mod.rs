/// Vector search module
/// 
/// This module implements parallel vector search coordination across Redis and Postgres
/// with result merging, deduplication, and graceful failure handling.

pub mod circuit_breaker;
pub mod retry;
pub mod fallback;

#[cfg(test)]
mod tests;

// Re-export main components
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitState};
pub use retry::{RetryExecutor, RetryConfig, RetryStrategy};
pub use fallback::{FallbackSearchService, FallbackHealthStatus};

use crate::cache::CacheManager;
use crate::database::DatabaseManager;
use crate::error::{SearchError, SearchResult};
use crate::types::{SearchCandidate, SearchSource};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, warn};

/// Vector search service that coordinates parallel searches across Redis and Postgres
pub struct VectorSearchService {
    /// Cache manager for Redis operations
    cache_manager: Arc<CacheManager>,
    /// Database manager for Postgres operations
    database_manager: Arc<DatabaseManager>,
    /// Maximum number of candidates to return after merging
    max_candidates: usize,
}

impl VectorSearchService {
    /// Create a new vector search service
    pub fn new(
        cache_manager: Arc<CacheManager>,
        database_manager: Arc<DatabaseManager>,
    ) -> Self {
        VectorSearchService {
            cache_manager,
            database_manager,
            max_candidates: 130, // As per requirements
        }
    }

    /// Perform parallel vector search across Redis and Postgres
    /// 
    /// This method queries both Redis and Postgres simultaneously, then merges
    /// and deduplicates the results. It handles partial failures gracefully.
    pub async fn parallel_search(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Starting parallel vector search with limit: {}", limit);

        // Launch both searches in parallel
        let (redis_result, postgres_result) = tokio::join!(
            self.redis_vector_search_with_timeout(query_vector, 100),
            self.postgres_vector_search_with_timeout(query_vector, 100)
        );

        // Collect successful results
        let mut all_candidates = Vec::new();
        let mut redis_failed = false;
        let mut postgres_failed = false;

        match redis_result {
            Ok(candidates) => {
                debug!("Redis search returned {} candidates", candidates.len());
                all_candidates.extend(candidates);
            }
            Err(e) => {
                warn!("Redis search failed: {}", e);
                redis_failed = true;
            }
        }

        match postgres_result {
            Ok(candidates) => {
                debug!("Postgres search returned {} candidates", candidates.len());
                all_candidates.extend(candidates);
            }
            Err(e) => {
                warn!("Postgres search failed: {}", e);
                postgres_failed = true;
            }
        }

        // Check if both searches failed
        if redis_failed && postgres_failed {
            return Err(SearchError::Internal(
                "Both Redis and Postgres searches failed".to_string()
            ));
        }

        // If only one source failed, log warning but continue
        if redis_failed {
            warn!("Continuing with Postgres-only results due to Redis failure");
        } else if postgres_failed {
            warn!("Continuing with Redis-only results due to Postgres failure");
        }

        // Merge and deduplicate results
        let merged_candidates = self.merge_and_dedup(all_candidates);
        
        // Limit to requested number of results
        let final_candidates: Vec<SearchCandidate> = merged_candidates
            .into_iter()
            .take(limit)
            .collect();

        info!(
            "Parallel search completed: {} final candidates (Redis: {}, Postgres: {})",
            final_candidates.len(),
            !redis_failed,
            !postgres_failed
        );

        Ok(final_candidates)
    }

    /// Search Redis vector store with timeout
    async fn redis_vector_search_with_timeout(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<SearchCandidate>> {
        let search_timeout = Duration::from_millis(400); // Reasonable timeout for Redis
        
        timeout(search_timeout, self.redis_vector_search(query_vector, limit))
            .await
            .map_err(|_| SearchError::RedisError("Redis search timeout".to_string()))?
    }

    /// Search Postgres with pgvector with timeout
    async fn postgres_vector_search_with_timeout(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<SearchCandidate>> {
        let search_timeout = Duration::from_millis(500); // 500ms as per requirements
        
        timeout(search_timeout, self.postgres_vector_search(query_vector, limit))
            .await
            .map_err(|_| SearchError::DatabaseError("Postgres search timeout".to_string()))?
    }

    /// Search Redis vector store
    async fn redis_vector_search(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Performing Redis vector search");
        self.cache_manager.vector_search(query_vector, limit).await
    }

    /// Search Postgres with pgvector
    async fn postgres_vector_search(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Performing Postgres vector search");
        self.database_manager.vector_search(query_vector, limit).await
    }

    /// Merge and deduplicate search candidates
    /// 
    /// This method combines results from Redis and Postgres, removes duplicates
    /// by post_id, and keeps the result with the higher score for each post.
    /// Results are sorted by cosine similarity score in descending order.
    fn merge_and_dedup(&self, candidates: Vec<SearchCandidate>) -> Vec<SearchCandidate> {
        debug!("Merging and deduplicating {} candidates", candidates.len());

        // Use HashMap to deduplicate by post_id, keeping the highest score
        let mut best_candidates: HashMap<String, SearchCandidate> = HashMap::new();

        for candidate in candidates {
            match best_candidates.get(&candidate.post_id) {
                Some(existing) => {
                    // Keep the candidate with higher score
                    if candidate.score > existing.score {
                        debug!(
                            "Replacing candidate {} (score: {:.4} -> {:.4}, source: {:?} -> {:?})",
                            candidate.post_id, existing.score, candidate.score, existing.source, candidate.source
                        );
                        best_candidates.insert(candidate.post_id.clone(), candidate);
                    } else {
                        debug!(
                            "Keeping existing candidate {} (score: {:.4} vs {:.4})",
                            candidate.post_id, existing.score, candidate.score
                        );
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

        // Limit to max candidates as per requirements
        merged_candidates.truncate(self.max_candidates);

        debug!(
            "Merge complete: {} unique candidates (limited to {})",
            merged_candidates.len(),
            self.max_candidates
        );

        merged_candidates
    }

    /// Get search statistics for monitoring
    pub async fn get_search_stats(&self) -> SearchResult<SearchStats> {
        let redis_stats = self.cache_manager.get_redis_stats().await?;
        let postgres_stats = self.database_manager.get_stats().await?;

        Ok(SearchStats {
            redis_connected: self.cache_manager.health_check().await.is_ok(),
            postgres_connected: self.database_manager.health_check().await.is_ok(),
            redis_memory_usage: redis_stats.used_memory_bytes,
            postgres_active_connections: postgres_stats.active_connections,
            postgres_total_posts: postgres_stats.total_posts,
        })
    }

    /// Perform health check on both search backends
    pub async fn health_check(&self) -> SearchResult<()> {
        let (redis_health, postgres_health): (SearchResult<()>, SearchResult<()>) = tokio::join!(
            self.cache_manager.health_check(),
            self.database_manager.health_check()
        );

        match (redis_health, postgres_health) {
            (Ok(_), Ok(_)) => {
                debug!("Both Redis and Postgres are healthy");
                Ok(())
            }
            (Err(redis_err), Ok(_)) => {
                warn!("Redis unhealthy but Postgres OK: {}", redis_err);
                Ok(()) // Can continue with Postgres only
            }
            (Ok(_), Err(postgres_err)) => {
                warn!("Postgres unhealthy but Redis OK: {}", postgres_err);
                Ok(()) // Can continue with Redis only
            }
            (Err(redis_err), Err(postgres_err)) => {
                error!("Both Redis and Postgres are unhealthy: Redis: {}, Postgres: {}", redis_err, postgres_err);
                Err(SearchError::Internal("All search backends unavailable".to_string()))
            }
        }
    }
}

/// Search statistics for monitoring
#[derive(Debug)]
pub struct SearchStats {
    pub redis_connected: bool,
    pub postgres_connected: bool,
    pub redis_memory_usage: u64,
    pub postgres_active_connections: u32,
    pub postgres_total_posts: u64,
}