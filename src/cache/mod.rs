/// Caching module
/// 
/// This module implements the three-tier caching strategy:
/// - Vector cache (permanent LRU)
/// - Top-k cache (60s TTL)
/// - Metadata cache (24h TTL)

mod redis_client;

#[cfg(test)]
mod tests;

use crate::config::RedisConfig;
use crate::error::{SearchError, SearchResult};
use crate::types::{CachedResult, PostMetadata, SearchCandidate};
use chrono::{DateTime, Utc};
use farmhash;
use redis_client::RedisClient;
use std::sync::Arc;
use tracing::{debug, info};

pub use redis_client::{RedisStats};

/// Cache manager for the three-tier caching strategy
pub struct CacheManager {
    /// Redis client for all cache operations
    redis_client: Arc<RedisClient>,
}

impl CacheManager {
    /// Create a new cache manager with Redis connection
    pub async fn new(redis_config: RedisConfig) -> SearchResult<Self> {
        info!("Initializing cache manager");
        
        let redis_client = RedisClient::new(redis_config).await?;
        
        // Perform health check
        redis_client.health_check().await?;
        
        info!("Cache manager initialized successfully");
        
        Ok(CacheManager {
            redis_client: Arc::new(redis_client),
        })
    }

    /// Get cached search results by query hash
    pub async fn get_top_k_cache(&self, query_hash: u64) -> SearchResult<Option<Vec<CachedResult>>> {
        self.redis_client.get_top_k_cache(query_hash).await
    }

    /// Store search results in top-k cache with 60s TTL
    pub async fn set_top_k_cache(
        &self,
        query_hash: u64,
        results: &[CachedResult],
    ) -> SearchResult<()> {
        self.redis_client.set_top_k_cache(query_hash, results).await
    }

    /// Get vector embedding from cache
    pub async fn get_vector_cache(&self, post_id: &str) -> SearchResult<Option<Vec<f32>>> {
        self.redis_client.get_vector(post_id).await
    }

    /// Store vector embedding in cache
    pub async fn set_vector_cache(&self, post_id: &str, embedding: &[f32]) -> SearchResult<()> {
        self.redis_client.set_vector(post_id, embedding).await
    }

    /// Get post metadata from cache
    pub async fn get_metadata_cache(&self, post_id: &str) -> SearchResult<Option<PostMetadata>> {
        self.redis_client.get_metadata_cache(post_id).await
    }

    /// Store post metadata in cache with 24h TTL
    pub async fn set_metadata_cache(
        &self,
        post_id: &str,
        metadata: &PostMetadata,
    ) -> SearchResult<()> {
        self.redis_client.set_metadata_cache(post_id, metadata).await
    }

    /// Generate cache key hash for query using farmhash64
    pub fn generate_query_hash(&self, query: &str) -> u64 {
        // Normalize query: lowercase, trim whitespace, remove extra spaces
        let normalized = query
            .to_lowercase()
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        
        debug!("Generating hash for normalized query: '{}'", normalized);
        farmhash::hash64(normalized.as_bytes())
    }

    /// Perform vector similarity search using Redis
    pub async fn vector_search(&self, query_embedding: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        self.redis_client.vector_search(query_embedding, limit).await
    }

    /// Invalidate cache entries for GDPR compliance
    pub async fn invalidate_post_data(&self, post_id: &str) -> SearchResult<()> {
        self.redis_client.delete_post_data(post_id).await
    }

    /// Get Redis connection statistics
    pub async fn get_redis_stats(&self) -> SearchResult<RedisStats> {
        self.redis_client.get_stats().await
    }

    /// Check Redis connection health
    pub async fn health_check(&self) -> SearchResult<()> {
        self.redis_client.health_check().await
    }
}