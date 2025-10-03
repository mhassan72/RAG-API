/// Caching module
/// 
/// This module implements a comprehensive three-tier caching strategy for the RAG Search API:
/// 
/// ## Three-Tier Cache Architecture
/// 
/// ### 1. Vector Cache (Permanent LRU)
/// - **Purpose**: Store post embeddings to avoid recomputation
/// - **Key Pattern**: `search:vec:<post_id>`
/// - **TTL**: Permanent (LRU eviction when memory limit reached)
/// - **Data**: 384-dimensional f32 vectors stored as binary data
/// 
/// ### 2. Top-K Cache (60s TTL)
/// - **Purpose**: Cache complete search results for identical queries
/// - **Key Pattern**: `search:topk:<query_hash>` (farmhash64 of normalized query)
/// - **TTL**: 60 seconds
/// - **Data**: Serialized JSON array of CachedResult structs
/// 
/// ### 3. Metadata Cache (24h TTL)
/// - **Purpose**: Cache post metadata to avoid database lookups
/// - **Key Pattern**: `search:meta:<post_id>`
/// - **TTL**: 24 hours
/// - **Data**: Serialized JSON PostMetadata structs
/// 
/// ## Cache Statistics and Monitoring
/// 
/// The cache system provides comprehensive hit/miss tracking:
/// - Per-cache-tier hit/miss ratios
/// - Overall cache performance metrics
/// - GDPR deletion tracking
/// - Thread-safe atomic counters for concurrent access
/// 
/// ## GDPR Compliance
/// 
/// The cache supports GDPR "right to be forgotten" through:
/// - `invalidate_post_data()` method for complete data deletion
/// - Audit logging of deletion operations
/// - Non-blocking UNLINK operations for performance
/// 
/// ## Query Normalization
/// 
/// Query hashing includes normalization to improve cache hit rates:
/// - Lowercase conversion
/// - Whitespace trimming and normalization
/// - Consistent hash generation using farmhash64
/// 
/// ## Performance Characteristics
/// 
/// - **Vector Cache**: O(1) lookup, permanent storage with LRU eviction
/// - **Top-K Cache**: O(1) lookup, 60s TTL for query result caching
/// - **Metadata Cache**: O(1) lookup, 24h TTL for metadata caching
/// - **Statistics**: Thread-safe atomic operations with minimal overhead

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

pub use redis_client::{RedisStats, CacheStats};

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

    /// Get cache hit/miss statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.redis_client.get_cache_stats()
    }

    /// Reset cache statistics (useful for testing and monitoring)
    pub fn reset_cache_stats(&self) {
        self.redis_client.reset_cache_stats()
    }
}