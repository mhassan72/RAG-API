use crate::config::RedisConfig;
use crate::error::{SearchError, SearchResult};
use crate::types::{CachedResult, PostMetadata, SearchCandidate, SearchSource};
use fred::{
    clients::RedisPool,
    interfaces::{ClientLike, KeysInterface},
    types::{Builder, Expiration, RedisConfig as FredRedisConfig, InfoKind},
};
use serde_json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Redis client wrapper with connection pooling and error handling
pub struct RedisClient {
    /// Fred Redis client with connection pooling
    client: RedisPool,
    /// Configuration
    config: RedisConfig,
    /// Cache statistics tracking
    stats: Arc<CacheStatsInternal>,
}

/// Internal cache statistics with atomic counters for thread safety
#[derive(Debug, Default)]
struct CacheStatsInternal {
    // Vector cache statistics
    vector_cache_hits: AtomicU64,
    vector_cache_misses: AtomicU64,
    
    // Top-k cache statistics
    topk_cache_hits: AtomicU64,
    topk_cache_misses: AtomicU64,
    
    // Metadata cache statistics
    metadata_cache_hits: AtomicU64,
    metadata_cache_misses: AtomicU64,
    
    // GDPR deletion statistics
    gdpr_deletions: AtomicU64,
    gdpr_keys_deleted: AtomicU64,
}

impl CacheStatsInternal {
    /// Convert to public CacheStats struct
    fn to_cache_stats(&self) -> CacheStats {
        CacheStats {
            vector_cache_hits: self.vector_cache_hits.load(Ordering::Relaxed),
            vector_cache_misses: self.vector_cache_misses.load(Ordering::Relaxed),
            topk_cache_hits: self.topk_cache_hits.load(Ordering::Relaxed),
            topk_cache_misses: self.topk_cache_misses.load(Ordering::Relaxed),
            metadata_cache_hits: self.metadata_cache_hits.load(Ordering::Relaxed),
            metadata_cache_misses: self.metadata_cache_misses.load(Ordering::Relaxed),
            gdpr_deletions: self.gdpr_deletions.load(Ordering::Relaxed),
            gdpr_keys_deleted: self.gdpr_keys_deleted.load(Ordering::Relaxed),
        }
    }
}

impl RedisClient {
    /// Create a new Redis client with TLS and cluster support
    pub async fn new(config: RedisConfig) -> SearchResult<Self> {
        info!("Initializing Redis client with URL: {}", &config.url);

        // Parse Redis URL to determine if TLS is needed
        let _use_tls = config.url.starts_with("rediss://");
        
        // Create Redis config
        let redis_config = FredRedisConfig::from_url(&config.url)
            .map_err(|e| SearchError::RedisError(format!("Invalid Redis URL: {}", e)))?;

        // Create the Redis client with proper configuration
        let timeout_secs = config.connection_timeout_secs;
        let client = Builder::from_config(redis_config)
            .with_connection_config(|conn_config| {
                conn_config.connection_timeout = Duration::from_secs(timeout_secs);
            })
            .with_performance_config(|perf_config| {
                perf_config.auto_pipeline = true;
                perf_config.default_command_timeout = Duration::from_secs(timeout_secs);
            })
            .build_pool(config.max_connections as usize)
            .map_err(|e| SearchError::RedisError(format!("Failed to create Redis pool: {}", e)))?;

        // Connect to Redis
        client
            .connect()
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to connect to Redis: {}", e)))?;

        // Wait for connection to be ready
        client
            .wait_for_connect()
            .await
            .map_err(|e| SearchError::RedisError(format!("Redis connection timeout: {}", e)))?;

        info!("Redis client connected successfully");

        Ok(RedisClient { 
            client, 
            config,
            stats: Arc::new(CacheStatsInternal::default()),
        })
    }

    /// Store vector embedding in Redis with permanent storage
    pub async fn set_vector(&self, post_id: &str, embedding: &[f32]) -> SearchResult<()> {
        let key = format!("search:vec:{}", post_id);
        
        // Serialize embedding as bytes for efficient storage
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        debug!("Storing vector for post_id: {} (size: {} bytes)", post_id, embedding_bytes.len());

        let _: () = self.client
            .set(&key, embedding_bytes, None, None, false)
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to store vector: {}", e)))?;

        Ok(())
    }

    /// Retrieve vector embedding from Redis
    pub async fn get_vector(&self, post_id: &str) -> SearchResult<Option<Vec<f32>>> {
        let key = format!("search:vec:{}", post_id);
        
        debug!("Retrieving vector for post_id: {}", post_id);

        let result: Option<Vec<u8>> = self.client
            .get(&key)
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to get vector: {}", e)))?;

        match result {
            Some(bytes) => {
                // Track cache hit
                self.stats.vector_cache_hits.fetch_add(1, Ordering::Relaxed);
                
                // Deserialize bytes back to f32 vector
                if bytes.len() % 4 != 0 {
                    return Err(SearchError::RedisError(
                        "Invalid vector data: length not divisible by 4".to_string()
                    ));
                }

                let embedding: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                debug!("Retrieved vector for post_id: {} (dimensions: {}) - CACHE HIT", post_id, embedding.len());
                Ok(Some(embedding))
            }
            None => {
                // Track cache miss
                self.stats.vector_cache_misses.fetch_add(1, Ordering::Relaxed);
                debug!("No vector found for post_id: {} - CACHE MISS", post_id);
                Ok(None)
            }
        }
    }

    /// Perform vector similarity search using Redis VSS
    pub async fn vector_search(&self, query_embedding: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Performing Redis vector search with limit: {}", limit);

        // For this implementation, we'll use a simple approach since Redis VSS setup
        // requires specific index configuration. In a real implementation, you would:
        // 1. Create a Redis Search index with vector field
        // 2. Use FT.SEARCH with KNN query
        // 
        // For now, we'll implement a fallback that scans available vectors
        // This is not optimal for production but demonstrates the interface

        // For this implementation, we'll use a simple approach since Redis VSS setup
        // requires specific index configuration. In a real implementation, you would:
        // 1. Create a Redis Search index with vector field using FT.CREATE
        // 2. Use FT.SEARCH with KNN query for efficient vector search
        // 
        // For now, we'll return empty results and log a warning
        warn!("Redis vector search not fully implemented - requires Redis Search module with vector indexing");
        
        // In a production system, this would be:
        // let search_query = format!("*=>[KNN {} @embedding $query_vec]", limit);
        // let results = self.client.ft_search("vector_index", &search_query, query_embedding).await?;
        
        let keys: Vec<String> = Vec::new(); // Placeholder - would come from FT.SEARCH results

        debug!("Found {} vector keys for similarity search", keys.len());

        let mut candidates = Vec::new();

        // Process keys in batches to avoid overwhelming Redis
        for chunk in keys.chunks(50) {
            let mut batch_candidates = Vec::new();
            
            for key in chunk {
                if let Some(post_id) = key.strip_prefix("search:vec:") {
                    if let Ok(Some(embedding)) = self.get_vector(post_id).await {
                        let score = cosine_similarity(query_embedding, &embedding);
                        batch_candidates.push(SearchCandidate {
                            post_id: post_id.to_string(),
                            score,
                            source: SearchSource::Redis,
                        });
                    }
                }
            }
            
            candidates.extend(batch_candidates);
        }

        // Sort by score (descending) and limit results
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);

        debug!("Redis vector search returned {} candidates", candidates.len());
        Ok(candidates)
    }

    /// Store top-k search results in cache with TTL
    pub async fn set_top_k_cache(&self, query_hash: u64, results: &[CachedResult]) -> SearchResult<()> {
        let key = format!("search:topk:{}", query_hash);
        let ttl = 60; // 60 seconds as per requirements

        debug!("Caching top-k results for query_hash: {} (count: {})", query_hash, results.len());

        let serialized = serde_json::to_string(results)
            .map_err(|e| SearchError::CacheError(format!("Failed to serialize results: {}", e)))?;

        let _: () = self.client
            .set(&key, serialized, Some(Expiration::EX(ttl)), None, false)
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to cache top-k results: {}", e)))?;

        Ok(())
    }

    /// Retrieve top-k search results from cache
    pub async fn get_top_k_cache(&self, query_hash: u64) -> SearchResult<Option<Vec<CachedResult>>> {
        let key = format!("search:topk:{}", query_hash);
        
        debug!("Retrieving top-k cache for query_hash: {}", query_hash);

        let result: Option<String> = self.client
            .get(&key)
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to get top-k cache: {}", e)))?;

        match result {
            Some(serialized) => {
                // Track cache hit
                self.stats.topk_cache_hits.fetch_add(1, Ordering::Relaxed);
                
                let results: Vec<CachedResult> = serde_json::from_str(&serialized)
                    .map_err(|e| SearchError::CacheError(format!("Failed to deserialize cached results: {}", e)))?;
                
                debug!("Retrieved {} cached results for query_hash: {} - CACHE HIT", results.len(), query_hash);
                Ok(Some(results))
            }
            None => {
                // Track cache miss
                self.stats.topk_cache_misses.fetch_add(1, Ordering::Relaxed);
                debug!("No cached results found for query_hash: {} - CACHE MISS", query_hash);
                Ok(None)
            }
        }
    }

    /// Store post metadata in cache with 24h TTL
    pub async fn set_metadata_cache(&self, post_id: &str, metadata: &PostMetadata) -> SearchResult<()> {
        let key = format!("search:meta:{}", post_id);
        let ttl = 24 * 60 * 60; // 24 hours

        debug!("Caching metadata for post_id: {}", post_id);

        let serialized = serde_json::to_string(metadata)
            .map_err(|e| SearchError::CacheError(format!("Failed to serialize metadata: {}", e)))?;

        let _: () = self.client
            .set(&key, serialized, Some(Expiration::EX(ttl)), None, false)
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to cache metadata: {}", e)))?;

        Ok(())
    }

    /// Retrieve post metadata from cache
    pub async fn get_metadata_cache(&self, post_id: &str) -> SearchResult<Option<PostMetadata>> {
        let key = format!("search:meta:{}", post_id);
        
        debug!("Retrieving metadata cache for post_id: {}", post_id);

        let result: Option<String> = self.client
            .get(&key)
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to get metadata cache: {}", e)))?;

        match result {
            Some(serialized) => {
                // Track cache hit
                self.stats.metadata_cache_hits.fetch_add(1, Ordering::Relaxed);
                
                let metadata: PostMetadata = serde_json::from_str(&serialized)
                    .map_err(|e| SearchError::CacheError(format!("Failed to deserialize metadata: {}", e)))?;
                
                debug!("Retrieved cached metadata for post_id: {} - CACHE HIT", post_id);
                Ok(Some(metadata))
            }
            None => {
                // Track cache miss
                self.stats.metadata_cache_misses.fetch_add(1, Ordering::Relaxed);
                debug!("No cached metadata found for post_id: {} - CACHE MISS", post_id);
                Ok(None)
            }
        }
    }

    /// Delete post data from all caches (GDPR compliance)
    pub async fn delete_post_data(&self, post_id: &str) -> SearchResult<()> {
        let keys = vec![
            format!("search:vec:{}", post_id),
            format!("search:meta:{}", post_id),
        ];

        debug!("Deleting cached data for post_id: {}", post_id);

        // Use UNLINK for non-blocking deletion
        let deleted_count: i64 = self.client
            .unlink(keys)
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to delete post data: {}", e)))?;

        // Track GDPR deletion statistics
        self.stats.gdpr_deletions.fetch_add(1, Ordering::Relaxed);
        self.stats.gdpr_keys_deleted.fetch_add(deleted_count as u64, Ordering::Relaxed);

        info!("Deleted {} cache entries for post_id: {} (GDPR compliance)", deleted_count, post_id);
        Ok(())
    }

    /// Check Redis connection health
    pub async fn health_check(&self) -> SearchResult<()> {
        let start = std::time::Instant::now();
        
        // Use timeout to prevent hanging
        let ping_result = timeout(
            Duration::from_secs(5),
            self.client.ping::<String>()
        ).await;

        match ping_result {
            Ok(Ok(_)) => {
                let duration = start.elapsed();
                debug!("Redis health check passed in {:?}", duration);
                Ok(())
            }
            Ok(Err(e)) => {
                error!("Redis health check failed: {}", e);
                Err(SearchError::RedisError(format!("Health check failed: {}", e)))
            }
            Err(_) => {
                error!("Redis health check timed out");
                Err(SearchError::RedisError("Health check timed out".to_string()))
            }
        }
    }

    /// Get Redis connection statistics
    pub async fn get_stats(&self) -> SearchResult<RedisStats> {
        // Get basic info from Redis
        let info: String = self.client
            .info(Some(InfoKind::Stats))
            .await
            .map_err(|e| SearchError::RedisError(format!("Failed to get Redis info: {}", e)))?;

        // Parse relevant statistics
        let mut stats = RedisStats::default();
        
        for line in info.lines() {
            if let Some((key, value)) = line.split_once(':') {
                match key {
                    "total_commands_processed" => {
                        stats.total_commands = value.parse().unwrap_or(0);
                    }
                    "total_connections_received" => {
                        stats.total_connections = value.parse().unwrap_or(0);
                    }
                    "connected_clients" => {
                        stats.connected_clients = value.parse().unwrap_or(0);
                    }
                    "used_memory" => {
                        stats.used_memory_bytes = value.parse().unwrap_or(0);
                    }
                    _ => {}
                }
            }
        }

        Ok(stats)
    }

    /// Get cache hit/miss statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.stats.to_cache_stats()
    }

    /// Reset cache statistics (useful for testing)
    pub fn reset_cache_stats(&self) {
        self.stats.vector_cache_hits.store(0, Ordering::Relaxed);
        self.stats.vector_cache_misses.store(0, Ordering::Relaxed);
        self.stats.topk_cache_hits.store(0, Ordering::Relaxed);
        self.stats.topk_cache_misses.store(0, Ordering::Relaxed);
        self.stats.metadata_cache_hits.store(0, Ordering::Relaxed);
        self.stats.metadata_cache_misses.store(0, Ordering::Relaxed);
        self.stats.gdpr_deletions.store(0, Ordering::Relaxed);
        self.stats.gdpr_keys_deleted.store(0, Ordering::Relaxed);
    }
}

/// Redis connection statistics
#[derive(Debug, Default)]
pub struct RedisStats {
    pub total_commands: u64,
    pub total_connections: u64,
    pub connected_clients: u32,
    pub used_memory_bytes: u64,
}

/// Cache statistics for monitoring hit/miss ratios
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    // Vector cache statistics
    pub vector_cache_hits: u64,
    pub vector_cache_misses: u64,
    
    // Top-k cache statistics
    pub topk_cache_hits: u64,
    pub topk_cache_misses: u64,
    
    // Metadata cache statistics
    pub metadata_cache_hits: u64,
    pub metadata_cache_misses: u64,
    
    // GDPR deletion statistics
    pub gdpr_deletions: u64,
    pub gdpr_keys_deleted: u64,
}

impl CacheStats {
    /// Calculate vector cache hit ratio
    pub fn vector_hit_ratio(&self) -> f64 {
        let total = self.vector_cache_hits + self.vector_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.vector_cache_hits as f64 / total as f64
        }
    }
    
    /// Calculate top-k cache hit ratio
    pub fn topk_hit_ratio(&self) -> f64 {
        let total = self.topk_cache_hits + self.topk_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.topk_cache_hits as f64 / total as f64
        }
    }
    
    /// Calculate metadata cache hit ratio
    pub fn metadata_hit_ratio(&self) -> f64 {
        let total = self.metadata_cache_hits + self.metadata_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.metadata_cache_hits as f64 / total as f64
        }
    }
    
    /// Calculate overall cache hit ratio
    pub fn overall_hit_ratio(&self) -> f64 {
        let total_hits = self.vector_cache_hits + self.topk_cache_hits + self.metadata_cache_hits;
        let total_misses = self.vector_cache_misses + self.topk_cache_misses + self.metadata_cache_misses;
        let total = total_hits + total_misses;
        
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!((result - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);

        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}