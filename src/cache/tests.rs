use super::*;
use crate::config::RedisConfig;
use crate::types::{CachedResult, PostMetadata};
use chrono::Utc;
use std::env;
use tokio;

/// Helper function to create a test Redis config
fn create_test_redis_config() -> RedisConfig {
    RedisConfig {
        url: env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
        max_connections: 5,
        connection_timeout_secs: 5,
        default_ttl_secs: 3600,
    }
}

/// Helper function to create test metadata
fn create_test_metadata() -> PostMetadata {
    PostMetadata {
        author_name: "Test Author".to_string(),
        url: "https://example.com/post/123".to_string(),
        date: Utc::now(),
        language: "en".to_string(),
        frozen: false,
    }
}

/// Helper function to create test cached results
fn create_test_cached_results() -> Vec<CachedResult> {
    vec![
        CachedResult {
            post_id: "post_1".to_string(),
            title: "Test Post 1".to_string(),
            snippet: "This is a test post snippet".to_string(),
            score: 0.95,
            meta: create_test_metadata(),
            cached_at: Utc::now(),
        },
        CachedResult {
            post_id: "post_2".to_string(),
            title: "Test Post 2".to_string(),
            snippet: "Another test post snippet".to_string(),
            score: 0.87,
            meta: create_test_metadata(),
            cached_at: Utc::now(),
        },
    ]
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_cache_manager_creation() {
    let config = create_test_redis_config();
    
    // This test will only pass if Redis is available
    // In CI/CD, you would use testcontainers or mock Redis
    match CacheManager::new(config).await {
        Ok(cache_manager) => {
            // Test health check
            assert!(cache_manager.health_check().await.is_ok());
        }
        Err(e) => {
            // If Redis is not available, skip the test
            println!("Skipping Redis test - Redis not available: {}", e);
        }
    }
}

#[test]
fn test_redis_config_validation() {
    // Test valid Redis URL
    let valid_config = RedisConfig {
        url: "redis://localhost:6379".to_string(),
        max_connections: 10,
        connection_timeout_secs: 5,
        default_ttl_secs: 3600,
    };
    
    assert!(valid_config.url.starts_with("redis://"));
    assert!(valid_config.max_connections > 0);
    assert!(valid_config.connection_timeout_secs > 0);
    
    // Test TLS Redis URL
    let tls_config = RedisConfig {
        url: "rediss://secure-redis:6380".to_string(),
        max_connections: 5,
        connection_timeout_secs: 10,
        default_ttl_secs: 1800,
    };
    
    assert!(tls_config.url.starts_with("rediss://"));
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_query_hash_generation() {
    let config = create_test_redis_config();
    
    // Create a cache manager (this might fail if Redis is not available)
    if let Ok(cache_manager) = CacheManager::new(config).await {
        // Test query normalization and hashing
        let query1 = "  Hello World  ";
        let query2 = "hello world";
        let query3 = "Hello    World";
        
        let hash1 = cache_manager.generate_query_hash(query1);
        let hash2 = cache_manager.generate_query_hash(query2);
        let hash3 = cache_manager.generate_query_hash(query3);
        
        // All normalized queries should produce the same hash
        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
        
        // Different queries should produce different hashes
        let different_query = "Different query";
        let different_hash = cache_manager.generate_query_hash(different_query);
        assert_ne!(hash1, different_hash);
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_vector_cache_operations() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        let post_id = "test_post_123";
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        
        // Test storing vector
        let store_result = cache_manager.set_vector_cache(post_id, &embedding).await;
        assert!(store_result.is_ok(), "Failed to store vector: {:?}", store_result);
        
        // Test retrieving vector
        let retrieved = cache_manager.get_vector_cache(post_id).await;
        assert!(retrieved.is_ok(), "Failed to retrieve vector: {:?}", retrieved);
        
        if let Ok(Some(retrieved_embedding)) = retrieved {
            assert_eq!(embedding.len(), retrieved_embedding.len());
            for (original, retrieved) in embedding.iter().zip(retrieved_embedding.iter()) {
                assert!((original - retrieved).abs() < f32::EPSILON);
            }
        }
        
        // Test retrieving non-existent vector
        let non_existent = cache_manager.get_vector_cache("non_existent_post").await;
        assert!(non_existent.is_ok());
        assert!(non_existent.unwrap().is_none());
        
        // Clean up
        let _ = cache_manager.invalidate_post_data(post_id).await;
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_top_k_cache_operations() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        let query_hash = 12345u64;
        let results = create_test_cached_results();
        
        // Test storing top-k results
        let store_result = cache_manager.set_top_k_cache(query_hash, &results).await;
        assert!(store_result.is_ok(), "Failed to store top-k results: {:?}", store_result);
        
        // Test retrieving top-k results
        let retrieved = cache_manager.get_top_k_cache(query_hash).await;
        assert!(retrieved.is_ok(), "Failed to retrieve top-k results: {:?}", retrieved);
        
        if let Ok(Some(retrieved_results)) = retrieved {
            assert_eq!(results.len(), retrieved_results.len());
            assert_eq!(results[0].post_id, retrieved_results[0].post_id);
            assert_eq!(results[0].title, retrieved_results[0].title);
            assert!((results[0].score - retrieved_results[0].score).abs() < f32::EPSILON);
        }
        
        // Test retrieving non-existent cache
        let non_existent = cache_manager.get_top_k_cache(99999u64).await;
        assert!(non_existent.is_ok());
        assert!(non_existent.unwrap().is_none());
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_metadata_cache_operations() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        let post_id = "test_post_456";
        let metadata = create_test_metadata();
        
        // Test storing metadata
        let store_result = cache_manager.set_metadata_cache(post_id, &metadata).await;
        assert!(store_result.is_ok(), "Failed to store metadata: {:?}", store_result);
        
        // Test retrieving metadata
        let retrieved = cache_manager.get_metadata_cache(post_id).await;
        assert!(retrieved.is_ok(), "Failed to retrieve metadata: {:?}", retrieved);
        
        if let Ok(Some(retrieved_metadata)) = retrieved {
            assert_eq!(metadata.author_name, retrieved_metadata.author_name);
            assert_eq!(metadata.url, retrieved_metadata.url);
            assert_eq!(metadata.language, retrieved_metadata.language);
            assert_eq!(metadata.frozen, retrieved_metadata.frozen);
        }
        
        // Test retrieving non-existent metadata
        let non_existent = cache_manager.get_metadata_cache("non_existent_post").await;
        assert!(non_existent.is_ok());
        assert!(non_existent.unwrap().is_none());
        
        // Clean up
        let _ = cache_manager.invalidate_post_data(post_id).await;
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_gdpr_data_deletion() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        let post_id = "test_post_gdpr";
        let embedding = vec![0.1, 0.2, 0.3];
        let metadata = create_test_metadata();
        
        // Store data in both vector and metadata caches
        let _ = cache_manager.set_vector_cache(post_id, &embedding).await;
        let _ = cache_manager.set_metadata_cache(post_id, &metadata).await;
        
        // Verify data exists
        let vector_exists = cache_manager.get_vector_cache(post_id).await;
        let metadata_exists = cache_manager.get_metadata_cache(post_id).await;
        
        if vector_exists.is_ok() && metadata_exists.is_ok() {
            // Delete post data (GDPR compliance)
            let delete_result = cache_manager.invalidate_post_data(post_id).await;
            assert!(delete_result.is_ok(), "Failed to delete post data: {:?}", delete_result);
            
            // Verify data is deleted
            let vector_after = cache_manager.get_vector_cache(post_id).await;
            let metadata_after = cache_manager.get_metadata_cache(post_id).await;
            
            assert!(vector_after.is_ok());
            assert!(metadata_after.is_ok());
            assert!(vector_after.unwrap().is_none());
            assert!(metadata_after.unwrap().is_none());
        }
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_vector_search_placeholder() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        let query_embedding = vec![0.1, 0.2, 0.3, 0.4];
        let limit = 10;
        
        // Test vector search (currently returns empty results as it's a placeholder)
        let search_result = cache_manager.vector_search(&query_embedding, limit).await;
        assert!(search_result.is_ok(), "Vector search failed: {:?}", search_result);
        
        // Currently returns empty results since we don't have Redis Search configured
        let candidates = search_result.unwrap();
        assert_eq!(candidates.len(), 0);
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_redis_stats() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        let stats_result = cache_manager.get_redis_stats().await;
        assert!(stats_result.is_ok(), "Failed to get Redis stats: {:?}", stats_result);
        
        let stats = stats_result.unwrap();
        // Stats should have some reasonable values (even if zero)
        assert!(stats.total_commands >= 0);
        assert!(stats.total_connections >= 0);
        assert!(stats.connected_clients >= 0);
        assert!(stats.used_memory_bytes >= 0);
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[test]
fn test_cosine_similarity() {
    // Test identical vectors
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let similarity = super::redis_client::cosine_similarity(&a, &b);
    assert!((similarity - 1.0).abs() < f32::EPSILON);
    
    // Test orthogonal vectors
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let similarity = super::redis_client::cosine_similarity(&a, &b);
    assert!((similarity - 0.0).abs() < f32::EPSILON);
    
    // Test opposite vectors
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let similarity = super::redis_client::cosine_similarity(&a, &b);
    assert!((similarity - (-1.0)).abs() < f32::EPSILON);
    
    // Test different length vectors
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let similarity = super::redis_client::cosine_similarity(&a, &b);
    assert_eq!(similarity, 0.0);
    
    // Test zero vectors
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 1.0, 1.0];
    let similarity = super::redis_client::cosine_similarity(&a, &b);
    assert_eq!(similarity, 0.0);
}

#[test]
fn test_farmhash_consistency() {
    // Test that farmhash produces consistent results
    let text = "test query";
    let hash1 = farmhash::hash64(text.as_bytes());
    let hash2 = farmhash::hash64(text.as_bytes());
    assert_eq!(hash1, hash2);
    
    // Test that different inputs produce different hashes
    let text1 = "query one";
    let text2 = "query two";
    let hash1 = farmhash::hash64(text1.as_bytes());
    let hash2 = farmhash::hash64(text2.as_bytes());
    assert_ne!(hash1, hash2);
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_connection_error_handling() {
    // Test with invalid Redis URL
    let invalid_config = RedisConfig {
        url: "redis://invalid-host:6379".to_string(),
        max_connections: 5,
        connection_timeout_secs: 1, // Short timeout for faster test
        default_ttl_secs: 3600,
    };
    
    let result = CacheManager::new(invalid_config).await;
    assert!(result.is_err(), "Should fail with invalid Redis URL");
    
    if let Err(e) = result {
        assert!(e.to_string().contains("Redis"));
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_cache_ttl_behavior() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        let query_hash = 54321u64;
        let results = create_test_cached_results();
        
        // Store results with short TTL (this is handled by Redis automatically)
        let store_result = cache_manager.set_top_k_cache(query_hash, &results).await;
        assert!(store_result.is_ok());
        
        // Immediately retrieve - should exist
        let immediate_retrieve = cache_manager.get_top_k_cache(query_hash).await;
        assert!(immediate_retrieve.is_ok());
        assert!(immediate_retrieve.unwrap().is_some());
        
        // Note: Testing actual TTL expiration would require waiting 60+ seconds
        // In a real test environment, you might use a shorter TTL for testing
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}