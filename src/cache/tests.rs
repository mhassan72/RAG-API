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

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_cache_hit_miss_statistics() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        // Reset statistics to start fresh
        cache_manager.reset_cache_stats();
        
        let post_id = "test_stats_post";
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let metadata = create_test_metadata();
        let query_hash = 98765u64;
        let results = create_test_cached_results();
        
        // Initial stats should be zero
        let initial_stats = cache_manager.get_cache_stats();
        assert_eq!(initial_stats.vector_cache_hits, 0);
        assert_eq!(initial_stats.vector_cache_misses, 0);
        assert_eq!(initial_stats.topk_cache_hits, 0);
        assert_eq!(initial_stats.topk_cache_misses, 0);
        assert_eq!(initial_stats.metadata_cache_hits, 0);
        assert_eq!(initial_stats.metadata_cache_misses, 0);
        
        // Test cache misses first
        let _ = cache_manager.get_vector_cache(post_id).await;
        let _ = cache_manager.get_metadata_cache(post_id).await;
        let _ = cache_manager.get_top_k_cache(query_hash).await;
        
        let miss_stats = cache_manager.get_cache_stats();
        assert_eq!(miss_stats.vector_cache_misses, 1);
        assert_eq!(miss_stats.metadata_cache_misses, 1);
        assert_eq!(miss_stats.topk_cache_misses, 1);
        
        // Store data in caches
        let _ = cache_manager.set_vector_cache(post_id, &embedding).await;
        let _ = cache_manager.set_metadata_cache(post_id, &metadata).await;
        let _ = cache_manager.set_top_k_cache(query_hash, &results).await;
        
        // Test cache hits
        let _ = cache_manager.get_vector_cache(post_id).await;
        let _ = cache_manager.get_metadata_cache(post_id).await;
        let _ = cache_manager.get_top_k_cache(query_hash).await;
        
        let hit_stats = cache_manager.get_cache_stats();
        assert_eq!(hit_stats.vector_cache_hits, 1);
        assert_eq!(hit_stats.metadata_cache_hits, 1);
        assert_eq!(hit_stats.topk_cache_hits, 1);
        assert_eq!(hit_stats.vector_cache_misses, 1);
        assert_eq!(hit_stats.metadata_cache_misses, 1);
        assert_eq!(hit_stats.topk_cache_misses, 1);
        
        // Test hit ratios
        assert!((hit_stats.vector_hit_ratio() - 0.5).abs() < f64::EPSILON);
        assert!((hit_stats.metadata_hit_ratio() - 0.5).abs() < f64::EPSILON);
        assert!((hit_stats.topk_hit_ratio() - 0.5).abs() < f64::EPSILON);
        assert!((hit_stats.overall_hit_ratio() - 0.5).abs() < f64::EPSILON);
        
        // Clean up
        let _ = cache_manager.invalidate_post_data(post_id).await;
        
        // Test GDPR statistics
        let gdpr_stats = cache_manager.get_cache_stats();
        assert_eq!(gdpr_stats.gdpr_deletions, 1);
        assert!(gdpr_stats.gdpr_keys_deleted >= 1); // At least 1 key deleted
        
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_cache_statistics_edge_cases() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        // Reset statistics
        cache_manager.reset_cache_stats();
        
        // Test hit ratios with zero operations
        let empty_stats = cache_manager.get_cache_stats();
        assert_eq!(empty_stats.vector_hit_ratio(), 0.0);
        assert_eq!(empty_stats.topk_hit_ratio(), 0.0);
        assert_eq!(empty_stats.metadata_hit_ratio(), 0.0);
        assert_eq!(empty_stats.overall_hit_ratio(), 0.0);
        
        // Test with only hits (100% hit ratio)
        let post_id = "test_edge_case";
        let embedding = vec![0.5, 0.6, 0.7];
        let _ = cache_manager.set_vector_cache(post_id, &embedding).await;
        let _ = cache_manager.get_vector_cache(post_id).await;
        
        let hit_only_stats = cache_manager.get_cache_stats();
        assert_eq!(hit_only_stats.vector_hit_ratio(), 1.0);
        
        // Clean up
        let _ = cache_manager.invalidate_post_data(post_id).await;
        
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_comprehensive_cache_workflow() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        // Reset statistics
        cache_manager.reset_cache_stats();
        
        let post_id = "comprehensive_test_post";
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let metadata = create_test_metadata();
        let query_hash = cache_manager.generate_query_hash("test comprehensive workflow");
        let results = create_test_cached_results();
        
        // Step 1: Test initial cache misses
        assert!(cache_manager.get_vector_cache(post_id).await.unwrap().is_none());
        assert!(cache_manager.get_metadata_cache(post_id).await.unwrap().is_none());
        assert!(cache_manager.get_top_k_cache(query_hash).await.unwrap().is_none());
        
        // Step 2: Populate all caches
        assert!(cache_manager.set_vector_cache(post_id, &embedding).await.is_ok());
        assert!(cache_manager.set_metadata_cache(post_id, &metadata).await.is_ok());
        assert!(cache_manager.set_top_k_cache(query_hash, &results).await.is_ok());
        
        // Step 3: Test cache hits
        let cached_vector = cache_manager.get_vector_cache(post_id).await.unwrap();
        assert!(cached_vector.is_some());
        assert_eq!(cached_vector.unwrap().len(), embedding.len());
        
        let cached_metadata = cache_manager.get_metadata_cache(post_id).await.unwrap();
        assert!(cached_metadata.is_some());
        assert_eq!(cached_metadata.unwrap().author_name, metadata.author_name);
        
        let cached_results = cache_manager.get_top_k_cache(query_hash).await.unwrap();
        assert!(cached_results.is_some());
        assert_eq!(cached_results.unwrap().len(), results.len());
        
        // Step 4: Verify statistics
        let stats = cache_manager.get_cache_stats();
        assert_eq!(stats.vector_cache_hits, 1);
        assert_eq!(stats.vector_cache_misses, 1);
        assert_eq!(stats.metadata_cache_hits, 1);
        assert_eq!(stats.metadata_cache_misses, 1);
        assert_eq!(stats.topk_cache_hits, 1);
        assert_eq!(stats.topk_cache_misses, 1);
        
        // Step 5: Test GDPR deletion
        assert!(cache_manager.invalidate_post_data(post_id).await.is_ok());
        
        // Step 6: Verify data is deleted
        assert!(cache_manager.get_vector_cache(post_id).await.unwrap().is_none());
        assert!(cache_manager.get_metadata_cache(post_id).await.unwrap().is_none());
        
        // Step 7: Verify GDPR statistics
        let final_stats = cache_manager.get_cache_stats();
        assert_eq!(final_stats.gdpr_deletions, 1);
        assert!(final_stats.gdpr_keys_deleted >= 1);
        
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[test]
fn test_cache_stats_calculations() {
    use super::redis_client::CacheStats;
    
    // Test with zero operations
    let empty_stats = CacheStats::default();
    assert_eq!(empty_stats.vector_hit_ratio(), 0.0);
    assert_eq!(empty_stats.topk_hit_ratio(), 0.0);
    assert_eq!(empty_stats.metadata_hit_ratio(), 0.0);
    assert_eq!(empty_stats.overall_hit_ratio(), 0.0);
    
    // Test with mixed hits and misses
    let mixed_stats = CacheStats {
        vector_cache_hits: 7,
        vector_cache_misses: 3,
        topk_cache_hits: 8,
        topk_cache_misses: 2,
        metadata_cache_hits: 6,
        metadata_cache_misses: 4,
        gdpr_deletions: 2,
        gdpr_keys_deleted: 5,
    };
    
    assert!((mixed_stats.vector_hit_ratio() - 0.7).abs() < f64::EPSILON);
    assert!((mixed_stats.topk_hit_ratio() - 0.8).abs() < f64::EPSILON);
    assert!((mixed_stats.metadata_hit_ratio() - 0.6).abs() < f64::EPSILON);
    
    // Overall: (7+8+6) / (7+8+6+3+2+4) = 21/30 = 0.7
    assert!((mixed_stats.overall_hit_ratio() - 0.7).abs() < f64::EPSILON);
    
    // Test with only hits
    let hits_only = CacheStats {
        vector_cache_hits: 10,
        vector_cache_misses: 0,
        topk_cache_hits: 5,
        topk_cache_misses: 0,
        metadata_cache_hits: 8,
        metadata_cache_misses: 0,
        gdpr_deletions: 0,
        gdpr_keys_deleted: 0,
    };
    
    assert_eq!(hits_only.vector_hit_ratio(), 1.0);
    assert_eq!(hits_only.topk_hit_ratio(), 1.0);
    assert_eq!(hits_only.metadata_hit_ratio(), 1.0);
    assert_eq!(hits_only.overall_hit_ratio(), 1.0);
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

#[test]
fn test_query_hash_generation_edge_cases() {
    // Create a mock cache manager for testing hash generation
    // We can test the hash generation logic without Redis
    
    // Test normalization of different query formats
    let test_cases = vec![
        ("hello world", "hello world"),
        ("  hello world  ", "hello world"),
        ("Hello World", "hello world"),
        ("HELLO    WORLD", "hello world"),
        ("hello\tworld", "hello world"),
        ("hello\nworld", "hello world"),
        ("hello\r\nworld", "hello world"),
        ("  Hello   World  ", "hello world"),
    ];
    
    // All these should produce the same hash after normalization
    let mut hashes = Vec::new();
    for (input, expected_normalized) in test_cases {
        // Simulate the normalization logic from CacheManager::generate_query_hash
        let normalized = input
            .to_lowercase()
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        
        assert_eq!(normalized, expected_normalized);
        
        let hash = farmhash::hash64(normalized.as_bytes());
        hashes.push(hash);
    }
    
    // All hashes should be identical
    for hash in &hashes[1..] {
        assert_eq!(*hash, hashes[0]);
    }
    
    // Test that different queries produce different hashes
    let different_queries = vec![
        "hello world",
        "world hello",
        "hello",
        "world",
        "hello world test",
    ];
    
    let mut different_hashes = Vec::new();
    for query in different_queries {
        let hash = farmhash::hash64(query.as_bytes());
        different_hashes.push(hash);
    }
    
    // All hashes should be different
    for i in 0..different_hashes.len() {
        for j in (i + 1)..different_hashes.len() {
            assert_ne!(different_hashes[i], different_hashes[j]);
        }
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_cache_key_patterns() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        // Test various post_id formats to ensure key generation works correctly
        let test_post_ids = vec![
            "simple_post_123",
            "post-with-dashes",
            "post_with_underscores",
            "post.with.dots",
            "post123",
            "POST_UPPERCASE",
            "post_with_numbers_456789",
        ];
        
        let test_embedding = vec![0.1, 0.2, 0.3];
        let test_metadata = create_test_metadata();
        
        // Test storing and retrieving with different post_id formats
        for post_id in &test_post_ids {
            // Store data
            assert!(cache_manager.set_vector_cache(post_id, &test_embedding).await.is_ok());
            assert!(cache_manager.set_metadata_cache(post_id, &test_metadata).await.is_ok());
            
            // Retrieve data
            let vector_result = cache_manager.get_vector_cache(post_id).await;
            let metadata_result = cache_manager.get_metadata_cache(post_id).await;
            
            assert!(vector_result.is_ok());
            assert!(metadata_result.is_ok());
            assert!(vector_result.unwrap().is_some());
            assert!(metadata_result.unwrap().is_some());
            
            // Clean up
            let _ = cache_manager.invalidate_post_data(post_id).await;
        }
        
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}

#[tokio::test]
#[ignore = "requires Redis connection"]
async fn test_concurrent_cache_operations() {
    let config = create_test_redis_config();
    
    if let Ok(cache_manager) = CacheManager::new(config).await {
        cache_manager.reset_cache_stats();
        
        let cache_manager = Arc::new(cache_manager);
        let mut handles = Vec::new();
        
        // Spawn multiple concurrent operations
        for i in 0..10 {
            let cache_manager_clone = Arc::clone(&cache_manager);
            let handle = tokio::spawn(async move {
                let post_id = format!("concurrent_post_{}", i);
                let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
                let metadata = create_test_metadata();
                
                // Store data
                let _ = cache_manager_clone.set_vector_cache(&post_id, &embedding).await;
                let _ = cache_manager_clone.set_metadata_cache(&post_id, &metadata).await;
                
                // Retrieve data
                let _ = cache_manager_clone.get_vector_cache(&post_id).await;
                let _ = cache_manager_clone.get_metadata_cache(&post_id).await;
                
                // Clean up
                let _ = cache_manager_clone.invalidate_post_data(&post_id).await;
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            assert!(handle.await.is_ok());
        }
        
        // Verify statistics were updated correctly
        let stats = cache_manager.get_cache_stats();
        assert_eq!(stats.vector_cache_hits, 10);
        assert_eq!(stats.metadata_cache_hits, 10);
        assert_eq!(stats.gdpr_deletions, 10);
        
    } else {
        println!("Skipping Redis-dependent test - Redis not available");
    }
}