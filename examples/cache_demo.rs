/// Cache Demo
/// 
/// This example demonstrates the three-tier caching strategy implementation
/// including cache statistics tracking and GDPR compliance features.

use rag_search_api::cache::{CacheManager, CacheStats};
use rag_search_api::config::RedisConfig;
use rag_search_api::types::{CachedResult, PostMetadata};
use chrono::Utc;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 RAG Search API - Cache Demo");
    println!("================================");

    // Create Redis configuration
    let redis_config = RedisConfig {
        url: env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
        max_connections: 5,
        connection_timeout_secs: 5,
        default_ttl_secs: 3600,
    };

    // Initialize cache manager
    println!("📡 Connecting to Redis...");
    let cache_manager = match CacheManager::new(redis_config).await {
        Ok(manager) => {
            println!("✅ Connected to Redis successfully");
            manager
        }
        Err(e) => {
            println!("❌ Failed to connect to Redis: {}", e);
            println!("💡 Make sure Redis is running on localhost:6379 or set REDIS_URL");
            return Ok(());
        }
    };

    // Reset statistics for clean demo
    cache_manager.reset_cache_stats();

    // Demo 1: Vector Cache Operations
    println!("\n📊 Demo 1: Vector Cache Operations");
    println!("----------------------------------");

    let post_id = "demo_post_123";
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    // Test cache miss
    println!("🔍 Looking for vector (should be cache miss)...");
    let result = cache_manager.get_vector_cache(post_id).await?;
    println!("   Result: {:?}", result.is_some());

    // Store vector
    println!("💾 Storing vector in cache...");
    cache_manager.set_vector_cache(post_id, &embedding).await?;
    println!("   ✅ Vector stored");

    // Test cache hit
    println!("🔍 Looking for vector again (should be cache hit)...");
    let result = cache_manager.get_vector_cache(post_id).await?;
    println!("   Result: Found vector with {} dimensions", result.unwrap().len());

    // Demo 2: Top-K Cache Operations
    println!("\n🎯 Demo 2: Top-K Cache Operations");
    println!("----------------------------------");

    let query = "machine learning algorithms";
    let query_hash = cache_manager.generate_query_hash(query);
    println!("📝 Query: '{}'", query);
    println!("🔢 Query hash: {}", query_hash);

    // Create sample cached results
    let cached_results = vec![
        CachedResult {
            post_id: "post_1".to_string(),
            title: "Introduction to Machine Learning".to_string(),
            snippet: "Machine learning is a subset of artificial intelligence...".to_string(),
            score: 0.95,
            meta: PostMetadata {
                author_name: "Dr. Smith".to_string(),
                url: "https://example.com/ml-intro".to_string(),
                date: Utc::now(),
                language: "en".to_string(),
                frozen: false,
            },
            cached_at: Utc::now(),
        },
        CachedResult {
            post_id: "post_2".to_string(),
            title: "Advanced ML Algorithms".to_string(),
            snippet: "This post covers advanced machine learning algorithms...".to_string(),
            score: 0.87,
            meta: PostMetadata {
                author_name: "Prof. Johnson".to_string(),
                url: "https://example.com/advanced-ml".to_string(),
                date: Utc::now(),
                language: "en".to_string(),
                frozen: false,
            },
            cached_at: Utc::now(),
        },
    ];

    // Test cache miss
    println!("🔍 Looking for top-k results (should be cache miss)...");
    let result = cache_manager.get_top_k_cache(query_hash).await?;
    println!("   Result: {:?}", result.is_some());

    // Store top-k results
    println!("💾 Storing top-k results in cache (60s TTL)...");
    cache_manager.set_top_k_cache(query_hash, &cached_results).await?;
    println!("   ✅ Top-k results stored");

    // Test cache hit
    println!("🔍 Looking for top-k results again (should be cache hit)...");
    let result = cache_manager.get_top_k_cache(query_hash).await?;
    println!("   Result: Found {} cached results", result.unwrap().len());

    // Demo 3: Metadata Cache Operations
    println!("\n📋 Demo 3: Metadata Cache Operations");
    println!("------------------------------------");

    let metadata = PostMetadata {
        author_name: "Alice Cooper".to_string(),
        url: "https://example.com/post/456".to_string(),
        date: Utc::now(),
        language: "en".to_string(),
        frozen: false,
    };

    // Test cache miss
    println!("🔍 Looking for metadata (should be cache miss)...");
    let result = cache_manager.get_metadata_cache(post_id).await?;
    println!("   Result: {:?}", result.is_some());

    // Store metadata
    println!("💾 Storing metadata in cache (24h TTL)...");
    cache_manager.set_metadata_cache(post_id, &metadata).await?;
    println!("   ✅ Metadata stored");

    // Test cache hit
    println!("🔍 Looking for metadata again (should be cache hit)...");
    let result = cache_manager.get_metadata_cache(post_id).await?;
    let retrieved_metadata = result.unwrap();
    println!("   Result: Found metadata for author '{}'", retrieved_metadata.author_name);

    // Demo 4: Cache Statistics
    println!("\n📈 Demo 4: Cache Statistics");
    println!("---------------------------");

    let stats = cache_manager.get_cache_stats();
    print_cache_stats(&stats);

    // Demo 5: Query Normalization
    println!("\n🔧 Demo 5: Query Normalization");
    println!("------------------------------");

    let test_queries = vec![
        "machine learning",
        "  machine learning  ",
        "Machine Learning",
        "MACHINE    LEARNING",
        "machine\tlearning",
    ];

    println!("Testing query normalization (all should produce same hash):");
    for query in &test_queries {
        let hash = cache_manager.generate_query_hash(query);
        println!("   '{}' -> {}", query, hash);
    }

    // Demo 6: GDPR Compliance
    println!("\n🛡️  Demo 6: GDPR Compliance");
    println!("---------------------------");

    println!("🗑️  Deleting all cached data for post '{}' (GDPR compliance)...", post_id);
    cache_manager.invalidate_post_data(post_id).await?;
    println!("   ✅ Data deleted");

    // Verify deletion
    println!("🔍 Verifying data deletion...");
    let vector_result = cache_manager.get_vector_cache(post_id).await?;
    let metadata_result = cache_manager.get_metadata_cache(post_id).await?;
    println!("   Vector cache: {:?}", vector_result.is_some());
    println!("   Metadata cache: {:?}", metadata_result.is_some());

    // Final statistics
    println!("\n📊 Final Cache Statistics");
    println!("-------------------------");
    let final_stats = cache_manager.get_cache_stats();
    print_cache_stats(&final_stats);

    println!("\n🎉 Cache demo completed successfully!");
    println!("💡 The cache system is ready for production use with:");
    println!("   • Three-tier caching strategy");
    println!("   • Comprehensive statistics tracking");
    println!("   • GDPR compliance features");
    println!("   • Query normalization for better hit rates");

    Ok(())
}

fn print_cache_stats(stats: &CacheStats) {
    println!("   Vector Cache:");
    println!("     Hits: {}, Misses: {}, Hit Ratio: {:.2}%", 
             stats.vector_cache_hits, 
             stats.vector_cache_misses, 
             stats.vector_hit_ratio() * 100.0);
    
    println!("   Top-K Cache:");
    println!("     Hits: {}, Misses: {}, Hit Ratio: {:.2}%", 
             stats.topk_cache_hits, 
             stats.topk_cache_misses, 
             stats.topk_hit_ratio() * 100.0);
    
    println!("   Metadata Cache:");
    println!("     Hits: {}, Misses: {}, Hit Ratio: {:.2}%", 
             stats.metadata_cache_hits, 
             stats.metadata_cache_misses, 
             stats.metadata_hit_ratio() * 100.0);
    
    println!("   Overall Hit Ratio: {:.2}%", stats.overall_hit_ratio() * 100.0);
    
    println!("   GDPR Compliance:");
    println!("     Deletions: {}, Keys Deleted: {}", 
             stats.gdpr_deletions, 
             stats.gdpr_keys_deleted);
}