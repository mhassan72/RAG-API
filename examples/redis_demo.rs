use rag_search_api::{CacheManager, Config, SearchResult};
use std::env;

#[tokio::main]
async fn main() -> SearchResult<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("Redis Cache Demo");
    println!("================");

    // Load configuration from environment
    let config = Config::from_env()?;
    
    println!("Connecting to Redis at: {}", config.redis.url);
    
    // Create cache manager
    let cache_manager = match CacheManager::new(config.redis).await {
        Ok(manager) => {
            println!("✅ Successfully connected to Redis");
            manager
        }
        Err(e) => {
            println!("❌ Failed to connect to Redis: {}", e);
            println!("Make sure Redis is running and REDIS_URL is set correctly");
            return Err(e);
        }
    };

    // Test health check
    match cache_manager.health_check().await {
        Ok(()) => println!("✅ Redis health check passed"),
        Err(e) => {
            println!("❌ Redis health check failed: {}", e);
            return Err(e);
        }
    }

    // Test vector cache operations
    println!("\n🔍 Testing Vector Cache Operations");
    let post_id = "demo_post_123";
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    
    println!("Storing vector for post_id: {}", post_id);
    cache_manager.set_vector_cache(post_id, &embedding).await?;
    
    println!("Retrieving vector for post_id: {}", post_id);
    match cache_manager.get_vector_cache(post_id).await? {
        Some(retrieved) => {
            println!("✅ Retrieved vector with {} dimensions", retrieved.len());
            println!("Original:  {:?}", &embedding[..3]);
            println!("Retrieved: {:?}", &retrieved[..3]);
        }
        None => println!("❌ Vector not found"),
    }

    // Test query hash generation
    println!("\n🔍 Testing Query Hash Generation");
    let queries = vec![
        "  Hello World  ",
        "hello world",
        "Hello    World",
        "HELLO WORLD",
    ];
    
    for query in &queries {
        let hash = cache_manager.generate_query_hash(query);
        println!("Query: '{}' -> Hash: {}", query, hash);
    }

    // Test top-k cache
    println!("\n🔍 Testing Top-K Cache");
    let query_hash = cache_manager.generate_query_hash("test query");
    
    // Create some dummy cached results
    use rag_search_api::{CachedResult, PostMetadata};
    use chrono::Utc;
    
    let cached_results = vec![
        CachedResult {
            post_id: "post_1".to_string(),
            title: "First Post".to_string(),
            snippet: "This is the first post snippet".to_string(),
            score: 0.95,
            meta: PostMetadata {
                author_name: "Author One".to_string(),
                url: "https://example.com/post/1".to_string(),
                date: Utc::now(),
                language: "en".to_string(),
                frozen: false,
            },
            cached_at: Utc::now(),
        },
        CachedResult {
            post_id: "post_2".to_string(),
            title: "Second Post".to_string(),
            snippet: "This is the second post snippet".to_string(),
            score: 0.87,
            meta: PostMetadata {
                author_name: "Author Two".to_string(),
                url: "https://example.com/post/2".to_string(),
                date: Utc::now(),
                language: "en".to_string(),
                frozen: false,
            },
            cached_at: Utc::now(),
        },
    ];
    
    println!("Storing {} cached results for query hash: {}", cached_results.len(), query_hash);
    cache_manager.set_top_k_cache(query_hash, &cached_results).await?;
    
    println!("Retrieving cached results for query hash: {}", query_hash);
    match cache_manager.get_top_k_cache(query_hash).await? {
        Some(retrieved) => {
            println!("✅ Retrieved {} cached results", retrieved.len());
            for (i, result) in retrieved.iter().enumerate() {
                println!("  {}. {} (score: {:.3})", i + 1, result.title, result.score);
            }
        }
        None => println!("❌ No cached results found"),
    }

    // Test Redis stats
    println!("\n📊 Redis Statistics");
    match cache_manager.get_redis_stats().await {
        Ok(stats) => {
            println!("Total commands: {}", stats.total_commands);
            println!("Total connections: {}", stats.total_connections);
            println!("Connected clients: {}", stats.connected_clients);
            println!("Used memory: {} bytes", stats.used_memory_bytes);
        }
        Err(e) => println!("❌ Failed to get Redis stats: {}", e),
    }

    // Test vector search (placeholder)
    println!("\n🔍 Testing Vector Search (Placeholder)");
    let query_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    match cache_manager.vector_search(&query_embedding, 10).await {
        Ok(candidates) => {
            println!("✅ Vector search completed, found {} candidates", candidates.len());
            if candidates.is_empty() {
                println!("ℹ️  Note: Vector search returns empty results as it requires Redis Search module configuration");
            }
        }
        Err(e) => println!("❌ Vector search failed: {}", e),
    }

    // Clean up
    println!("\n🧹 Cleaning up test data");
    cache_manager.invalidate_post_data(post_id).await?;
    println!("✅ Test data cleaned up");

    println!("\n🎉 Redis demo completed successfully!");
    Ok(())
}