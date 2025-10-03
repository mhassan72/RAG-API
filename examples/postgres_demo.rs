use rag_search_api::{Config, DatabaseManager, Post, SearchResult};
use chrono::Utc;
use std::env;
use uuid::Uuid;

/// Sanitize URL for logging by masking credentials
fn sanitize_url_for_logging(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        let mut sanitized = parsed.clone();
        if parsed.password().is_some() {
            let _ = sanitized.set_password(Some("***"));
        }
        if !parsed.username().is_empty() {
            let _ = sanitized.set_username("***");
        }
        sanitized.to_string()
    } else {
        // If URL parsing fails, just mask the entire thing after the protocol
        if let Some(pos) = url.find("://") {
            format!("{}://***", &url[..pos])
        } else {
            "***".to_string()
        }
    }
}

#[tokio::main]
async fn main() -> SearchResult<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("Postgres Database Demo");
    println!("=====================");

    // Load configuration from environment
    let config = Config::from_env()?;
    
    println!("Connecting to Postgres at: {}", sanitize_url_for_logging(&config.database.supabase_url));
    
    // Create database manager
    let db_manager = match DatabaseManager::new(config.database).await {
        Ok(manager) => {
            println!("✅ Successfully connected to Postgres");
            manager
        }
        Err(e) => {
            println!("❌ Failed to connect to Postgres: {}", e);
            println!("Make sure Postgres is running and DATABASE_URL is set correctly");
            return Err(e);
        }
    };

    // Test health check
    match db_manager.health_check().await {
        Ok(()) => println!("✅ Postgres health check passed"),
        Err(e) => {
            println!("❌ Postgres health check failed: {}", e);
            return Err(e);
        }
    }

    // Initialize schema
    println!("\n🔧 Initializing Database Schema");
    match db_manager.initialize_schema().await {
        Ok(()) => println!("✅ Database schema initialized"),
        Err(e) => {
            println!("❌ Failed to initialize schema: {}", e);
            return Err(e);
        }
    }

    // Create vector indexes
    println!("🔧 Creating pgvector indexes");
    match db_manager.create_vector_indexes().await {
        Ok(()) => println!("✅ Vector indexes created"),
        Err(e) => {
            println!("⚠️  Failed to create vector indexes (pgvector may not be installed): {}", e);
            // Continue without vector indexes for basic demo
        }
    }

    // Test CRUD operations
    println!("\n📝 Testing CRUD Operations");
    
    // Create test posts
    let test_posts = vec![
        Post {
            id: Uuid::new_v4(),
            post_id: "demo_post_1".to_string(),
            title: "Introduction to Rust".to_string(),
            content: "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.".to_string(),
            author_name: "Rust Developer".to_string(),
            language: "en".to_string(),
            frozen: false,
            date_gmt: Utc::now(),
            url: "https://example.com/rust-intro".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], // 8-dim for demo
        },
        Post {
            id: Uuid::new_v4(),
            post_id: "demo_post_2".to_string(),
            title: "Vector Databases Explained".to_string(),
            content: "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.".to_string(),
            author_name: "Data Scientist".to_string(),
            language: "en".to_string(),
            frozen: false,
            date_gmt: Utc::now(),
            url: "https://example.com/vector-db".to_string(),
            embedding: vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], // Different embedding
        },
        Post {
            id: Uuid::new_v4(),
            post_id: "demo_post_3".to_string(),
            title: "Machine Learning with Rust".to_string(),
            content: "Combining Rust's performance with machine learning capabilities opens up new possibilities for AI applications.".to_string(),
            author_name: "ML Engineer".to_string(),
            language: "en".to_string(),
            frozen: false,
            date_gmt: Utc::now(),
            url: "https://example.com/ml-rust".to_string(),
            embedding: vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], // Another embedding
        },
    ];

    // Store posts
    println!("Storing {} test posts...", test_posts.len());
    for post in &test_posts {
        match db_manager.store_post(post).await {
            Ok(()) => println!("✅ Stored post: {}", post.title),
            Err(e) => {
                println!("❌ Failed to store post {}: {}", post.title, e);
                return Err(e);
            }
        }
    }

    // Retrieve individual posts
    println!("\n🔍 Testing Individual Post Retrieval");
    for post in &test_posts {
        match db_manager.get_post_by_id(&post.post_id).await {
            Ok(Some(retrieved)) => {
                println!("✅ Retrieved post: {} by {}", retrieved.title, retrieved.author_name);
            }
            Ok(None) => {
                println!("❌ Post not found: {}", post.post_id);
            }
            Err(e) => {
                println!("❌ Failed to retrieve post {}: {}", post.post_id, e);
            }
        }
    }

    // Test batch retrieval
    println!("\n🔍 Testing Batch Post Retrieval");
    let post_ids: Vec<String> = test_posts.iter().map(|p| p.post_id.clone()).collect();
    match db_manager.get_posts_by_ids(&post_ids).await {
        Ok(retrieved_posts) => {
            println!("✅ Retrieved {} posts in batch", retrieved_posts.len());
            for post in &retrieved_posts {
                println!("  - {} ({})", post.title, post.post_id);
            }
        }
        Err(e) => {
            println!("❌ Failed to retrieve posts in batch: {}", e);
        }
    }

    // Test vector search
    println!("\n🔍 Testing Vector Search");
    let query_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // Similar to first post
    match db_manager.vector_search(&query_embedding, 10).await {
        Ok(candidates) => {
            println!("✅ Vector search completed, found {} candidates", candidates.len());
            for (i, candidate) in candidates.iter().enumerate() {
                println!("  {}. {} (score: {:.3})", i + 1, candidate.post_id, candidate.score);
            }
        }
        Err(e) => {
            println!("⚠️  Vector search failed (may require pgvector): {}", e);
        }
    }

    // Test embedding update
    println!("\n🔄 Testing Embedding Update");
    let new_embedding = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
    match db_manager.update_post_embedding(&test_posts[0].post_id, &new_embedding).await {
        Ok(()) => println!("✅ Updated embedding for post: {}", test_posts[0].post_id),
        Err(e) => println!("❌ Failed to update embedding: {}", e),
    }

    // Get database statistics
    println!("\n📊 Database Statistics");
    match db_manager.get_stats().await {
        Ok(stats) => {
            println!("Total posts: {}", stats.total_posts);
            println!("Posts with embeddings: {}", stats.posts_with_embeddings);
            println!("Frozen posts: {}", stats.frozen_posts);
            println!("Database size: {} bytes", stats.database_size_bytes);
            println!("Active connections: {}/{}", stats.active_connections, stats.max_connections);
        }
        Err(e) => println!("❌ Failed to get database stats: {}", e),
    }

    // Test GDPR compliance (data deletion)
    println!("\n🗑️  Testing GDPR Data Deletion");
    for post in &test_posts {
        match db_manager.delete_post(&post.post_id).await {
            Ok(()) => println!("✅ Deleted post: {}", post.post_id),
            Err(e) => println!("❌ Failed to delete post {}: {}", post.post_id, e),
        }
    }

    // Verify deletion
    println!("\n🔍 Verifying Post Deletion");
    for post in &test_posts {
        match db_manager.get_post_by_id(&post.post_id).await {
            Ok(None) => println!("✅ Confirmed deletion: {}", post.post_id),
            Ok(Some(_)) => println!("❌ Post still exists: {}", post.post_id),
            Err(e) => println!("❌ Error checking deletion: {}", e),
        }
    }

    println!("\n🎉 Postgres demo completed successfully!");
    Ok(())
}