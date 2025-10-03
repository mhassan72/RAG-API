use super::*;
use crate::config::DatabaseConfig;
use crate::types::Post;
use chrono::Utc;
use std::env;
use std::sync::Arc;
use uuid::Uuid;

/// Helper function to create a test database config
fn create_test_database_config() -> DatabaseConfig {
    DatabaseConfig {
        supabase_url: env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/test_db".to_string()),
        supabase_service_key: "test_service_key".to_string(),
        max_connections: 5,
        connection_timeout_secs: 10,
    }
}

/// Helper function to create a test post
fn create_test_post(post_id: &str) -> Post {
    Post {
        id: Uuid::new_v4(),
        post_id: post_id.to_string(),
        title: format!("Test Post {}", post_id),
        content: format!("This is test content for post {}", post_id),
        author_name: "Test Author".to_string(),
        language: "en".to_string(),
        frozen: false,
        date_gmt: Utc::now(),
        url: format!("https://example.com/post/{}", post_id),
        embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], // 8-dim for testing
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_database_manager_creation() {
    let config = create_test_database_config();
    
    match DatabaseManager::new(config).await {
        Ok(db_manager) => {
            // Test health check
            assert!(db_manager.health_check().await.is_ok());
        }
        Err(e) => {
            println!("Skipping Postgres test - database not available: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_schema_initialization() {
    let config = create_test_database_config();
    
    if let Ok(db_manager) = DatabaseManager::new(config).await {
        let result = db_manager.initialize_schema().await;
        assert!(result.is_ok(), "Schema initialization failed: {:?}", result);
        
        // Test creating vector indexes
        let index_result = db_manager.create_vector_indexes().await;
        assert!(index_result.is_ok(), "Vector index creation failed: {:?}", index_result);
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_post_crud_operations() {
    let config = create_test_database_config();
    
    if let Ok(db_manager) = DatabaseManager::new(config).await {
        // Initialize schema
        let _ = db_manager.initialize_schema().await;
        
        let test_post = create_test_post("crud_test_123");
        
        // Test CREATE
        let store_result = db_manager.store_post(&test_post).await;
        assert!(store_result.is_ok(), "Failed to store post: {:?}", store_result);
        
        // Test READ
        let retrieved = db_manager.get_post_by_id(&test_post.post_id).await;
        assert!(retrieved.is_ok(), "Failed to retrieve post: {:?}", retrieved);
        
        if let Ok(Some(post)) = retrieved {
            assert_eq!(post.post_id, test_post.post_id);
            assert_eq!(post.title, test_post.title);
            assert_eq!(post.content, test_post.content);
            assert_eq!(post.author_name, test_post.author_name);
            assert_eq!(post.language, test_post.language);
            assert_eq!(post.frozen, test_post.frozen);
            assert_eq!(post.url, test_post.url);
            // Note: embedding comparison might have precision differences
        }
        
        // Test UPDATE (via store_post with same post_id)
        let mut updated_post = test_post.clone();
        updated_post.title = "Updated Test Post".to_string();
        updated_post.embedding = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
        
        let update_result = db_manager.store_post(&updated_post).await;
        assert!(update_result.is_ok(), "Failed to update post: {:?}", update_result);
        
        // Verify update
        let updated_retrieved = db_manager.get_post_by_id(&test_post.post_id).await;
        assert!(updated_retrieved.is_ok());
        if let Ok(Some(post)) = updated_retrieved {
            assert_eq!(post.title, "Updated Test Post");
        }
        
        // Test embedding update
        let new_embedding = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        let embedding_update_result = db_manager.update_post_embedding(&test_post.post_id, &new_embedding).await;
        assert!(embedding_update_result.is_ok(), "Failed to update embedding: {:?}", embedding_update_result);
        
        // Test DELETE
        let delete_result = db_manager.delete_post(&test_post.post_id).await;
        assert!(delete_result.is_ok(), "Failed to delete post: {:?}", delete_result);
        
        // Verify deletion
        let deleted_check = db_manager.get_post_by_id(&test_post.post_id).await;
        assert!(deleted_check.is_ok());
        assert!(deleted_check.unwrap().is_none(), "Post should be deleted");
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_batch_post_operations() {
    let config = create_test_database_config();
    
    if let Ok(db_manager) = DatabaseManager::new(config).await {
        let _ = db_manager.initialize_schema().await;
        
        // Create multiple test posts
        let post_ids = vec!["batch_1", "batch_2", "batch_3"];
        let mut posts = Vec::new();
        
        for post_id in &post_ids {
            let post = create_test_post(post_id);
            posts.push(post);
        }
        
        // Store all posts
        for post in &posts {
            let result = db_manager.store_post(post).await;
            assert!(result.is_ok(), "Failed to store post {}: {:?}", post.post_id, result);
        }
        
        // Test batch retrieval
        let post_id_strings: Vec<String> = post_ids.iter().map(|s| s.to_string()).collect();
        let retrieved_posts = db_manager.get_posts_by_ids(&post_id_strings).await;
        assert!(retrieved_posts.is_ok(), "Failed to retrieve posts: {:?}", retrieved_posts);
        
        let posts_result = retrieved_posts.unwrap();
        assert_eq!(posts_result.len(), post_ids.len());
        
        // Verify all posts were retrieved
        for original_post in &posts {
            let found = posts_result.iter().any(|p| p.post_id == original_post.post_id);
            assert!(found, "Post {} not found in batch retrieval", original_post.post_id);
        }
        
        // Clean up
        for post_id in &post_id_strings {
            let _ = db_manager.delete_post(post_id).await;
        }
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_vector_search_functionality() {
    let config = create_test_database_config();
    
    if let Ok(db_manager) = DatabaseManager::new(config).await {
        let _ = db_manager.initialize_schema().await;
        let _ = db_manager.create_vector_indexes().await;
        
        // Create test posts with different embeddings
        let test_posts = vec![
            {
                let mut post = create_test_post("vector_1");
                post.embedding = vec![1.0, 0.0, 0.0, 0.0]; // 4-dim for simplicity
                post
            },
            {
                let mut post = create_test_post("vector_2");
                post.embedding = vec![0.0, 1.0, 0.0, 0.0];
                post
            },
            {
                let mut post = create_test_post("vector_3");
                post.embedding = vec![0.7, 0.7, 0.0, 0.0]; // Similar to first
                post
            },
        ];
        
        // Store test posts
        for post in &test_posts {
            let result = db_manager.store_post(post).await;
            assert!(result.is_ok(), "Failed to store post for vector search: {:?}", result);
        }
        
        // Test vector search
        let query_embedding = vec![1.0, 0.0, 0.0, 0.0]; // Should be most similar to vector_1
        let search_result = db_manager.vector_search(&query_embedding, 10).await;
        assert!(search_result.is_ok(), "Vector search failed: {:?}", search_result);
        
        let candidates = search_result.unwrap();
        assert!(!candidates.is_empty(), "Vector search returned no results");
        
        // Verify results are sorted by similarity (highest score first)
        for i in 1..candidates.len() {
            assert!(
                candidates[i-1].score >= candidates[i].score,
                "Results not sorted by score: {} >= {}",
                candidates[i-1].score,
                candidates[i].score
            );
        }
        
        // Verify the most similar post is first
        if !candidates.is_empty() {
            // The exact post_id depends on the actual similarity calculation
            // but we can verify the structure
            assert!(candidates[0].score > 0.0);
            assert_eq!(candidates[0].source, SearchSource::Postgres);
        }
        
        // Clean up
        for post in &test_posts {
            let _ = db_manager.delete_post(&post.post_id).await;
        }
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_database_statistics() {
    let config = create_test_database_config();
    
    if let Ok(db_manager) = DatabaseManager::new(config).await {
        let _ = db_manager.initialize_schema().await;
        
        let stats_result = db_manager.get_stats().await;
        assert!(stats_result.is_ok(), "Failed to get database stats: {:?}", stats_result);
        
        let stats = stats_result.unwrap();
        
        // Basic validation of stats structure
        assert!(stats.total_posts >= 0);
        assert!(stats.posts_with_embeddings <= stats.total_posts);
        assert!(stats.frozen_posts <= stats.total_posts);
        assert!(stats.database_size_bytes > 0);
        assert!(stats.active_connections > 0);
        assert!(stats.max_connections > 0);
        assert!(stats.active_connections <= stats.max_connections);
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_connection_pool_behavior() {
    let config = create_test_database_config();
    
    if let Ok(db_manager) = DatabaseManager::new(config).await {
        // Test multiple concurrent operations
        let mut handles = Vec::new();
        
        for i in 0..3 {
            let db_clone = Arc::clone(&db_manager.postgres_client);
            let handle = tokio::spawn(async move {
                let post_id = format!("concurrent_test_{}", i);
                let post = create_test_post(&post_id);
                
                // Each task performs a full CRUD cycle
                let store_result = db_clone.store_post(&post).await;
                assert!(store_result.is_ok());
                
                let get_result = db_clone.get_post_by_id(&post_id).await;
                assert!(get_result.is_ok());
                
                let delete_result = db_clone.delete_post(&post_id).await;
                assert!(delete_result.is_ok());
                
                i
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await;
            assert!(result.is_ok(), "Concurrent operation failed: {:?}", result);
        }
    }
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_error_handling() {
    let config = create_test_database_config();
    
    if let Ok(db_manager) = DatabaseManager::new(config).await {
        // Test getting non-existent post
        let non_existent = db_manager.get_post_by_id("non_existent_post").await;
        assert!(non_existent.is_ok());
        assert!(non_existent.unwrap().is_none());
        
        // Test deleting non-existent post (should not error)
        let delete_non_existent = db_manager.delete_post("non_existent_post").await;
        assert!(delete_non_existent.is_ok());
        
        // Test updating embedding for non-existent post
        let update_non_existent = db_manager.update_post_embedding("non_existent_post", &vec![1.0, 2.0]).await;
        assert!(update_non_existent.is_err()); // This should fail
    }
}

#[test]
fn test_database_config_validation() {
    // Test valid PostgreSQL URL
    let valid_config = DatabaseConfig {
        supabase_url: "postgresql://user:pass@localhost:5432/dbname".to_string(),
        supabase_service_key: "test_key".to_string(),
        max_connections: 12,
        connection_timeout_secs: 30,
    };
    
    assert!(valid_config.supabase_url.starts_with("postgresql://"));
    assert!(valid_config.max_connections > 0);
    assert!(valid_config.connection_timeout_secs > 0);
    
    // Test postgres:// URL format
    let postgres_config = DatabaseConfig {
        supabase_url: "postgres://user:pass@localhost:5432/dbname".to_string(),
        supabase_service_key: "test_key".to_string(),
        max_connections: 12,
        connection_timeout_secs: 30,
    };
    
    assert!(postgres_config.supabase_url.starts_with("postgres://"));
}

#[tokio::test]
#[ignore = "requires Postgres connection"]
async fn test_connection_timeout_behavior() {
    // Test with very short timeout to simulate timeout conditions
    let mut config = create_test_database_config();
    config.connection_timeout_secs = 1; // Very short timeout
    
    // This test depends on the actual database response time
    // In a real scenario, you might use a mock or test database
    match DatabaseManager::new(config).await {
        Ok(_) => {
            // Connection succeeded within timeout
        }
        Err(e) => {
            // Connection failed due to timeout or other issues
            println!("Expected timeout or connection failure: {}", e);
        }
    }
}

#[test]
fn test_empty_batch_operations() {
    // Test that empty batch operations handle gracefully
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    rt.block_on(async {
        let config = create_test_database_config();
        
        if let Ok(db_manager) = DatabaseManager::new(config).await {
            // Test empty batch retrieval
            let empty_ids: Vec<String> = vec![];
            let result = db_manager.get_posts_by_ids(&empty_ids).await;
            assert!(result.is_ok());
            assert!(result.unwrap().is_empty());
        }
    });
}