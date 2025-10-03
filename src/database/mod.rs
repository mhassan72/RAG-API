/// Database module
/// 
/// This module implements Postgres connection pooling and pgvector search functionality
/// with IVFFlat indexing, connection management, and statement timeouts.

mod postgres_client;
mod schema;

#[cfg(test)]
mod tests;

use crate::config::DatabaseConfig;
use crate::error::{SearchError, SearchResult};
use crate::types::{Post, SearchCandidate, SearchSource};
use postgres_client::PostgresClient;
use std::sync::Arc;
use tracing::{debug, info};

pub use postgres_client::PostgresStats;
pub use schema::DatabaseSchema;

/// Database manager for Postgres operations
pub struct DatabaseManager {
    /// Postgres client for all database operations
    postgres_client: Arc<PostgresClient>,
}

impl DatabaseManager {
    /// Create a new database manager with Postgres connection pool
    pub async fn new(database_config: DatabaseConfig) -> SearchResult<Self> {
        info!("Initializing database manager");
        
        let postgres_client = PostgresClient::new(database_config).await?;
        
        // Perform health check
        postgres_client.health_check().await?;
        
        info!("Database manager initialized successfully");
        
        Ok(DatabaseManager {
            postgres_client: Arc::new(postgres_client),
        })
    }

    /// Perform vector similarity search using pgvector
    pub async fn vector_search(&self, query_embedding: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        self.postgres_client.vector_search(query_embedding, limit).await
    }

    /// Get post by ID
    pub async fn get_post_by_id(&self, post_id: &str) -> SearchResult<Option<Post>> {
        self.postgres_client.get_post_by_id(post_id).await
    }

    /// Get multiple posts by IDs
    pub async fn get_posts_by_ids(&self, post_ids: &[String]) -> SearchResult<Vec<Post>> {
        self.postgres_client.get_posts_by_ids(post_ids).await
    }

    /// Store post with vector embedding
    pub async fn store_post(&self, post: &Post) -> SearchResult<()> {
        self.postgres_client.store_post(post).await
    }

    /// Update post embedding
    pub async fn update_post_embedding(&self, post_id: &str, embedding: &[f32]) -> SearchResult<()> {
        self.postgres_client.update_post_embedding(post_id, embedding).await
    }

    /// Delete post (GDPR compliance)
    pub async fn delete_post(&self, post_id: &str) -> SearchResult<()> {
        self.postgres_client.delete_post(post_id).await
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> SearchResult<PostgresStats> {
        self.postgres_client.get_stats().await
    }

    /// Check database connection health
    pub async fn health_check(&self) -> SearchResult<()> {
        self.postgres_client.health_check().await
    }

    /// Initialize database schema and indexes
    pub async fn initialize_schema(&self) -> SearchResult<()> {
        self.postgres_client.initialize_schema().await
    }

    /// Create or update pgvector indexes
    pub async fn create_vector_indexes(&self) -> SearchResult<()> {
        self.postgres_client.create_vector_indexes().await
    }
}