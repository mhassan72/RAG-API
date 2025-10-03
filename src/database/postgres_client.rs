use crate::config::DatabaseConfig;
use crate::error::{SearchError, SearchResult};
use crate::types::{Post, SearchCandidate, SearchSource};
use deadpool_postgres::{Config, Pool, Runtime};
use std::time::Duration;
use tokio::time::timeout;
use tokio_postgres::{NoTls, Row};
use tracing::{debug, info, warn};

/// Postgres client wrapper with connection pooling and pgvector support
pub struct PostgresClient {
    /// Connection pool for Postgres
    pool: Pool,
    /// Configuration
    config: DatabaseConfig,
}

impl PostgresClient {
    /// Create a new Postgres client with connection pooling
    pub async fn new(config: DatabaseConfig) -> SearchResult<Self> {
        info!("Initializing Postgres client with URL: {}", sanitize_url_for_logging(&config.supabase_url));

        // Create deadpool configuration
        let mut pg_config = Config::new();
        
        // Parse Supabase URL to extract connection parameters
        let url = &config.supabase_url;
        if !url.starts_with("postgresql://") && !url.starts_with("postgres://") {
            return Err(SearchError::DatabaseError(
                "Invalid database URL format".to_string()
            ));
        }

        // Set connection parameters
        pg_config.url = Some(url.clone());
        pg_config.pool = Some(deadpool_postgres::PoolConfig::new(config.max_connections as usize));

        // Create the connection pool
        let pool = pg_config
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| SearchError::DatabaseError(format!("Failed to create connection pool: {}", e)))?;

        // Test connection
        let client = pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection from pool: {}", e)))?;

        // Test basic connectivity
        let _rows = client
            .query("SELECT 1", &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to test connection: {}", e)))?;

        info!("Postgres client connected successfully");

        Ok(PostgresClient { pool, config })
    }

    /// Perform vector similarity search using pgvector with IVFFlat
    pub async fn vector_search(&self, query_embedding: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        debug!("Performing Postgres vector search with limit: {}", limit);

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        // Convert f32 vector to pgvector format (array of floats)
        let embedding_str = format!("[{}]", 
            query_embedding.iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Use cosine distance with IVFFlat index
        // The query uses the <=> operator for cosine distance
        let query = "
            SELECT post_id, (embedding <=> $1::vector) as distance
            FROM posts 
            WHERE embedding IS NOT NULL 
              AND NOT frozen
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        ";

        let statement_timeout = Duration::from_millis(500); // 500ms timeout as per requirements
        
        let rows = timeout(statement_timeout, client.query(query, &[&embedding_str, &(limit as i64)]))
            .await
            .map_err(|_| SearchError::DatabaseError("Query timeout exceeded 500ms".to_string()))?
            .map_err(|e| SearchError::DatabaseError(format!("Vector search query failed: {}", e)))?;

        let mut candidates = Vec::new();
        for row in rows {
            let post_id: String = row.get(0);
            let distance: f32 = row.get(1);
            
            // Convert cosine distance to similarity score (1 - distance)
            let score = 1.0 - distance;
            
            candidates.push(SearchCandidate {
                post_id,
                score,
                source: SearchSource::Postgres,
            });
        }

        debug!("Postgres vector search returned {} candidates", candidates.len());
        Ok(candidates)
    }

    /// Get post by ID
    pub async fn get_post_by_id(&self, post_id: &str) -> SearchResult<Option<Post>> {
        debug!("Retrieving post by ID: {}", post_id);

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        let query = "
            SELECT id, post_id, title, content, author_name, language, frozen, date_gmt, url, embedding
            FROM posts 
            WHERE post_id = $1
        ";

        let rows = client
            .query(query, &[&post_id])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get post: {}", e)))?;

        if rows.is_empty() {
            debug!("No post found with ID: {}", post_id);
            return Ok(None);
        }

        let row = &rows[0];
        let post = self.row_to_post(row)?;
        
        debug!("Retrieved post: {}", post.post_id);
        Ok(Some(post))
    }

    /// Get multiple posts by IDs
    pub async fn get_posts_by_ids(&self, post_ids: &[String]) -> SearchResult<Vec<Post>> {
        if post_ids.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Retrieving {} posts by IDs", post_ids.len());

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        // Create placeholders for the IN clause
        let placeholders: Vec<String> = (1..=post_ids.len()).map(|i| format!("${}", i)).collect();
        let query = format!(
            "SELECT id, post_id, title, content, author_name, language, frozen, date_gmt, url, embedding
             FROM posts 
             WHERE post_id IN ({})",
            placeholders.join(", ")
        );

        // Convert post_ids to references for the query
        let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = 
            post_ids.iter().map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync)).collect();

        let rows = client
            .query(&query, &params)
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get posts: {}", e)))?;

        let mut posts = Vec::new();
        for row in rows {
            let post = self.row_to_post(&row)?;
            posts.push(post);
        }

        debug!("Retrieved {} posts", posts.len());
        Ok(posts)
    }

    /// Store post with vector embedding
    pub async fn store_post(&self, post: &Post) -> SearchResult<()> {
        debug!("Storing post: {}", post.post_id);

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        // Convert embedding to pgvector format
        let embedding_str = if post.embedding.is_empty() {
            None
        } else {
            Some(format!("[{}]", 
                post.embedding.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ))
        };

        let query = "
            INSERT INTO posts (id, post_id, title, content, author_name, language, frozen, date_gmt, url, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector)
            ON CONFLICT (post_id) 
            DO UPDATE SET 
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                author_name = EXCLUDED.author_name,
                language = EXCLUDED.language,
                frozen = EXCLUDED.frozen,
                date_gmt = EXCLUDED.date_gmt,
                url = EXCLUDED.url,
                embedding = EXCLUDED.embedding
        ";

        client
            .execute(query, &[
                &post.id,
                &post.post_id,
                &post.title,
                &post.content,
                &post.author_name,
                &post.language,
                &post.frozen,
                &post.date_gmt,
                &post.url,
                &embedding_str,
            ])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to store post: {}", e)))?;

        debug!("Successfully stored post: {}", post.post_id);
        Ok(())
    }

    /// Update post embedding
    pub async fn update_post_embedding(&self, post_id: &str, embedding: &[f32]) -> SearchResult<()> {
        debug!("Updating embedding for post: {}", post_id);

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        let embedding_str = format!("[{}]", 
            embedding.iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        let query = "UPDATE posts SET embedding = $1::vector WHERE post_id = $2";

        let rows_affected = client
            .execute(query, &[&embedding_str, &post_id])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to update embedding: {}", e)))?;

        if rows_affected == 0 {
            return Err(SearchError::DatabaseError(format!("Post not found: {}", post_id)));
        }

        debug!("Successfully updated embedding for post: {}", post_id);
        Ok(())
    }

    /// Delete post (GDPR compliance)
    pub async fn delete_post(&self, post_id: &str) -> SearchResult<()> {
        debug!("Deleting post: {}", post_id);

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        let query = "DELETE FROM posts WHERE post_id = $1";

        let rows_affected = client
            .execute(query, &[&post_id])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to delete post: {}", e)))?;

        if rows_affected == 0 {
            warn!("Attempted to delete non-existent post: {}", post_id);
        } else {
            info!("Successfully deleted post: {}", post_id);
        }

        Ok(())
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> SearchResult<PostgresStats> {
        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        // Get basic database statistics
        let query = "
            SELECT 
                (SELECT COUNT(*) FROM posts) as total_posts,
                (SELECT COUNT(*) FROM posts WHERE embedding IS NOT NULL) as posts_with_embeddings,
                (SELECT COUNT(*) FROM posts WHERE frozen = true) as frozen_posts,
                (SELECT pg_database_size(current_database())) as database_size_bytes
        ";

        let rows = client
            .query(query, &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get stats: {}", e)))?;

        if rows.is_empty() {
            return Err(SearchError::DatabaseError("No stats returned".to_string()));
        }

        let row = &rows[0];
        let stats = PostgresStats {
            total_posts: row.get::<_, i64>(0) as u64,
            posts_with_embeddings: row.get::<_, i64>(1) as u64,
            frozen_posts: row.get::<_, i64>(2) as u64,
            database_size_bytes: row.get::<_, i64>(3) as u64,
            active_connections: self.pool.status().size as u32,
            max_connections: self.pool.status().max_size as u32,
        };

        Ok(stats)
    }

    /// Check database connection health
    pub async fn health_check(&self) -> SearchResult<()> {
        let start = std::time::Instant::now();
        
        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        // Test basic connectivity and pgvector extension
        let rows = client
            .query("SELECT 1, version()", &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Health check query failed: {}", e)))?;

        if rows.is_empty() {
            return Err(SearchError::DatabaseError("Health check returned no results".to_string()));
        }

        // Check if pgvector extension is available
        let extension_check = client
            .query("SELECT 1 FROM pg_extension WHERE extname = 'vector'", &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to check pgvector extension: {}", e)))?;

        if extension_check.is_empty() {
            warn!("pgvector extension not found - vector operations may fail");
        }

        let duration = start.elapsed();
        debug!("Database health check passed in {:?}", duration);
        Ok(())
    }

    /// Initialize database schema and indexes
    pub async fn initialize_schema(&self) -> SearchResult<()> {
        info!("Initializing database schema");

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        // Create pgvector extension if it doesn't exist
        client
            .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to create vector extension: {}", e)))?;

        // Create posts table if it doesn't exist
        let create_table_query = "
            CREATE TABLE IF NOT EXISTS posts (
                id UUID PRIMARY KEY,
                post_id VARCHAR(255) UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                author_name VARCHAR(255) NOT NULL,
                language VARCHAR(10) NOT NULL DEFAULT 'en',
                frozen BOOLEAN NOT NULL DEFAULT false,
                date_gmt TIMESTAMPTZ NOT NULL,
                url TEXT NOT NULL,
                embedding vector(384),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        ";

        client
            .execute(create_table_query, &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to create posts table: {}", e)))?;

        // Create indexes for common queries
        let indexes = vec![
            "CREATE INDEX IF NOT EXISTS idx_posts_post_id ON posts(post_id)",
            "CREATE INDEX IF NOT EXISTS idx_posts_language ON posts(language)",
            "CREATE INDEX IF NOT EXISTS idx_posts_frozen ON posts(frozen)",
            "CREATE INDEX IF NOT EXISTS idx_posts_date_gmt ON posts(date_gmt)",
        ];

        for index_query in indexes {
            client
                .execute(index_query, &[])
                .await
                .map_err(|e| SearchError::DatabaseError(format!("Failed to create index: {}", e)))?;
        }

        info!("Database schema initialized successfully");
        Ok(())
    }

    /// Create or update pgvector indexes with IVFFlat
    pub async fn create_vector_indexes(&self) -> SearchResult<()> {
        info!("Creating pgvector indexes");

        let client = self.pool
            .get()
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to get connection: {}", e)))?;

        // Drop existing vector index if it exists
        client
            .execute("DROP INDEX IF EXISTS idx_posts_embedding_ivfflat", &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to drop existing index: {}", e)))?;

        // Create IVFFlat index for vector similarity search
        // Using 100 lists as a reasonable default for moderate-sized datasets
        let create_index_query = "
            CREATE INDEX idx_posts_embedding_ivfflat 
            ON posts 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100)
        ";

        client
            .execute(create_index_query, &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to create vector index: {}", e)))?;

        // Set probes for query optimization (affects recall vs speed tradeoff)
        client
            .execute("SET ivfflat.probes = 10", &[])
            .await
            .map_err(|e| SearchError::DatabaseError(format!("Failed to set probes: {}", e)))?;

        info!("pgvector indexes created successfully");
        Ok(())
    }

    /// Convert database row to Post struct
    fn row_to_post(&self, row: &Row) -> SearchResult<Post> {
        // Parse embedding from pgvector format
        let embedding_str: Option<String> = row.get(9);
        let embedding = if let Some(emb_str) = embedding_str {
            // Parse "[1.0,2.0,3.0]" format
            let trimmed = emb_str.trim_start_matches('[').trim_end_matches(']');
            if trimmed.is_empty() {
                Vec::new()
            } else {
                trimmed
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<Result<Vec<f32>, _>>()
                    .map_err(|e| SearchError::DatabaseError(format!("Failed to parse embedding: {}", e)))?
            }
        } else {
            Vec::new()
        };

        Ok(Post {
            id: row.get(0),
            post_id: row.get(1),
            title: row.get(2),
            content: row.get(3),
            author_name: row.get(4),
            language: row.get(5),
            frozen: row.get(6),
            date_gmt: row.get(7),
            url: row.get(8),
            embedding,
        })
    }
}

/// Postgres connection statistics
#[derive(Debug, Default)]
pub struct PostgresStats {
    pub total_posts: u64,
    pub posts_with_embeddings: u64,
    pub frozen_posts: u64,
    pub database_size_bytes: u64,
    pub active_connections: u32,
    pub max_connections: u32,
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatabaseConfig;
    use chrono::Utc;
    use std::env;
    use uuid::Uuid;

    fn create_test_database_config() -> DatabaseConfig {
        DatabaseConfig {
            supabase_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/test_db".to_string()),
            supabase_service_key: "test_key".to_string(),
            max_connections: 5,
            connection_timeout_secs: 10,
        }
    }

    fn create_test_post() -> Post {
        Post {
            id: Uuid::new_v4(),
            post_id: "test_post_123".to_string(),
            title: "Test Post".to_string(),
            content: "This is a test post content".to_string(),
            author_name: "Test Author".to_string(),
            language: "en".to_string(),
            frozen: false,
            date_gmt: Utc::now(),
            url: "https://example.com/test-post".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        }
    }

    #[tokio::test]
    #[ignore = "requires Postgres connection"]
    async fn test_postgres_client_creation() {
        let config = create_test_database_config();
        
        match PostgresClient::new(config).await {
            Ok(client) => {
                assert!(client.health_check().await.is_ok());
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
        
        if let Ok(client) = PostgresClient::new(config).await {
            let result = client.initialize_schema().await;
            assert!(result.is_ok(), "Schema initialization failed: {:?}", result);
        }
    }

    #[tokio::test]
    #[ignore = "requires Postgres connection"]
    async fn test_post_operations() {
        let config = create_test_database_config();
        
        if let Ok(client) = PostgresClient::new(config).await {
            // Initialize schema first
            let _ = client.initialize_schema().await;
            
            let test_post = create_test_post();
            
            // Test storing post
            let store_result = client.store_post(&test_post).await;
            assert!(store_result.is_ok(), "Failed to store post: {:?}", store_result);
            
            // Test retrieving post
            let retrieved = client.get_post_by_id(&test_post.post_id).await;
            assert!(retrieved.is_ok(), "Failed to retrieve post: {:?}", retrieved);
            
            if let Ok(Some(post)) = retrieved {
                assert_eq!(post.post_id, test_post.post_id);
                assert_eq!(post.title, test_post.title);
            }
            
            // Test deleting post
            let delete_result = client.delete_post(&test_post.post_id).await;
            assert!(delete_result.is_ok(), "Failed to delete post: {:?}", delete_result);
        }
    }

    #[tokio::test]
    #[ignore = "requires Postgres connection"]
    async fn test_vector_search() {
        let config = create_test_database_config();
        
        if let Ok(client) = PostgresClient::new(config).await {
            let query_embedding = vec![0.1, 0.2, 0.3, 0.4];
            let limit = 10;
            
            let search_result = client.vector_search(&query_embedding, limit).await;
            assert!(search_result.is_ok(), "Vector search failed: {:?}", search_result);
            
            let candidates = search_result.unwrap();
            assert!(candidates.len() <= limit);
        }
    }

    #[test]
    fn test_database_config_validation() {
        let valid_config = DatabaseConfig {
            supabase_url: "postgresql://user:pass@localhost:5432/db".to_string(),
            supabase_service_key: "test_key".to_string(),
            max_connections: 10,
            connection_timeout_secs: 30,
        };
        
        assert!(valid_config.supabase_url.starts_with("postgresql://"));
        assert!(valid_config.max_connections > 0);
        assert!(valid_config.connection_timeout_secs > 0);
    }
}