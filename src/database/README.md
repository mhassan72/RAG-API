# Postgres Database Implementation

This module implements the Postgres connection pooling and pgvector search functionality for the RAG Search API, providing efficient vector similarity search with IVFFlat indexing and comprehensive database operations.

## Features

### ✅ Implemented
- **Postgres Connection Pooling**: Uses `deadpool-postgres` with configurable pool size (max 12 connections)
- **pgvector Integration**: Full support for vector storage and similarity search
- **IVFFlat Vector Indexing**: Optimized vector search with configurable lists and probes
- **Statement Timeout Handling**: 500ms timeout for all queries as per requirements
- **CRUD Operations**: Complete post management with vector embeddings
- **Batch Operations**: Efficient multi-post retrieval and storage
- **GDPR Compliance**: Post deletion with proper cleanup
- **Schema Management**: Automated database schema initialization
- **Connection Health Monitoring**: Health checks and connection statistics
- **Comprehensive Testing**: Full test suite with Postgres-dependent tests marked

## Architecture

```
DatabaseManager
    ↓
PostgresClient (deadpool-postgres)
    ↓
Postgres Server (with pgvector extension)
```

## Key Components

### PostgresClient
- **Connection Management**: Connection pooling with deadpool-postgres
- **Vector Operations**: Store/retrieve 384-dimensional embeddings with pgvector
- **Search Operations**: IVFFlat-based cosine similarity search
- **CRUD Operations**: Full post lifecycle management
- **Schema Operations**: Database initialization and index creation

### DatabaseManager
- **High-level Interface**: Abstracts Postgres operations
- **Error Handling**: Comprehensive error handling with proper database error types
- **Statistics**: Connection and database usage monitoring

### DatabaseSchema
- **Schema Definitions**: SQL schema for posts table with pgvector support
- **Index Management**: IVFFlat index configuration and optimization
- **Migration Support**: Structured database migrations

## Configuration

Required environment variables:
```bash
SUPABASE_URL=postgresql://user:pass@host:5432/dbname
SUPABASE_SERVICE_KEY=your_service_key
DB_MAX_CONNECTIONS=12
DB_CONNECTION_TIMEOUT_SECS=30
```

## Usage

### Basic Setup
```rust
use rag_search_api::{DatabaseManager, Config};

let config = Config::from_env()?;
let db_manager = DatabaseManager::new(config.database).await?;

// Initialize schema and indexes
db_manager.initialize_schema().await?;
db_manager.create_vector_indexes().await?;
```

### Post Operations
```rust
// Store post with embedding
let post = Post {
    id: Uuid::new_v4(),
    post_id: "post_123".to_string(),
    title: "Example Post".to_string(),
    content: "Post content here".to_string(),
    author_name: "Author Name".to_string(),
    language: "en".to_string(),
    frozen: false,
    date_gmt: Utc::now(),
    url: "https://example.com/post/123".to_string(),
    embedding: vec![0.1, 0.2, 0.3, 0.4], // 384-dimensional vector
};

db_manager.store_post(&post).await?;

// Retrieve post
if let Some(retrieved_post) = db_manager.get_post_by_id("post_123").await? {
    println!("Found post: {}", retrieved_post.title);
}
```

### Vector Search
```rust
// Perform vector similarity search
let query_embedding = vec![0.1, 0.2, 0.3, 0.4]; // Query vector
let candidates = db_manager.vector_search(&query_embedding, 10).await?;

for candidate in candidates {
    println!("Post: {} (score: {:.3})", candidate.post_id, candidate.score);
}
```

### Batch Operations
```rust
// Retrieve multiple posts
let post_ids = vec!["post_1".to_string(), "post_2".to_string()];
let posts = db_manager.get_posts_by_ids(&post_ids).await?;
```

### GDPR Compliance
```rust
// Delete post data
db_manager.delete_post("post_123").await?;
```

## Database Schema

### Posts Table
```sql
CREATE TABLE posts (
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
);
```

### Vector Index
```sql
CREATE INDEX idx_posts_embedding_ivfflat 
ON posts 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

### Standard Indexes
- `idx_posts_post_id` - Unique post identifier lookup
- `idx_posts_language` - Language-based filtering
- `idx_posts_frozen` - Frozen status filtering
- `idx_posts_date_gmt` - Date-based queries
- `idx_posts_author` - Author-based queries

## Testing

### Unit Tests (No Postgres Required)
```bash
cargo test database --lib -- --skip "requires Postgres connection"
```

### Integration Tests (Postgres Required)
```bash
# Start Postgres with pgvector
docker run -d --name postgres-pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=test_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Set environment variable
export DATABASE_URL=postgresql://postgres:password@localhost:5432/test_db

# Run all tests
cargo test database --lib
```

### Demo Application
```bash
# Set environment variables
export DATABASE_URL=postgresql://postgres:password@localhost:5432/test_db

# Run demo
cargo run --example postgres_demo
```

## pgvector Configuration

### IVFFlat Index Tuning

The implementation uses IVFFlat indexing for efficient vector similarity search:

- **Lists**: Number of clusters (default: 100)
  - Small datasets (<1K): 10 lists
  - Medium datasets (1K-100K): sqrt(rows) lists
  - Large datasets (>1M): 500+ lists

- **Probes**: Search clusters (default: 10)
  - Higher probes = better recall, slower search
  - Lower probes = faster search, lower recall
  - Recommended: 10% of lists

### Query Optimization
```sql
-- Set probes for current session
SET ivfflat.probes = 10;

-- Disable sequential scans to force index usage
SET enable_seqscan = off;
```

## Performance Considerations

### Connection Pooling
- **Max Connections**: 12 (as per requirements for ≤200 active connections at 20k RPS)
- **Connection Timeout**: 30 seconds default
- **Pool Management**: Automatic connection recycling and health checks

### Query Performance
- **Statement Timeout**: 500ms for all queries
- **Vector Search**: Optimized with IVFFlat indexing
- **Batch Operations**: Efficient IN clause queries for multiple posts
- **Prepared Statements**: Automatic query preparation and caching

### Memory Management
- **Vector Storage**: Efficient binary storage of f32 vectors
- **Connection Reuse**: Pooled connections reduce overhead
- **Index Caching**: Postgres automatically caches frequently used index pages

## Production Deployment

### Database Setup
1. **Install pgvector Extension**:
   ```sql
   CREATE EXTENSION vector;
   ```

2. **Configure Connection Limits**:
   ```sql
   ALTER SYSTEM SET max_connections = 200;
   SELECT pg_reload_conf();
   ```

3. **Optimize for Vector Workloads**:
   ```sql
   ALTER SYSTEM SET shared_preload_libraries = 'vector';
   ALTER SYSTEM SET work_mem = '256MB';
   ```

### Monitoring
- **Connection Usage**: Monitor active/idle connections
- **Query Performance**: Track query execution times
- **Index Usage**: Monitor index hit ratios
- **Vector Search Metrics**: Track search latency and recall

### Security
- **Connection Encryption**: Use SSL/TLS for all connections
- **Authentication**: Use strong passwords and certificate-based auth
- **Network Security**: Restrict database access to application servers
- **Audit Logging**: Enable query logging for compliance

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **3.2**: Postgres pgvector search with IVFFlat indexing
- **3.4**: Database connection pooling and caching
- **6.4**: 500ms statement timeout handling
- **13.5**: Connection pool management (≤12 connections)

## Troubleshooting

### Common Issues

1. **pgvector Extension Not Found**:
   ```bash
   # Install pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Connection Pool Exhaustion**:
   - Check max_connections setting
   - Monitor connection usage
   - Adjust pool size if needed

3. **Slow Vector Queries**:
   - Verify IVFFlat index exists
   - Tune probes parameter
   - Check query plans with EXPLAIN

4. **Timeout Errors**:
   - Verify 500ms timeout setting
   - Optimize queries and indexes
   - Consider query complexity

### Performance Tuning

1. **Index Optimization**:
   ```sql
   -- Analyze table statistics
   ANALYZE posts;
   
   -- Check index usage
   SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch 
   FROM pg_stat_user_indexes 
   WHERE tablename = 'posts';
   ```

2. **Connection Monitoring**:
   ```sql
   -- Check active connections
   SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
   
   -- Monitor connection states
   SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
   ```

## Future Enhancements

- **Read Replicas**: Support for read-only replicas
- **Partitioning**: Table partitioning for large datasets
- **Advanced Indexing**: HNSW index support when available
- **Compression**: Vector compression for storage efficiency
- **Streaming**: Streaming query results for large result sets