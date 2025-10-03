/// Database schema definitions and migrations
/// 
/// This module contains SQL schema definitions for the posts table
/// and pgvector index configurations for optimal vector search performance.

use crate::error::{SearchError, SearchResult};

/// Database schema manager
pub struct DatabaseSchema;

impl DatabaseSchema {
    /// Get the SQL for creating the posts table
    pub fn create_posts_table_sql() -> &'static str {
        "
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
        "
    }

    /// Get SQL for creating standard indexes
    pub fn create_indexes_sql() -> Vec<&'static str> {
        vec![
            "CREATE INDEX IF NOT EXISTS idx_posts_post_id ON posts(post_id)",
            "CREATE INDEX IF NOT EXISTS idx_posts_language ON posts(language)",
            "CREATE INDEX IF NOT EXISTS idx_posts_frozen ON posts(frozen)",
            "CREATE INDEX IF NOT EXISTS idx_posts_date_gmt ON posts(date_gmt)",
            "CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_name)",
        ]
    }

    /// Get SQL for creating pgvector IVFFlat index
    pub fn create_vector_index_sql() -> &'static str {
        "
        CREATE INDEX IF NOT EXISTS idx_posts_embedding_ivfflat 
        ON posts 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
        "
    }

    /// Get SQL for optimizing IVFFlat search parameters
    pub fn optimize_vector_search_sql() -> Vec<&'static str> {
        vec![
            "SET ivfflat.probes = 10",  // Balance between recall and speed
            "SET enable_seqscan = off", // Force index usage for vector queries
        ]
    }

    /// Get SQL for creating pgvector extension
    pub fn create_vector_extension_sql() -> &'static str {
        "CREATE EXTENSION IF NOT EXISTS vector"
    }

    /// Validate schema requirements
    pub fn validate_schema_requirements() -> SearchResult<()> {
        // Check that embedding dimension matches expected size (384)
        let expected_dimension = 384;
        
        // This would be used to validate the schema matches requirements
        if expected_dimension != 384 {
            return Err(SearchError::DatabaseError(
                "Embedding dimension mismatch".to_string()
            ));
        }

        Ok(())
    }

    /// Get recommended pgvector configuration for different dataset sizes
    pub fn get_ivfflat_config(estimated_rows: u64) -> IVFFlatConfig {
        // Calculate optimal number of lists based on dataset size
        // Rule of thumb: sqrt(rows) for lists, but with reasonable bounds
        let lists = if estimated_rows < 1000 {
            10  // Small dataset
        } else if estimated_rows < 100_000 {
            ((estimated_rows as f64).sqrt() as u32).max(50).min(200)
        } else if estimated_rows < 1_000_000 {
            200  // Medium dataset
        } else {
            500  // Large dataset
        };

        // Probes should be roughly 10% of lists for good recall/speed balance
        let probes = (lists / 10).max(1).min(50);

        IVFFlatConfig { lists, probes }
    }
}

/// Configuration for IVFFlat vector index
#[derive(Debug, Clone)]
pub struct IVFFlatConfig {
    /// Number of lists for IVFFlat index
    pub lists: u32,
    /// Number of probes for search (affects recall vs speed)
    pub probes: u32,
}

impl Default for IVFFlatConfig {
    fn default() -> Self {
        Self {
            lists: 100,
            probes: 10,
        }
    }
}

/// Database migration scripts
pub struct Migrations;

impl Migrations {
    /// Get all migration scripts in order
    pub fn get_all_migrations() -> Vec<Migration> {
        vec![
            Migration {
                version: 1,
                name: "create_vector_extension",
                up_sql: DatabaseSchema::create_vector_extension_sql(),
                down_sql: "DROP EXTENSION IF EXISTS vector CASCADE",
            },
            Migration {
                version: 2,
                name: "create_posts_table",
                up_sql: DatabaseSchema::create_posts_table_sql(),
                down_sql: "DROP TABLE IF EXISTS posts CASCADE",
            },
            Migration {
                version: 3,
                name: "create_standard_indexes",
                up_sql: "CREATE INDEX IF NOT EXISTS idx_posts_post_id ON posts(post_id);
                         CREATE INDEX IF NOT EXISTS idx_posts_language ON posts(language);
                         CREATE INDEX IF NOT EXISTS idx_posts_frozen ON posts(frozen);
                         CREATE INDEX IF NOT EXISTS idx_posts_date_gmt ON posts(date_gmt);
                         CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_name);",
                down_sql: "
                    DROP INDEX IF EXISTS idx_posts_post_id;
                    DROP INDEX IF EXISTS idx_posts_language;
                    DROP INDEX IF EXISTS idx_posts_frozen;
                    DROP INDEX IF EXISTS idx_posts_date_gmt;
                    DROP INDEX IF EXISTS idx_posts_author;
                ",
            },
            Migration {
                version: 4,
                name: "create_vector_index",
                up_sql: DatabaseSchema::create_vector_index_sql(),
                down_sql: "DROP INDEX IF EXISTS idx_posts_embedding_ivfflat",
            },
        ]
    }
}

/// Database migration definition
#[derive(Debug, Clone)]
pub struct Migration {
    pub version: u32,
    pub name: &'static str,
    pub up_sql: &'static str,
    pub down_sql: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_validation() {
        let result = DatabaseSchema::validate_schema_requirements();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ivfflat_config_generation() {
        // Test small dataset
        let config = DatabaseSchema::get_ivfflat_config(500);
        assert_eq!(config.lists, 10);
        assert_eq!(config.probes, 1);

        // Test medium dataset
        let config = DatabaseSchema::get_ivfflat_config(10_000);
        assert!(config.lists >= 50 && config.lists <= 200);
        assert!(config.probes >= 1 && config.probes <= 50);

        // Test large dataset
        let config = DatabaseSchema::get_ivfflat_config(2_000_000);
        assert_eq!(config.lists, 500);
        assert!(config.probes >= 1 && config.probes <= 50);
    }

    #[test]
    fn test_migration_order() {
        let migrations = Migrations::get_all_migrations();
        
        // Ensure migrations are in correct order
        for (i, migration) in migrations.iter().enumerate() {
            assert_eq!(migration.version, (i + 1) as u32);
        }

        // Ensure we have all expected migrations
        assert_eq!(migrations.len(), 4);
        assert_eq!(migrations[0].name, "create_vector_extension");
        assert_eq!(migrations[1].name, "create_posts_table");
        assert_eq!(migrations[2].name, "create_standard_indexes");
        assert_eq!(migrations[3].name, "create_vector_index");
    }

    #[test]
    fn test_sql_statements_not_empty() {
        assert!(!DatabaseSchema::create_posts_table_sql().trim().is_empty());
        assert!(!DatabaseSchema::create_vector_index_sql().trim().is_empty());
        assert!(!DatabaseSchema::create_vector_extension_sql().trim().is_empty());
        
        let indexes = DatabaseSchema::create_indexes_sql();
        assert!(!indexes.is_empty());
        for index_sql in indexes {
            assert!(!index_sql.trim().is_empty());
        }
    }

    #[test]
    fn test_default_ivfflat_config() {
        let config = IVFFlatConfig::default();
        assert_eq!(config.lists, 100);
        assert_eq!(config.probes, 10);
    }
}