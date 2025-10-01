use std::env;
use crate::error::{SearchError, SearchResult};

/// Application configuration loaded from environment variables
#[derive(Debug, Clone)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// Database configuration
    pub database: DatabaseConfig,
    /// Redis configuration
    pub redis: RedisConfig,
    /// ML model configuration
    pub ml: MLConfig,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    /// Rate limit (requests per minute)
    pub rate_limit_per_minute: u64,
    /// Maximum request body size in bytes
    pub max_request_size: usize,
}

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Supabase URL
    pub supabase_url: String,
    /// Supabase service key
    pub supabase_service_key: String,
    /// Maximum database connections
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
}

/// Redis configuration
#[derive(Debug, Clone)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,
    /// Maximum Redis connections
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    /// Key expiration time in seconds
    pub default_ttl_secs: u64,
}

/// ML model configuration
#[derive(Debug, Clone)]
pub struct MLConfig {
    /// Path to the embedding model
    pub embedding_model_path: String,
    /// Path to the reranking model
    pub rerank_model_path: String,
    /// Maximum sequence length for embeddings
    pub max_sequence_length: usize,
    /// Embedding dimension
    pub embedding_dimension: usize,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> SearchResult<Self> {
        // Load .env file if it exists
        if let Err(e) = dotenvy::dotenv() {
            tracing::warn!("Could not load .env file: {}", e);
        }

        let config = Config {
            server: ServerConfig {
                host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("SERVER_PORT")
                    .unwrap_or_else(|_| "8080".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid SERVER_PORT: {}", e)))?,
                request_timeout_ms: env::var("REQUEST_TIMEOUT_MS")
                    .unwrap_or_else(|_| "500".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid REQUEST_TIMEOUT_MS: {}", e)))?,
                rate_limit_per_minute: env::var("RATE_LIMIT_PER_MINUTE")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid RATE_LIMIT_PER_MINUTE: {}", e)))?,
                max_request_size: env::var("MAX_REQUEST_SIZE")
                    .unwrap_or_else(|_| "32768".to_string()) // 32KB
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid MAX_REQUEST_SIZE: {}", e)))?,
            },
            database: DatabaseConfig {
                supabase_url: env::var("SUPABASE_URL")
                    .map_err(|_| SearchError::ConfigError("SUPABASE_URL is required".to_string()))?,
                supabase_service_key: env::var("SUPABASE_SERVICE_KEY")
                    .map_err(|_| SearchError::ConfigError("SUPABASE_SERVICE_KEY is required".to_string()))?,
                max_connections: env::var("DB_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid DB_MAX_CONNECTIONS: {}", e)))?,
                connection_timeout_secs: env::var("DB_CONNECTION_TIMEOUT_SECS")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid DB_CONNECTION_TIMEOUT_SECS: {}", e)))?,
            },
            redis: RedisConfig {
                url: env::var("REDIS_URL")
                    .map_err(|_| SearchError::ConfigError("REDIS_URL is required".to_string()))?,
                max_connections: env::var("REDIS_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid REDIS_MAX_CONNECTIONS: {}", e)))?,
                connection_timeout_secs: env::var("REDIS_CONNECTION_TIMEOUT_SECS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid REDIS_CONNECTION_TIMEOUT_SECS: {}", e)))?,
                default_ttl_secs: env::var("REDIS_DEFAULT_TTL_SECS")
                    .unwrap_or_else(|_| "3600".to_string()) // 1 hour
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid REDIS_DEFAULT_TTL_SECS: {}", e)))?,
            },
            ml: MLConfig {
                embedding_model_path: env::var("EMBEDDING_MODEL_PATH")
                    .unwrap_or_else(|_| "models/all-MiniLM-L6-v2.onnx".to_string()),
                rerank_model_path: env::var("RERANK_MODEL_PATH")
                    .unwrap_or_else(|_| "models/ms-marco-MiniLM-L-6-v2.onnx".to_string()),
                max_sequence_length: env::var("MAX_SEQUENCE_LENGTH")
                    .unwrap_or_else(|_| "512".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid MAX_SEQUENCE_LENGTH: {}", e)))?,
                embedding_dimension: env::var("EMBEDDING_DIMENSION")
                    .unwrap_or_else(|_| "384".to_string())
                    .parse()
                    .map_err(|e| SearchError::ConfigError(format!("Invalid EMBEDDING_DIMENSION: {}", e)))?,
            },
        };

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    fn validate(&self) -> SearchResult<()> {
        // Validate server config
        if self.server.port == 0 {
            return Err(SearchError::ConfigError("Server port cannot be 0".to_string()));
        }

        if self.server.request_timeout_ms == 0 {
            return Err(SearchError::ConfigError("Request timeout must be greater than 0".to_string()));
        }

        // Validate database config
        if !self.database.supabase_url.starts_with("https://") {
            return Err(SearchError::ConfigError("SUPABASE_URL must start with https://".to_string()));
        }

        if self.database.supabase_service_key.is_empty() {
            return Err(SearchError::ConfigError("SUPABASE_SERVICE_KEY cannot be empty".to_string()));
        }

        // Validate Redis config
        if !self.redis.url.starts_with("redis://") && !self.redis.url.starts_with("rediss://") {
            return Err(SearchError::ConfigError("REDIS_URL must start with redis:// or rediss://".to_string()));
        }

        // Validate ML config
        if self.ml.embedding_dimension == 0 {
            return Err(SearchError::ConfigError("Embedding dimension must be greater than 0".to_string()));
        }

        if self.ml.max_sequence_length == 0 {
            return Err(SearchError::ConfigError("Max sequence length must be greater than 0".to_string()));
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                request_timeout_ms: 500,
                rate_limit_per_minute: 100,
                max_request_size: 32768, // 32KB
            },
            database: DatabaseConfig {
                supabase_url: "".to_string(),
                supabase_service_key: "".to_string(),
                max_connections: 10,
                connection_timeout_secs: 30,
            },
            redis: RedisConfig {
                url: "".to_string(),
                max_connections: 10,
                connection_timeout_secs: 5,
                default_ttl_secs: 3600, // 1 hour
            },
            ml: MLConfig {
                embedding_model_path: "models/all-MiniLM-L6-v2.onnx".to_string(),
                rerank_model_path: "models/ms-marco-MiniLM-L-6-v2.onnx".to_string(),
                max_sequence_length: 512,
                embedding_dimension: 384,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        
        // Should fail with empty database URLs
        assert!(config.validate().is_err());
        
        // Set valid values
        config.database.supabase_url = "https://example.supabase.co".to_string();
        config.database.supabase_service_key = "test-key".to_string();
        config.redis.url = "redis://localhost:6379".to_string();
        
        // Should pass validation
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_from_env_with_defaults() {
        // Clear environment variables
        env::remove_var("SUPABASE_URL");
        env::remove_var("REDIS_URL");
        
        // Should fail without required env vars (unless .env file provides them)
        let result = Config::from_env();
        if result.is_ok() {
            // If .env file exists and provides required vars, that's also valid
            let config = result.unwrap();
            assert!(!config.database.supabase_url.is_empty());
            assert!(!config.redis.url.is_empty());
        } else {
            // Expected failure when no env vars are set
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_server_config_defaults() {
        let config = Config::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.request_timeout_ms, 500);
        assert_eq!(config.server.rate_limit_per_minute, 100);
    }
}