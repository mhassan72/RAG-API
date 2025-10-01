use thiserror::Error;

/// Main error type for the search service
#[derive(Debug, Error)]
pub enum SearchError {
    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Request timeout
    #[error("Request timeout")]
    Timeout,

    /// Redis connection or operation error
    #[error("Redis error: {0}")]
    RedisError(String),

    /// Database connection or query error
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// ML model inference error
    #[error("Model inference error: {0}")]
    ModelError(String),

    /// Cache operation error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Generic internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl SearchError {
    /// Check if error is related to Redis
    pub fn is_redis_error(&self) -> bool {
        matches!(self, SearchError::RedisError(_))
    }

    /// Check if error is related to database
    pub fn is_database_error(&self) -> bool {
        matches!(self, SearchError::DatabaseError(_))
    }

    /// Check if error is related to model inference
    pub fn is_model_error(&self) -> bool {
        matches!(self, SearchError::ModelError(_))
    }

    /// Get HTTP status code for the error
    pub fn status_code(&self) -> u16 {
        match self {
            SearchError::InvalidRequest(_) => 400,
            SearchError::RateLimitExceeded => 429,
            SearchError::Timeout => 504,
            SearchError::RedisError(_) => 500,
            SearchError::DatabaseError(_) => 500,
            SearchError::ModelError(_) => 500,
            SearchError::CacheError(_) => 500,
            SearchError::ConfigError(_) => 500,
            SearchError::IoError(_) => 500,
            SearchError::SerializationError(_) => 500,
            SearchError::Internal(_) => 500,
        }
    }
}

/// Result type alias for search operations
pub type SearchResult<T> = Result<T, SearchError>;

/// Validation error for request parameters
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("Invalid k parameter: {0}")]
    InvalidK(String),
    
    #[error("Invalid score parameter: {0}")]
    InvalidScore(String),
    
    #[error("Invalid filter: {0}")]
    InvalidFilter(String),
}

impl From<ValidationError> for SearchError {
    fn from(err: ValidationError) -> Self {
        SearchError::InvalidRequest(err.to_string())
    }
}