use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Core search request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Natural language query
    pub query: String,
    /// Maximum number of results to return (max 50)
    pub k: u32,
    /// Minimum similarity score threshold (optional)
    pub min_score: Option<f32>,
    /// Enable cross-encoder reranking
    pub rerank: bool,
    /// Optional filters for search results
    pub filters: Option<SearchFilters>,
}

/// Search filters for metadata-based filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Filter by language (e.g., "en", "es")
    pub language: Option<String>,
    /// Filter by frozen status (false excludes frozen posts)
    pub frozen: Option<bool>,
}

/// Search response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Unique post identifier
    pub post_id: String,
    /// Post title
    pub title: String,
    /// Post snippet (truncated to 300 chars for GDPR)
    pub snippet: String,
    /// Similarity score (0.0 to 1.0)
    pub score: f32,
    /// Additional post metadata
    pub meta: PostMetadata,
}

/// Post metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostMetadata {
    /// Post author name
    pub author_name: String,
    /// Post URL
    pub url: String,
    /// Post publication date
    pub date: DateTime<Utc>,
    /// Post language
    pub language: String,
    /// Whether the post is frozen
    pub frozen: bool,
}

/// Internal post representation
#[derive(Debug, Clone)]
pub struct Post {
    /// Database UUID
    pub id: Uuid,
    /// External post identifier
    pub post_id: String,
    /// Post title
    pub title: String,
    /// Full post content
    pub content: String,
    /// Author name
    pub author_name: String,
    /// Post language
    pub language: String,
    /// Frozen status
    pub frozen: bool,
    /// Publication date
    pub date_gmt: DateTime<Utc>,
    /// Post URL
    pub url: String,
    /// Vector embedding (384 dimensions)
    pub embedding: Vec<f32>,
}

/// Search candidate from vector search
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    /// Post identifier
    pub post_id: String,
    /// Similarity score
    pub score: f32,
    /// Source of the candidate (Redis or Postgres)
    pub source: SearchSource,
}

/// Source of search results
#[derive(Debug, Clone, PartialEq)]
pub enum SearchSource {
    Redis,
    Postgres,
}

/// Cached search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    /// Post identifier
    pub post_id: String,
    /// Post title
    pub title: String,
    /// Post snippet
    pub snippet: String,
    /// Similarity score
    pub score: f32,
    /// Metadata
    pub meta: PostMetadata,
    /// Cache timestamp
    pub cached_at: DateTime<Utc>,
}

/// Search operation mode for graceful degradation
#[derive(Debug, Clone, PartialEq)]
pub enum SearchMode {
    /// Full functionality: Redis + Postgres + Rerank
    Full,
    /// Postgres only (Redis circuit open)
    PostgresOnly,
    /// Redis cache only (Postgres timeout)
    CacheOnly,
    /// No reranking (model inference issues)
    Degraded,
}



impl Post {
    /// Convert to search response with truncated snippet
    pub fn to_search_response(&self, score: f32) -> SearchResponse {
        let snippet = if self.content.len() > 300 {
            format!("{}...", &self.content[..297])
        } else {
            self.content.clone()
        };
        
        SearchResponse {
            post_id: self.post_id.clone(),
            title: self.title.clone(),
            snippet,
            score,
            meta: PostMetadata {
                author_name: self.author_name.clone(),
                url: self.url.clone(),
                date: self.date_gmt,
                language: self.language.clone(),
                frozen: self.frozen,
            },
        }
    }
}