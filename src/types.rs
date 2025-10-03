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
    /// Convert to search response with GDPR-compliant snippet truncation
    pub fn to_search_response(&self, score: f32) -> SearchResponse {
        let snippet = Self::truncate_snippet_for_gdpr(&self.content);
        
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

    /// Truncate content to 300 characters for GDPR compliance
    /// Ensures we don't break in the middle of a word and adds ellipsis if truncated
    pub fn truncate_snippet_for_gdpr(content: &str) -> String {
        const MAX_SNIPPET_LENGTH: usize = 300;
        
        if content.len() <= MAX_SNIPPET_LENGTH {
            return content.to_string();
        }
        
        // Reserve 3 characters for "..."
        let max_content_length = MAX_SNIPPET_LENGTH - 3;
        
        // Find the last word boundary before the limit
        let truncate_at = if let Some(last_space_pos) = content[..max_content_length].rfind(char::is_whitespace) {
            last_space_pos
        } else {
            // No whitespace found, truncate at character boundary
            max_content_length
        };
        
        format!("{}...", &content[..truncate_at].trim_end())
    }
}

impl SearchResponse {
    /// Create a search response with proper GDPR-compliant snippet truncation
    pub fn new(
        post_id: String,
        title: String,
        content: String,
        score: f32,
        meta: PostMetadata,
    ) -> Self {
        let snippet = Post::truncate_snippet_for_gdpr(&content);
        
        Self {
            post_id,
            title,
            snippet,
            score,
            meta,
        }
    }

    /// Validate that the response complies with GDPR requirements
    pub fn validate_gdpr_compliance(&self) -> Result<(), String> {
        // Check snippet length
        if self.snippet.len() > 300 {
            return Err(format!(
                "Snippet exceeds GDPR limit: {} characters (max 300)",
                self.snippet.len()
            ));
        }
        
        // Check for sensitive data patterns (basic validation)
        if self.snippet.contains('\0') || self.snippet.contains('\x1b') {
            return Err("Snippet contains potentially unsafe characters".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_metadata() -> PostMetadata {
        PostMetadata {
            author_name: "Test Author".to_string(),
            url: "https://example.com/test".to_string(),
            date: Utc::now(),
            language: "en".to_string(),
            frozen: false,
        }
    }

    #[test]
    fn test_snippet_truncation_under_limit() {
        let short_content = "This is a short post content.";
        let snippet = Post::truncate_snippet_for_gdpr(short_content);
        
        assert_eq!(snippet, short_content);
        assert!(snippet.len() <= 300);
    }

    #[test]
    fn test_snippet_truncation_exactly_300_chars() {
        let content = "a".repeat(300);
        let snippet = Post::truncate_snippet_for_gdpr(&content);
        
        assert_eq!(snippet, content);
        assert_eq!(snippet.len(), 300);
    }

    #[test]
    fn test_snippet_truncation_over_limit() {
        let long_content = "This is a very long post content that exceeds the 300 character limit for GDPR compliance. ".repeat(10);
        let snippet = Post::truncate_snippet_for_gdpr(&long_content);
        
        assert!(snippet.len() <= 300);
        assert!(snippet.ends_with("..."));
        assert!(!snippet.contains("  ")); // Should not end with double spaces
    }

    #[test]
    fn test_snippet_truncation_word_boundary() {
        let content = "This is a test content with many words that should be truncated at word boundaries to ensure readability and proper formatting for users when the content exceeds the maximum allowed length of three hundred characters for GDPR compliance requirements.";
        let snippet = Post::truncate_snippet_for_gdpr(&content);
        
        assert!(snippet.len() <= 300);
        
        // Only check for ellipsis if content was actually truncated
        if content.len() > 300 {
            assert!(snippet.ends_with("..."));
            
            // Should not end with a partial word (before the ellipsis)
            let without_ellipsis = snippet.trim_end_matches("...");
            assert!(!without_ellipsis.ends_with(char::is_alphabetic));
        } else {
            // If content wasn't truncated, it should be the same as original
            assert_eq!(snippet, content);
        }
    }

    #[test]
    fn test_snippet_truncation_no_whitespace() {
        let content = "a".repeat(350); // No whitespace to break on
        let snippet = Post::truncate_snippet_for_gdpr(&content);
        
        assert!(snippet.len() <= 300);
        assert!(snippet.ends_with("..."));
    }

    #[test]
    fn test_search_response_new_with_truncation() {
        let long_content = "This is a very long post content. ".repeat(20);
        let response = SearchResponse::new(
            "test_post".to_string(),
            "Test Title".to_string(),
            long_content,
            0.85,
            create_test_metadata(),
        );
        
        assert!(response.snippet.len() <= 300);
        assert_eq!(response.post_id, "test_post");
        assert_eq!(response.title, "Test Title");
        assert_eq!(response.score, 0.85);
    }

    #[test]
    fn test_gdpr_compliance_validation_valid() {
        let response = SearchResponse {
            post_id: "test_post".to_string(),
            title: "Test Title".to_string(),
            snippet: "This is a valid snippet under 300 characters.".to_string(),
            score: 0.85,
            meta: create_test_metadata(),
        };
        
        assert!(response.validate_gdpr_compliance().is_ok());
    }

    #[test]
    fn test_gdpr_compliance_validation_too_long() {
        let response = SearchResponse {
            post_id: "test_post".to_string(),
            title: "Test Title".to_string(),
            snippet: "a".repeat(301), // Exceeds 300 character limit
            score: 0.85,
            meta: create_test_metadata(),
        };
        
        let result = response.validate_gdpr_compliance();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds GDPR limit"));
    }

    #[test]
    fn test_gdpr_compliance_validation_unsafe_chars() {
        let response = SearchResponse {
            post_id: "test_post".to_string(),
            title: "Test Title".to_string(),
            snippet: "This snippet contains a null byte\0".to_string(),
            score: 0.85,
            meta: create_test_metadata(),
        };
        
        let result = response.validate_gdpr_compliance();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsafe characters"));
    }

    #[test]
    fn test_post_to_search_response_conversion() {
        let post = Post {
            id: uuid::Uuid::new_v4(),
            post_id: "test_post".to_string(),
            title: "Test Title".to_string(),
            content: "This is the full post content that might be longer than the snippet limit.".to_string(),
            author_name: "Test Author".to_string(),
            language: "en".to_string(),
            frozen: false,
            date_gmt: Utc::now(),
            url: "https://example.com/test".to_string(),
            embedding: vec![0.1; 384],
        };
        
        let response = post.to_search_response(0.92);
        
        assert_eq!(response.post_id, "test_post");
        assert_eq!(response.title, "Test Title");
        assert_eq!(response.score, 0.92);
        assert_eq!(response.meta.author_name, "Test Author");
        assert_eq!(response.meta.language, "en");
        assert!(!response.meta.frozen);
        assert!(response.validate_gdpr_compliance().is_ok());
    }

    #[test]
    fn test_post_to_search_response_long_content() {
        let long_content = "This is a very long post content. ".repeat(20);
        let post = Post {
            id: uuid::Uuid::new_v4(),
            post_id: "test_post".to_string(),
            title: "Test Title".to_string(),
            content: long_content,
            author_name: "Test Author".to_string(),
            language: "en".to_string(),
            frozen: false,
            date_gmt: Utc::now(),
            url: "https://example.com/test".to_string(),
            embedding: vec![0.1; 384],
        };
        
        let response = post.to_search_response(0.92);
        
        assert!(response.snippet.len() <= 300);
        assert!(response.snippet.ends_with("..."));
        assert!(response.validate_gdpr_compliance().is_ok());
    }

    #[test]
    fn test_search_filters_serialization() {
        let filters = SearchFilters {
            language: Some("en".to_string()),
            frozen: Some(false),
        };
        
        // Test that filters can be serialized/deserialized
        let json = serde_json::to_string(&filters).unwrap();
        let deserialized: SearchFilters = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.language, Some("en".to_string()));
        assert_eq!(deserialized.frozen, Some(false));
    }

    #[test]
    fn test_search_request_with_filters() {
        let request = SearchRequest {
            query: "test query".to_string(),
            k: 10,
            min_score: Some(0.5),
            rerank: true,
            filters: Some(SearchFilters {
                language: Some("en".to_string()),
                frozen: Some(false),
            }),
        };
        
        // Test serialization
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: SearchRequest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.query, "test query");
        assert_eq!(deserialized.k, 10);
        assert_eq!(deserialized.min_score, Some(0.5));
        assert!(deserialized.rerank);
        assert!(deserialized.filters.is_some());
        
        let filters = deserialized.filters.unwrap();
        assert_eq!(filters.language, Some("en".to_string()));
        assert_eq!(filters.frozen, Some(false));
    }

    #[test]
    fn test_search_response_serialization() {
        let response = SearchResponse {
            post_id: "test_post".to_string(),
            title: "Test Title".to_string(),
            snippet: "Test snippet content.".to_string(),
            score: 0.85,
            meta: create_test_metadata(),
        };
        
        // Test JSON serialization
        let json = serde_json::to_string(&response).unwrap();
        let deserialized: SearchResponse = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.post_id, "test_post");
        assert_eq!(deserialized.title, "Test Title");
        assert_eq!(deserialized.snippet, "Test snippet content.");
        assert_eq!(deserialized.score, 0.85);
        assert_eq!(deserialized.meta.author_name, "Test Author");
    }
}