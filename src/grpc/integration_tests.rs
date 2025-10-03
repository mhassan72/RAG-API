#[cfg(test)]
mod integration_tests {
    use crate::grpc::{
        GrpcSearchRequest, GrpcSearchFilters, 
        validate_grpc_search_request, 
        convert_grpc_to_internal_request, 
        convert_internal_to_grpc_response
    };

    /// Simple integration tests for gRPC functionality
    /// Note: These tests focus on the gRPC layer validation and streaming behavior
    /// Full end-to-end tests would require actual Redis/Postgres instances

    /// Note: These tests are simplified and focus on validation logic
    /// Full integration tests would require mock services or test containers
    
    #[tokio::test]
    async fn test_grpc_request_validation_empty_query() {
        let request = GrpcSearchRequest {
            query: "".to_string(),
            k: 10,
            min_score: None,
            rerank: false,
            filters: None,
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_err(), "Empty query should fail validation");
        assert!(result.unwrap_err().contains("Query cannot be empty"));
    }

    #[tokio::test]
    async fn test_grpc_request_validation_k_parameter() {
        // Test k = 0
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 0,
            min_score: None,
            rerank: false,
            filters: None,
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_err(), "k=0 should fail validation");
        assert!(result.unwrap_err().contains("must be greater than 0"));

        // Test k > 50
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 51,
            min_score: None,
            rerank: false,
            filters: None,
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_err(), "k>50 should fail validation");
        assert!(result.unwrap_err().contains("must not exceed 50"));
    }

    #[tokio::test]
    async fn test_grpc_request_validation_min_score() {
        // Test min_score < 0
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 10,
            min_score: Some(-0.1),
            rerank: false,
            filters: None,
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_err(), "min_score<0 should fail validation");
        assert!(result.unwrap_err().contains("must be between 0.0 and 1.0"));

        // Test min_score > 1
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 10,
            min_score: Some(1.1),
            rerank: false,
            filters: None,
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_err(), "min_score>1 should fail validation");
        assert!(result.unwrap_err().contains("must be between 0.0 and 1.0"));
    }

    #[tokio::test]
    async fn test_grpc_malicious_query_detection() {
        let request = GrpcSearchRequest {
            query: "'; DROP TABLE users;".to_string(),
            k: 10,
            min_score: None,
            rerank: false,
            filters: None,
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_err(), "Malicious query should fail validation");
        assert!(result.unwrap_err().contains("malicious"));
    }

    #[tokio::test]
    async fn test_grpc_language_filter_validation() {
        let request = GrpcSearchRequest {
            query: "test".to_string(),
            k: 10,
            min_score: None,
            rerank: false,
            filters: Some(GrpcSearchFilters {
                language: Some("INVALID123".to_string()),
                frozen: None,
            }),
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_err(), "Invalid language code should fail validation");
        assert!(result.unwrap_err().contains("invalid characters"));
    }

    #[tokio::test]
    async fn test_grpc_valid_request() {
        let request = GrpcSearchRequest {
            query: "machine learning".to_string(),
            k: 10,
            min_score: Some(0.5),
            rerank: true,
            filters: Some(GrpcSearchFilters {
                language: Some("en".to_string()),
                frozen: Some(false),
            }),
        };

        let result = validate_grpc_search_request(&request);
        assert!(result.is_ok(), "Valid request should pass validation");
    }

    #[tokio::test]
    async fn test_grpc_request_conversion() {
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

    #[tokio::test]
    async fn test_grpc_response_conversion() {
        use crate::types::{SearchResponse, PostMetadata};
        use chrono::Utc;

        let internal_response = SearchResponse {
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
}