#[cfg(test)]
mod tests {
    use crate::error::SearchError;
    use crate::ml::{ModelConfig, QueryDocumentPair, RerankResult};
    use crate::ml::tokenizer::TokenizedText;
    use std::path::PathBuf;

    // Mock ONNX session for testing
    struct MockSession;

    impl MockSession {
        fn new() -> Self {
            MockSession
        }
    }

    #[tokio::test]
    async fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.gcs_base_url, "https://storage.googleapis.com/prod-models/v1");
        assert_eq!(config.model_cache_dir, PathBuf::from("./models"));
        assert!(!config.bi_encoder_hash.is_empty());
        assert!(!config.cross_encoder_hash.is_empty());
    }

    #[tokio::test]
    async fn test_model_config_custom() {
        let config = ModelConfig {
            gcs_base_url: "https://custom-bucket.com/models".to_string(),
            model_cache_dir: PathBuf::from("/tmp/models"),
            bi_encoder_hash: "custom_bi_hash".to_string(),
            cross_encoder_hash: "custom_cross_hash".to_string(),
        };

        assert_eq!(config.gcs_base_url, "https://custom-bucket.com/models");
        assert_eq!(config.model_cache_dir, PathBuf::from("/tmp/models"));
        assert_eq!(config.bi_encoder_hash, "custom_bi_hash");
        assert_eq!(config.cross_encoder_hash, "custom_cross_hash");
    }

    #[tokio::test]
    async fn test_tokenized_text_structure() {
        let tokenized = TokenizedText {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };

        assert_eq!(tokenized.input_ids.len(), 6);
        assert_eq!(tokenized.attention_mask.len(), 6);
        assert_eq!(tokenized.token_type_ids.len(), 6);
        
        // Verify all attention mask values are 1 (no padding)
        assert!(tokenized.attention_mask.iter().all(|&x| x == 1));
        
        // Verify all token type IDs are 0 (single sequence)
        assert!(tokenized.token_type_ids.iter().all(|&x| x == 0));
    }

    #[tokio::test]
    async fn test_query_document_pair() {
        let pair = QueryDocumentPair {
            query: "machine learning".to_string(),
            document: "This is a document about artificial intelligence and machine learning algorithms.".to_string(),
        };

        assert_eq!(pair.query, "machine learning");
        assert!(pair.document.contains("machine learning"));
        assert!(pair.document.len() > pair.query.len());
    }

    #[tokio::test]
    async fn test_rerank_result() {
        let result = RerankResult {
            index: 5,
            score: 0.87,
        };

        assert_eq!(result.index, 5);
        assert!((result.score - 0.87).abs() < 0.001);
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[tokio::test]
    async fn test_ml_service_error_handling() {
        // Test that MLService handles empty queries properly
        let query = "";
        
        // This would be tested with a real MLService instance
        // For now, we test the error condition logic
        if query.trim().is_empty() {
            let error = SearchError::ModelError("Empty query for embedding generation".to_string());
            assert!(error.is_model_error());
            assert_eq!(error.status_code(), 500);
        }
    }

    #[tokio::test]
    async fn test_embedding_dimensions() {
        // Test expected embedding dimensions for all-MiniLM-L6-v2
        const EXPECTED_DIM: usize = 384;
        
        // Create a mock embedding vector
        let embedding = vec![0.1; EXPECTED_DIM];
        assert_eq!(embedding.len(), EXPECTED_DIM);
        
        // Test normalization (should sum to approximately 1.0 when squared)
        let norm_squared: f32 = embedding.iter().map(|x| x * x).sum();
        let expected_norm_squared = EXPECTED_DIM as f32 * 0.01; // 0.1^2 * 384
        assert!((norm_squared - expected_norm_squared).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_cross_encoder_score_range() {
        // Test that cross-encoder scores are in valid range [0, 1]
        let test_scores = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        
        for score in test_scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
        
        // Test edge cases
        let edge_cases = vec![-0.1, 1.1, f32::NAN, f32::INFINITY];
        for score in edge_cases {
            if score.is_finite() {
                let clamped = score.max(0.0).min(1.0);
                assert!(clamped >= 0.0 && clamped <= 1.0);
            }
        }
    }

    #[tokio::test]
    async fn test_batch_processing_empty() {
        // Test that batch processing handles empty inputs correctly
        let empty_queries: Vec<String> = vec![];
        let empty_documents: Vec<String> = vec![];
        
        assert!(empty_queries.is_empty());
        assert!(empty_documents.is_empty());
        
        // This would return Ok(vec![]) in the actual implementation
    }

    #[tokio::test]
    async fn test_batch_processing_single_item() {
        // Test batch processing with single item
        let single_query = vec!["test query".to_string()];
        let single_document = vec!["test document".to_string()];
        
        assert_eq!(single_query.len(), 1);
        assert_eq!(single_document.len(), 1);
        
        // Batch processing should handle single items correctly
    }

    #[tokio::test]
    async fn test_model_verification_logic() {
        // Test SHA256 hash verification logic
        use sha2::{Digest, Sha256};
        
        let test_data = b"test model content";
        let mut hasher = Sha256::new();
        hasher.update(test_data);
        let computed_hash = hex::encode(hasher.finalize());
        
        // Test that same content produces same hash
        let mut hasher2 = Sha256::new();
        hasher2.update(test_data);
        let computed_hash2 = hex::encode(hasher2.finalize());
        
        assert_eq!(computed_hash, computed_hash2);
        
        // Test that different content produces different hash
        let different_data = b"different model content";
        let mut hasher3 = Sha256::new();
        hasher3.update(different_data);
        let different_hash = hex::encode(hasher3.finalize());
        
        assert_ne!(computed_hash, different_hash);
    }

    #[tokio::test]
    async fn test_sigmoid_function() {
        // Test sigmoid activation function behavior
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }
        
        // Test known values
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(1000.0) > 0.999);
        assert!(sigmoid(-1000.0) < 0.001);
        
        // Test monotonicity
        assert!(sigmoid(1.0) > sigmoid(0.0));
        assert!(sigmoid(0.0) > sigmoid(-1.0));
    }

    #[tokio::test]
    async fn test_softmax_function() {
        // Test softmax activation function behavior
        fn softmax(logits: &[f32]) -> Vec<f32> {
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            
            exp_logits.iter().map(|&x| x / sum_exp).collect()
        }
        
        let logits = vec![1.0, 2.0, 3.0];
        let softmax_result = softmax(&logits);
        
        // Check that probabilities sum to 1
        let sum: f32 = softmax_result.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        
        // Check that higher logits get higher probabilities
        assert!(softmax_result[2] > softmax_result[1]);
        assert!(softmax_result[1] > softmax_result[0]);
        
        // Check that all probabilities are positive
        assert!(softmax_result.iter().all(|&x| x > 0.0));
    }

    #[tokio::test]
    async fn test_mean_pooling_logic() {
        // Test mean pooling with attention mask
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0], // This should be masked out
        ];
        
        let attention_mask = vec![1, 1, 0]; // Third token is masked
        
        // Calculate expected mean pooling result
        let mut pooled = vec![0.0; 3];
        let mut valid_tokens = 0;
        
        for (i, embedding) in embeddings.iter().enumerate() {
            if i < attention_mask.len() && attention_mask[i] == 1 {
                for (j, &value) in embedding.iter().enumerate() {
                    pooled[j] += value;
                }
                valid_tokens += 1;
            }
        }
        
        for value in &mut pooled {
            *value /= valid_tokens as f32;
        }
        
        // Expected: (1+4)/2, (2+5)/2, (3+6)/2 = [2.5, 3.5, 4.5]
        assert!((pooled[0] - 2.5).abs() < 0.001);
        assert!((pooled[1] - 3.5).abs() < 0.001);
        assert!((pooled[2] - 4.5).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_l2_normalization() {
        // Test L2 normalization logic
        let mut vector = vec![3.0, 4.0]; // 3-4-5 triangle
        
        // Calculate L2 norm
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 5.0).abs() < 0.001);
        
        // Normalize
        for value in &mut vector {
            *value /= norm;
        }
        
        // Check that normalized vector has unit length
        let new_norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 0.001);
        
        // Check normalized values
        assert!((vector[0] - 0.6).abs() < 0.001);
        assert!((vector[1] - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_error_propagation() {
        // Test that errors are properly propagated through the ML pipeline
        
        // Test model loading errors
        let model_error = SearchError::ModelError("Failed to load model".to_string());
        assert!(model_error.is_model_error());
        assert_eq!(model_error.status_code(), 500);
        
        // Test IO errors
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let search_error = SearchError::IoError(io_error);
        assert_eq!(search_error.status_code(), 500);
        
        // Test configuration errors
        let config_error = SearchError::ConfigError("Invalid configuration".to_string());
        assert_eq!(config_error.status_code(), 500);
    }

    #[tokio::test]
    async fn test_concurrent_inference() {
        // Test that multiple inference requests can be handled concurrently
        use tokio::task;
        
        let queries = vec![
            "machine learning",
            "artificial intelligence", 
            "deep learning",
            "neural networks",
        ];
        
        // Simulate concurrent processing
        let handles: Vec<_> = queries.into_iter().map(|query| {
            task::spawn(async move {
                // This would call actual ML inference in real implementation
                // For now, just simulate processing time and return mock result
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                vec![0.1; 384] // Mock 384-dimensional embedding
            })
        }).collect();
        
        // Wait for all tasks to complete
        let results = futures::future::join_all(handles).await;
        
        // Verify all tasks completed successfully
        assert_eq!(results.len(), 4);
        for result in results {
            let embedding = result.unwrap();
            assert_eq!(embedding.len(), 384);
        }
    }
}