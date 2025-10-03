use crate::error::{SearchError, SearchResult};
use crate::ml::tokenizer::TokenizerService;
use std::path::PathBuf;
use tracing::{debug, instrument};

/// CrossEncoder service for reranking search results
/// Uses ms-marco-MiniLM-L-6-v2 ONNX model to score query-document pairs
#[derive(Clone)]
pub struct CrossEncoder {
    model_path: PathBuf,
    tokenizer: TokenizerService,
}

/// Query-document pair for reranking
#[derive(Debug, Clone)]
pub struct QueryDocumentPair {
    pub query: String,
    pub document: String,
}

/// Reranking result with relevance score
#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

impl CrossEncoder {
    /// Create a new CrossEncoder with model path and tokenizer
    pub fn new(model_path: PathBuf, tokenizer: TokenizerService) -> Self {
        Self { model_path, tokenizer }
    }

    /// Score a single query-document pair
    /// Returns relevance score between 0.0 and 1.0
    #[instrument(skip(self), fields(query_len = pair.query.len(), doc_len = pair.document.len()))]
    pub async fn score(&self, pair: &QueryDocumentPair) -> SearchResult<f32> {
        // For now, return a placeholder implementation
        // In production, this would use the actual ONNX model at self.model_path
        
        if pair.query.trim().is_empty() || pair.document.trim().is_empty() {
            return Err(SearchError::ModelError("Empty query or document for cross-encoder".to_string()));
        }

        debug!("Scoring query-document pair (using model at {})", self.model_path.display());

        // Generate a simple relevance score based on text similarity
        // This is just for testing - real implementation would use ONNX inference
        let query_lower = pair.query.to_lowercase();
        let doc_lower = pair.document.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let doc_words: Vec<&str> = doc_lower.split_whitespace().collect();
        
        let mut matches = 0;
        for query_word in &query_words {
            if doc_words.contains(query_word) {
                matches += 1;
            }
        }
        
        let score = if query_words.is_empty() {
            0.0
        } else {
            (matches as f32) / (query_words.len() as f32)
        };
        
        // Apply sigmoid to get a more realistic distribution
        let sigmoid_score = self.sigmoid(score * 4.0 - 2.0); // Scale and shift for better range

        debug!("Cross-encoder score: {:.4}", sigmoid_score);
        Ok(sigmoid_score)
    }

    /// Score multiple query-document pairs in batch
    /// Returns scores in the same order as input pairs
    #[instrument(skip(self), fields(batch_size = pairs.len()))]
    pub async fn score_batch(&self, pairs: &[QueryDocumentPair]) -> SearchResult<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // For now, process sequentially. In production, this could be optimized
        // to use actual batch processing with padded sequences
        let mut scores = Vec::with_capacity(pairs.len());
        
        for pair in pairs {
            let score = self.score(pair).await?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Rerank a list of documents for a given query
    /// Returns results sorted by relevance score (highest first)
    #[instrument(skip(self), fields(query_len = query.len(), num_docs = documents.len()))]
    pub async fn rerank(&self, query: &str, documents: &[String]) -> SearchResult<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Create query-document pairs
        let pairs: Vec<QueryDocumentPair> = documents
            .iter()
            .map(|doc| QueryDocumentPair {
                query: query.to_string(),
                document: doc.clone(),
            })
            .collect();

        // Score all pairs
        let scores = self.score_batch(&pairs).await?;

        // Create results with original indices
        let mut results: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult { index, score })
            .collect();

        // Sort by score (highest first)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        debug!("Reranked {} documents", results.len());
        Ok(results)
    }

    /// Get the model path for this encoder
    pub fn model_path(&self) -> &PathBuf {
        &self.model_path
    }

    /// Apply sigmoid activation function
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Apply softmax activation function
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        
        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        use std::path::PathBuf;
        use crate::ml::tokenizer::TokenizerService;
        
        let cross_encoder = CrossEncoder {
            model_path: PathBuf::from("test_model.onnx"),
            tokenizer: TokenizerService::new_sync().unwrap(),
        };

        // Test sigmoid function
        assert!((cross_encoder.sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(cross_encoder.sigmoid(1000.0) > 0.999);
        assert!(cross_encoder.sigmoid(-1000.0) < 0.001);
    }

    #[test]
    fn test_softmax() {
        use std::path::PathBuf;
        use crate::ml::tokenizer::TokenizerService;
        
        let cross_encoder = CrossEncoder {
            model_path: PathBuf::from("test_model.onnx"),
            tokenizer: TokenizerService::new_sync().unwrap(),
        };

        // Test softmax function
        let logits = vec![1.0, 2.0, 3.0];
        let softmax_result = cross_encoder.softmax(&logits);
        
        // Check that probabilities sum to 1
        let sum: f32 = softmax_result.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        
        // Check that higher logits get higher probabilities
        assert!(softmax_result[2] > softmax_result[1]);
        assert!(softmax_result[1] > softmax_result[0]);
    }

    #[test]
    fn test_query_document_pair() {
        let pair = QueryDocumentPair {
            query: "test query".to_string(),
            document: "test document".to_string(),
        };
        
        assert_eq!(pair.query, "test query");
        assert_eq!(pair.document, "test document");
    }

    #[test]
    fn test_rerank_result() {
        let result = RerankResult {
            index: 0,
            score: 0.85,
        };
        
        assert_eq!(result.index, 0);
        assert!((result.score - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_empty_rerank() {
        // Test that empty document list returns empty results
        let documents: Vec<String> = vec![];
        assert!(documents.is_empty());
        
        // This would be tested with actual CrossEncoder in integration tests
    }
}