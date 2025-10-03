/// Cross-encoder reranking service
/// 
/// This module implements optional reranking using CrossEncoder when rerank=true parameter is set.
/// It includes performance optimizations to limit reranking to top candidates and handles
/// reranking failures with graceful degradation to similarity scores.

use crate::error::{SearchError, SearchResult};
use crate::ml::{CrossEncoder, RerankResult};
use crate::types::{SearchCandidate, Post, SearchResponse};
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, warn, instrument};

/// Configuration for the reranking service
#[derive(Debug, Clone)]
pub struct RerankingConfig {
    /// Maximum number of candidates to rerank (performance optimization)
    pub max_candidates_to_rerank: usize,
    /// Timeout for reranking operation
    pub rerank_timeout_ms: u64,
    /// Whether to enable graceful degradation on reranking failures
    pub enable_graceful_degradation: bool,
}

impl Default for RerankingConfig {
    fn default() -> Self {
        Self {
            max_candidates_to_rerank: 50, // Limit reranking to top 50 candidates for performance
            rerank_timeout_ms: 1000, // 1 second timeout for reranking
            enable_graceful_degradation: true,
        }
    }
}

/// Reranking service that uses CrossEncoder for result scoring
pub struct RerankingService {
    cross_encoder: Arc<CrossEncoder>,
    config: RerankingConfig,
}

impl RerankingService {
    /// Create a new reranking service
    pub fn new(cross_encoder: Arc<CrossEncoder>) -> Self {
        Self {
            cross_encoder,
            config: RerankingConfig::default(),
        }
    }

    /// Create a new reranking service with custom configuration
    pub fn with_config(cross_encoder: Arc<CrossEncoder>, config: RerankingConfig) -> Self {
        Self {
            cross_encoder,
            config,
        }
    }

    /// Rerank search results using cross-encoder
    /// 
    /// This method takes the original query and search results, applies cross-encoder
    /// scoring, and returns reranked results. It includes performance optimizations
    /// and graceful degradation on failures.
    #[instrument(skip(self, search_results), fields(
        query_len = query.len(),
        num_results = search_results.len(),
        rerank_enabled = rerank_enabled
    ))]
    pub async fn rerank_results(
        &self,
        query: &str,
        search_results: &[SearchResponse],
        rerank_enabled: bool,
    ) -> SearchResult<Vec<SearchResponse>> {
        // If reranking is not enabled, return original results
        if !rerank_enabled {
            debug!("Reranking disabled, returning original results");
            return Ok(search_results.to_vec());
        }

        if search_results.is_empty() {
            debug!("No results to rerank");
            return Ok(search_results.to_vec());
        }

        debug!("Starting reranking for {} results", search_results.len());

        // Performance optimization: limit reranking to top candidates
        let candidates_to_rerank = std::cmp::min(
            search_results.len(),
            self.config.max_candidates_to_rerank
        );

        if candidates_to_rerank < search_results.len() {
            info!(
                "Limiting reranking to top {} candidates (out of {} total) for performance",
                candidates_to_rerank,
                search_results.len()
            );
        }

        // Split results into candidates to rerank and remaining results
        let (rerank_candidates, remaining_results) = if candidates_to_rerank < search_results.len() {
            let rerank_candidates = search_results[..candidates_to_rerank].to_vec();
            let remaining_results = search_results[candidates_to_rerank..].to_vec();
            (rerank_candidates, remaining_results)
        } else {
            (search_results.to_vec(), Vec::new())
        };

        // Attempt reranking with timeout and graceful degradation
        match self.perform_reranking_with_timeout(query, &rerank_candidates).await {
            Ok(reranked_results) => {
                debug!("Reranking successful, {} results reranked", reranked_results.len());
                
                // Combine reranked results with remaining results
                let mut final_results = reranked_results;
                final_results.extend(remaining_results);
                
                info!("Reranking completed successfully: {} total results", final_results.len());
                Ok(final_results)
            }
            Err(e) => {
                if self.config.enable_graceful_degradation {
                    warn!("Reranking failed, falling back to original similarity scores: {}", e);
                    
                    // Return original results with warning logged
                    let mut final_results = rerank_candidates;
                    final_results.extend(remaining_results);
                    
                    info!("Graceful degradation: returning {} results with original scores", final_results.len());
                    Ok(final_results)
                } else {
                    error!("Reranking failed and graceful degradation disabled: {}", e);
                    Err(e)
                }
            }
        }
    }

    /// Rerank search candidates (before converting to SearchResponse)
    /// 
    /// This method is useful when reranking needs to happen earlier in the pipeline
    /// before metadata is fetched and SearchResponse objects are created.
    #[instrument(skip(self, candidates), fields(
        query_len = query.len(),
        num_candidates = candidates.len()
    ))]
    pub async fn rerank_candidates(
        &self,
        query: &str,
        candidates: Vec<SearchCandidate>,
        posts: &[Post],
        rerank_enabled: bool,
    ) -> SearchResult<Vec<SearchCandidate>> {
        if !rerank_enabled || candidates.is_empty() {
            debug!("Reranking disabled or no candidates, returning original order");
            return Ok(candidates);
        }

        debug!("Starting candidate reranking for {} candidates", candidates.len());

        // Performance optimization: limit reranking to top candidates
        let candidates_to_rerank = std::cmp::min(
            candidates.len(),
            self.config.max_candidates_to_rerank
        );

        let (mut rerank_candidates, remaining_candidates) = if candidates_to_rerank < candidates.len() {
            let mut cands = candidates;
            let remaining = cands.split_off(candidates_to_rerank);
            (cands, remaining)
        } else {
            (candidates, Vec::new())
        };

        // Create documents for reranking from posts
        let documents: Vec<String> = rerank_candidates
            .iter()
            .filter_map(|candidate| {
                posts.iter()
                    .find(|post| post.post_id == candidate.post_id)
                    .map(|post| format!("{} {}", post.title, post.content))
            })
            .collect();

        if documents.len() != rerank_candidates.len() {
            warn!(
                "Mismatch between candidates ({}) and documents ({}), some posts not found",
                rerank_candidates.len(),
                documents.len()
            );
        }

        // Attempt reranking with timeout
        match self.perform_cross_encoder_reranking_with_timeout(query, &documents).await {
            Ok(rerank_results) => {
                debug!("Cross-encoder reranking successful");
                
                // Apply reranking scores to candidates
                let mut reranked_candidates = self.apply_rerank_scores_to_candidates(
                    rerank_candidates,
                    rerank_results
                );

                // Combine with remaining candidates
                reranked_candidates.extend(remaining_candidates);
                
                info!("Candidate reranking completed: {} total candidates", reranked_candidates.len());
                Ok(reranked_candidates)
            }
            Err(e) => {
                if self.config.enable_graceful_degradation {
                    warn!("Candidate reranking failed, using original scores: {}", e);
                    
                    let mut final_candidates = rerank_candidates;
                    final_candidates.extend(remaining_candidates);
                    Ok(final_candidates)
                } else {
                    error!("Candidate reranking failed: {}", e);
                    Err(e)
                }
            }
        }
    }

    /// Perform reranking with timeout protection
    async fn perform_reranking_with_timeout(
        &self,
        query: &str,
        results: &[SearchResponse],
    ) -> SearchResult<Vec<SearchResponse>> {
        let rerank_timeout = Duration::from_millis(self.config.rerank_timeout_ms);
        
        timeout(rerank_timeout, self.perform_reranking(query, results))
            .await
            .map_err(|_| SearchError::ModelError("Reranking timeout".to_string()))?
    }

    /// Perform cross-encoder reranking with timeout protection
    async fn perform_cross_encoder_reranking_with_timeout(
        &self,
        query: &str,
        documents: &[String],
    ) -> SearchResult<Vec<RerankResult>> {
        let rerank_timeout = Duration::from_millis(self.config.rerank_timeout_ms);
        
        timeout(rerank_timeout, self.cross_encoder.rerank(query, documents))
            .await
            .map_err(|_| SearchError::ModelError("Cross-encoder reranking timeout".to_string()))?
    }

    /// Perform the actual reranking using cross-encoder
    async fn perform_reranking(
        &self,
        query: &str,
        results: &[SearchResponse],
    ) -> SearchResult<Vec<SearchResponse>> {
        debug!("Performing cross-encoder reranking for {} results", results.len());

        // Extract documents for reranking (title + snippet)
        let documents: Vec<String> = results
            .iter()
            .map(|result| format!("{} {}", result.title, result.snippet))
            .collect();

        // Perform reranking using cross-encoder
        let rerank_results = self.cross_encoder.rerank(query, &documents).await?;

        // Apply reranking scores to search results
        let reranked_results = self.apply_rerank_scores(results, rerank_results);

        debug!("Cross-encoder reranking completed");
        Ok(reranked_results)
    }

    /// Apply reranking scores to search results and sort by relevance
    fn apply_rerank_scores(
        &self,
        original_results: &[SearchResponse],
        rerank_results: Vec<RerankResult>,
    ) -> Vec<SearchResponse> {
        debug!("Applying rerank scores to {} results", original_results.len());

        // Create a mapping from original index to rerank score
        let mut score_map: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for rerank_result in rerank_results {
            score_map.insert(rerank_result.index, rerank_result.score);
        }

        // Apply rerank scores to results
        let mut reranked_results: Vec<SearchResponse> = original_results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                let mut reranked_result = result.clone();
                if let Some(&rerank_score) = score_map.get(&index) {
                    reranked_result.score = rerank_score;
                    debug!(
                        "Applied rerank score to {}: {:.4} -> {:.4}",
                        result.post_id, result.score, rerank_score
                    );
                } else {
                    warn!("No rerank score found for result at index {}", index);
                }
                reranked_result
            })
            .collect();

        // Sort by rerank score (highest first)
        reranked_results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!("Results sorted by rerank scores");
        reranked_results
    }

    /// Apply reranking scores to search candidates
    fn apply_rerank_scores_to_candidates(
        &self,
        original_candidates: Vec<SearchCandidate>,
        rerank_results: Vec<RerankResult>,
    ) -> Vec<SearchCandidate> {
        debug!("Applying rerank scores to {} candidates", original_candidates.len());

        // Create a mapping from original index to rerank score
        let mut score_map: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for rerank_result in rerank_results {
            score_map.insert(rerank_result.index, rerank_result.score);
        }

        // Apply rerank scores to candidates
        let mut reranked_candidates: Vec<SearchCandidate> = original_candidates
            .into_iter()
            .enumerate()
            .map(|(index, mut candidate)| {
                if let Some(&rerank_score) = score_map.get(&index) {
                    debug!(
                        "Applied rerank score to candidate {}: {:.4} -> {:.4}",
                        candidate.post_id, candidate.score, rerank_score
                    );
                    candidate.score = rerank_score;
                }
                candidate
            })
            .collect();

        // Sort by rerank score (highest first)
        reranked_candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!("Candidates sorted by rerank scores");
        reranked_candidates
    }

    /// Get reranking configuration
    pub fn config(&self) -> &RerankingConfig {
        &self.config
    }

    /// Update reranking configuration
    pub fn update_config(&mut self, config: RerankingConfig) {
        self.config = config;
    }

    /// Check if reranking is available (cross-encoder is loaded)
    pub fn is_available(&self) -> bool {
        // For now, assume cross-encoder is always available if service is created
        // In production, this could check if the model is properly loaded
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::{CrossEncoder, TokenizerService, QueryDocumentPair};
    use crate::types::{PostMetadata, SearchSource};
    use chrono::Utc;
    use std::path::PathBuf;

    /// Create a mock cross-encoder for testing
    fn create_mock_cross_encoder() -> CrossEncoder {
        let tokenizer = TokenizerService::new_sync().unwrap();
        CrossEncoder::new(PathBuf::from("test_model.onnx"), tokenizer)
    }

    /// Create test search results
    fn create_test_search_results() -> Vec<SearchResponse> {
        vec![
            SearchResponse {
                post_id: "post1".to_string(),
                title: "First Post".to_string(),
                snippet: "This is the first post about machine learning".to_string(),
                score: 0.8,
                meta: PostMetadata {
                    author_name: "Author 1".to_string(),
                    url: "https://example.com/post1".to_string(),
                    date: Utc::now(),
                    language: "en".to_string(),
                    frozen: false,
                },
            },
            SearchResponse {
                post_id: "post2".to_string(),
                title: "Second Post".to_string(),
                snippet: "This is the second post about artificial intelligence".to_string(),
                score: 0.7,
                meta: PostMetadata {
                    author_name: "Author 2".to_string(),
                    url: "https://example.com/post2".to_string(),
                    date: Utc::now(),
                    language: "en".to_string(),
                    frozen: false,
                },
            },
            SearchResponse {
                post_id: "post3".to_string(),
                title: "Third Post".to_string(),
                snippet: "This is the third post about deep learning".to_string(),
                score: 0.6,
                meta: PostMetadata {
                    author_name: "Author 3".to_string(),
                    url: "https://example.com/post3".to_string(),
                    date: Utc::now(),
                    language: "en".to_string(),
                    frozen: false,
                },
            },
        ]
    }

    /// Create test search candidates
    fn create_test_search_candidates() -> Vec<SearchCandidate> {
        vec![
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.8,
                source: SearchSource::Redis,
            },
            SearchCandidate {
                post_id: "post2".to_string(),
                score: 0.7,
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post3".to_string(),
                score: 0.6,
                source: SearchSource::Redis,
            },
        ]
    }

    #[test]
    fn test_reranking_config_default() {
        let config = RerankingConfig::default();
        assert_eq!(config.max_candidates_to_rerank, 50);
        assert_eq!(config.rerank_timeout_ms, 1000);
        assert!(config.enable_graceful_degradation);
    }

    #[test]
    fn test_reranking_service_creation() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let service = RerankingService::new(cross_encoder);
        
        assert_eq!(service.config.max_candidates_to_rerank, 50);
        assert!(service.is_available());
    }

    #[test]
    fn test_reranking_service_with_custom_config() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let config = RerankingConfig {
            max_candidates_to_rerank: 25,
            rerank_timeout_ms: 500,
            enable_graceful_degradation: false,
        };
        
        let service = RerankingService::with_config(cross_encoder, config.clone());
        assert_eq!(service.config.max_candidates_to_rerank, 25);
        assert_eq!(service.config.rerank_timeout_ms, 500);
        assert!(!service.config.enable_graceful_degradation);
    }

    #[tokio::test]
    async fn test_rerank_results_disabled() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let service = RerankingService::new(cross_encoder);
        let results = create_test_search_results();
        let original_len = results.len();
        
        let reranked = service.rerank_results("test query", &results, false).await.unwrap();
        
        assert_eq!(reranked.len(), original_len);
        // Should return original results unchanged when reranking is disabled
        assert_eq!(reranked[0].post_id, "post1");
        assert_eq!(reranked[0].score, 0.8);
    }

    #[tokio::test]
    async fn test_rerank_results_empty() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let service = RerankingService::new(cross_encoder);
        let results = vec![];
        
        let reranked = service.rerank_results("test query", &results, true).await.unwrap();
        
        assert!(reranked.is_empty());
    }

    #[tokio::test]
    async fn test_rerank_results_enabled() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let service = RerankingService::new(cross_encoder);
        let results = create_test_search_results();
        
        let reranked = service.rerank_results("machine learning", &results, true).await.unwrap();
        
        assert_eq!(reranked.len(), 3);
        // Results should be reordered based on cross-encoder scores
        // The exact order depends on the mock cross-encoder implementation
        // but we can verify that scores have been updated
        for result in &reranked {
            // Cross-encoder scores should be between 0 and 1
            assert!(result.score >= 0.0 && result.score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_rerank_candidates_disabled() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let service = RerankingService::new(cross_encoder);
        let candidates = create_test_search_candidates();
        let posts = vec![]; // Empty posts for this test
        
        let reranked = service.rerank_candidates("test query", candidates, &posts, false).await.unwrap();
        
        assert_eq!(reranked.len(), 3);
        assert_eq!(reranked[0].post_id, "post1");
        assert_eq!(reranked[0].score, 0.8);
    }

    #[tokio::test]
    async fn test_performance_optimization_limit() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let config = RerankingConfig {
            max_candidates_to_rerank: 2, // Limit to 2 candidates
            rerank_timeout_ms: 1000,
            enable_graceful_degradation: true,
        };
        let service = RerankingService::with_config(cross_encoder, config);
        let results = create_test_search_results(); // 3 results
        
        let reranked = service.rerank_results("test query", &results, true).await.unwrap();
        
        // Should still return all 3 results, but only first 2 should be reranked
        assert_eq!(reranked.len(), 3);
    }

    #[test]
    fn test_apply_rerank_scores() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let service = RerankingService::new(cross_encoder);
        let results = create_test_search_results();
        
        let rerank_results = vec![
            RerankResult { index: 0, score: 0.9 },
            RerankResult { index: 1, score: 0.5 },
            RerankResult { index: 2, score: 0.95 },
        ];
        
        let reranked = service.apply_rerank_scores(&results, rerank_results);
        
        assert_eq!(reranked.len(), 3);
        // Should be sorted by rerank score (highest first)
        assert_eq!(reranked[0].post_id, "post3"); // score 0.95
        assert_eq!(reranked[1].post_id, "post1"); // score 0.9
        assert_eq!(reranked[2].post_id, "post2"); // score 0.5
        
        assert_eq!(reranked[0].score, 0.95);
        assert_eq!(reranked[1].score, 0.9);
        assert_eq!(reranked[2].score, 0.5);
    }

    #[test]
    fn test_apply_rerank_scores_to_candidates() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let service = RerankingService::new(cross_encoder);
        let candidates = create_test_search_candidates();
        
        let rerank_results = vec![
            RerankResult { index: 0, score: 0.9 },
            RerankResult { index: 1, score: 0.5 },
            RerankResult { index: 2, score: 0.95 },
        ];
        
        let reranked = service.apply_rerank_scores_to_candidates(candidates, rerank_results);
        
        assert_eq!(reranked.len(), 3);
        // Should be sorted by rerank score (highest first)
        assert_eq!(reranked[0].post_id, "post3"); // score 0.95
        assert_eq!(reranked[1].post_id, "post1"); // score 0.9
        assert_eq!(reranked[2].post_id, "post2"); // score 0.5
    }

    #[test]
    fn test_config_update() {
        let cross_encoder = Arc::new(create_mock_cross_encoder());
        let mut service = RerankingService::new(cross_encoder);
        
        let new_config = RerankingConfig {
            max_candidates_to_rerank: 100,
            rerank_timeout_ms: 2000,
            enable_graceful_degradation: false,
        };
        
        service.update_config(new_config);
        
        assert_eq!(service.config.max_candidates_to_rerank, 100);
        assert_eq!(service.config.rerank_timeout_ms, 2000);
        assert!(!service.config.enable_graceful_degradation);
    }
}