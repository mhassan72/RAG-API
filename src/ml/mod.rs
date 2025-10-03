/// ML inference module for ONNX models
/// 
/// This module contains:
/// - TokenizerService for text preprocessing and tokenization
/// - BiEncoder for generating embeddings using all-MiniLM-L6-v2
/// - CrossEncoder for reranking using ms-marco-MiniLM-L-6-v2
/// - ModelLoader for downloading and verifying models from GCS

pub mod tokenizer;
pub mod model_loader;
pub mod bi_encoder;
pub mod cross_encoder;

#[cfg(test)]
mod tests;

use crate::error::{SearchError, SearchResult};
pub use tokenizer::TokenizerService;
pub use model_loader::{ModelLoader, ModelConfig};
pub use bi_encoder::BiEncoder;
pub use cross_encoder::{CrossEncoder, QueryDocumentPair, RerankResult};

use std::sync::Arc;
use tracing::{info, error};

/// Complete ML service with ONNX model inference capabilities
pub struct MLService {
    bi_encoder: Arc<BiEncoder>,
    cross_encoder: Arc<CrossEncoder>,
}

impl MLService {
    /// Create a new ML service instance with model loading and verification
    pub async fn new() -> SearchResult<Self> {
        Self::new_with_config(ModelConfig::default()).await
    }

    /// Create ML service with custom model configuration
    pub async fn new_with_config(config: ModelConfig) -> SearchResult<Self> {
        info!("Initializing ML service with ONNX models...");

        // Initialize model loader
        let model_loader = ModelLoader::new(config)?;

        // Initialize tokenizer service
        let tokenizer = TokenizerService::new().await?;

        // Load bi-encoder model with SHA256 verification
        info!("Loading bi-encoder model (all-MiniLM-L6-v2)...");
        let bi_encoder_path = model_loader.load_bi_encoder().await
            .map_err(|e| {
                error!("Failed to load bi-encoder model: {}", e);
                // Crash on model verification failure as per requirements
                if e.to_string().contains("incorrect SHA256 hash") {
                    std::process::exit(1);
                }
                e
            })?;

        // Load cross-encoder model with SHA256 verification
        info!("Loading cross-encoder model (ms-marco-MiniLM-L-6-v2)...");
        let cross_encoder_path = model_loader.load_cross_encoder().await
            .map_err(|e| {
                error!("Failed to load cross-encoder model: {}", e);
                // Crash on model verification failure as per requirements
                if e.to_string().contains("incorrect SHA256 hash") {
                    std::process::exit(1);
                }
                e
            })?;

        // Create encoder services
        let bi_encoder = Arc::new(BiEncoder::new(bi_encoder_path, tokenizer.clone()));
        let cross_encoder = Arc::new(CrossEncoder::new(cross_encoder_path, tokenizer));

        info!("ML service initialized successfully");

        Ok(MLService {
            bi_encoder,
            cross_encoder,
        })
    }

    /// Generate embedding for a query using bi-encoder
    /// Returns 384-dimensional normalized vector
    pub async fn generate_embedding(&self, query: &str) -> SearchResult<Vec<f32>> {
        if query.trim().is_empty() {
            return Err(SearchError::ModelError("Empty query for embedding generation".to_string()));
        }

        self.bi_encoder.encode(query).await
    }

    /// Generate embeddings for multiple queries in batch
    pub async fn generate_embeddings_batch(&self, queries: &[String]) -> SearchResult<Vec<Vec<f32>>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }

        self.bi_encoder.encode_batch(queries).await
    }

    /// Rerank search results using cross-encoder
    /// Returns reranked results with relevance scores
    pub async fn rerank_results(
        &self,
        query: &str,
        candidates: &[String],
    ) -> SearchResult<Vec<RerankResult>> {
        if query.trim().is_empty() {
            return Err(SearchError::ModelError("Empty query for reranking".to_string()));
        }

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        self.cross_encoder.rerank(query, candidates).await
    }

    /// Score a single query-document pair using cross-encoder
    pub async fn score_pair(&self, query: &str, document: &str) -> SearchResult<f32> {
        let pair = QueryDocumentPair {
            query: query.to_string(),
            document: document.to_string(),
        };

        self.cross_encoder.score(&pair).await
    }

    /// Get reference to bi-encoder for advanced usage
    pub fn bi_encoder(&self) -> &BiEncoder {
        &self.bi_encoder
    }

    /// Get reference to cross-encoder for advanced usage
    pub fn cross_encoder(&self) -> &CrossEncoder {
        &self.cross_encoder
    }
}