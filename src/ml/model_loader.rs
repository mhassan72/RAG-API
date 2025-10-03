use crate::error::{SearchError, SearchResult};
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

/// Configuration for model loading
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// GCS bucket URL for model storage
    pub gcs_base_url: String,
    /// Local directory for model cache
    pub model_cache_dir: PathBuf,
    /// Expected SHA256 hash for bi-encoder model
    pub bi_encoder_hash: String,
    /// Expected SHA256 hash for cross-encoder model
    pub cross_encoder_hash: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            gcs_base_url: "https://storage.googleapis.com/prod-models/v1".to_string(),
            model_cache_dir: PathBuf::from("./models"),
            // These would be the actual SHA256 hashes of the production models
            bi_encoder_hash: "placeholder_bi_encoder_hash".to_string(),
            cross_encoder_hash: "placeholder_cross_encoder_hash".to_string(),
        }
    }
}

/// Model loader responsible for downloading and verifying ONNX models
pub struct ModelLoader {
    config: ModelConfig,
    http_client: Client,
}

impl ModelLoader {
    /// Create a new model loader with configuration
    pub fn new(config: ModelConfig) -> SearchResult<Self> {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout for downloads
            .build()
            .map_err(|e| SearchError::ConfigError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Load bi-encoder model with verification
    /// Returns the path to the verified model file
    pub async fn load_bi_encoder(&self) -> SearchResult<PathBuf> {
        self.ensure_model_available(
            "all-MiniLM-L6-v2.onnx",
            &self.config.bi_encoder_hash,
        ).await
    }

    /// Load cross-encoder model with verification
    /// Returns the path to the verified model file
    pub async fn load_cross_encoder(&self) -> SearchResult<PathBuf> {
        self.ensure_model_available(
            "ms-marco-MiniLM-L-6-v2.onnx",
            &self.config.cross_encoder_hash,
        ).await
    }

    /// Ensure model is available locally, download if necessary
    async fn ensure_model_available(
        &self,
        model_filename: &str,
        expected_hash: &str,
    ) -> SearchResult<PathBuf> {
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&self.config.model_cache_dir)
            .await
            .map_err(|e| SearchError::IoError(e))?;

        let model_path = self.config.model_cache_dir.join(model_filename);

        // Check if model exists and has correct hash
        if model_path.exists() {
            match self.verify_model_hash(&model_path, expected_hash).await {
                Ok(true) => {
                    info!("Model {} found with correct hash", model_filename);
                    return Ok(model_path);
                }
                Ok(false) => {
                    warn!("Model {} has incorrect hash, re-downloading", model_filename);
                    fs::remove_file(&model_path)
                        .await
                        .map_err(|e| SearchError::IoError(e))?;
                }
                Err(e) => {
                    warn!("Failed to verify model hash: {}, re-downloading", e);
                    let _ = fs::remove_file(&model_path).await; // Ignore errors
                }
            }
        }

        // Download model from GCS
        self.download_model(model_filename, &model_path).await?;

        // Verify downloaded model
        if !self.verify_model_hash(&model_path, expected_hash).await? {
            fs::remove_file(&model_path)
                .await
                .map_err(|e| SearchError::IoError(e))?;
            
            return Err(SearchError::ModelError(format!(
                "Downloaded model {} has incorrect SHA256 hash. Expected: {}, service will crash to prevent using corrupted model.",
                model_filename, expected_hash
            )));
        }

        info!("Model {} downloaded and verified successfully", model_filename);
        Ok(model_path)
    }

    /// Download model from GCS
    async fn download_model(&self, model_filename: &str, local_path: &Path) -> SearchResult<()> {
        let download_url = format!("{}/{}", self.config.gcs_base_url, model_filename);
        
        info!("Downloading model from: {}", download_url);

        let response = self.http_client
            .get(&download_url)
            .send()
            .await
            .map_err(|e| SearchError::ModelError(format!("Failed to download model: {}", e)))?;

        if !response.status().is_success() {
            return Err(SearchError::ModelError(format!(
                "Failed to download model: HTTP {}",
                response.status()
            )));
        }

        let mut file = fs::File::create(local_path)
            .await
            .map_err(|e| SearchError::IoError(e))?;

        let mut stream = response.bytes_stream();
        use futures::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| SearchError::ModelError(format!("Download error: {}", e)))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| SearchError::IoError(e))?;
        }

        file.flush().await.map_err(|e| SearchError::IoError(e))?;
        info!("Model downloaded to: {}", local_path.display());

        Ok(())
    }

    /// Verify model file SHA256 hash
    async fn verify_model_hash(&self, model_path: &Path, expected_hash: &str) -> SearchResult<bool> {
        let file_content = fs::read(model_path)
            .await
            .map_err(|e| SearchError::IoError(e))?;

        let mut hasher = Sha256::new();
        hasher.update(&file_content);
        let computed_hash = hex::encode(hasher.finalize());

        Ok(computed_hash.eq_ignore_ascii_case(expected_hash))
    }

    /// Verify that a model file exists and has the correct hash
    pub async fn verify_model(&self, model_filename: &str, expected_hash: &str) -> SearchResult<bool> {
        let model_path = self.config.model_cache_dir.join(model_filename);
        
        if !model_path.exists() {
            return Ok(false);
        }
        
        self.verify_model_hash(&model_path, expected_hash).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_verify_model_hash() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");
        
        // Create a test file with known content
        let test_content = b"test model content";
        std::fs::write(&model_path, test_content).unwrap();
        
        // Calculate expected hash
        let mut hasher = Sha256::new();
        hasher.update(test_content);
        let expected_hash = hex::encode(hasher.finalize());
        
        let config = ModelConfig::default();
        let loader = ModelLoader::new(config).unwrap();
        
        // Test correct hash
        let result = loader.verify_model_hash(&model_path, &expected_hash).await.unwrap();
        assert!(result);
        
        // Test incorrect hash
        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        let result = loader.verify_model_hash(&model_path, wrong_hash).await.unwrap();
        assert!(!result);
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.gcs_base_url, "https://storage.googleapis.com/prod-models/v1");
        assert_eq!(config.model_cache_dir, PathBuf::from("./models"));
    }

    #[test]
    fn test_model_loader_creation() {
        let config = ModelConfig::default();
        let loader = ModelLoader::new(config);
        assert!(loader.is_ok());
    }
}