use crate::error::{SearchError, SearchResult};
use farmhash;
use std::collections::HashMap;
use tokenizers::tokenizer::Tokenizer;

/// Tokenized text with all necessary components for ONNX model inference
#[derive(Debug, Clone)]
pub struct TokenizedText {
    /// Token IDs for the input text
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,
    /// Token type IDs (0 for first sequence, 1 for second sequence in pairs)
    pub token_type_ids: Vec<u32>,
}

/// TokenizerService handles text preprocessing, normalization, and tokenization
/// for semantic search queries. It provides query normalization, text cleaning,
/// and cache key generation using farmhash64.
#[derive(Clone)]
pub struct TokenizerService {
    tokenizer: Option<Tokenizer>,
}

impl TokenizerService {
    /// Create a new TokenizerService instance with async initialization
    /// This will load the actual tokenizer for ONNX model integration
    pub async fn new() -> SearchResult<Self> {
        // For now, create without tokenizer - in production this would load
        // the actual tokenizer.json file for the BERT-based models
        Ok(TokenizerService {
            tokenizer: None,
        })
    }

    /// Create a new TokenizerService instance (sync version for compatibility)
    pub fn new_sync() -> SearchResult<Self> {
        Ok(TokenizerService {
            tokenizer: None,
        })
    }

    /// Create TokenizerService with a specific tokenizer
    /// This will be used when we integrate with actual ONNX models
    pub fn with_tokenizer(tokenizer: Tokenizer) -> Self {
        TokenizerService {
            tokenizer: Some(tokenizer),
        }
    }

    /// Normalize and clean query text for consistent processing
    /// 
    /// This function:
    /// - Trims whitespace
    /// - Converts to lowercase
    /// - Removes excessive whitespace (multiple spaces/tabs/newlines)
    /// - Removes control characters
    /// - Handles Unicode normalization
    pub fn normalize_query(&self, query: &str) -> String {
        // Trim leading/trailing whitespace
        let mut normalized = query.trim().to_string();
        
        // Convert to lowercase for consistency
        normalized = normalized.to_lowercase();
        
        // Remove control characters (except space, tab, newline)
        normalized = normalized
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect();
        
        // Normalize whitespace - replace multiple whitespace chars with single space
        let mut result = String::new();
        let mut prev_was_space = false;
        
        for ch in normalized.chars() {
            if ch.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(ch);
                prev_was_space = false;
            }
        }
        
        // Final trim to remove any trailing space
        result.trim().to_string()
    }

    /// Clean text by removing unwanted characters and normalizing content
    /// 
    /// This is more aggressive than normalize_query and is used for
    /// preprocessing text content before tokenization.
    pub fn clean_text(&self, text: &str) -> String {
        let normalized = self.normalize_query(text);
        
        // Remove common punctuation that doesn't add semantic value
        let cleaned = normalized
            .chars()
            .filter(|c| {
                // Keep alphanumeric, basic punctuation, and whitespace
                c.is_alphanumeric() 
                    || c.is_whitespace()
                    || matches!(*c, '.' | ',' | '!' | '?' | ':' | ';' | '-' | '_' | '\'' | '"')
            })
            .collect::<String>();
        
        // Normalize whitespace again after character filtering
        self.normalize_whitespace(&cleaned)
    }

    /// Generate a cache key for a query using farmhash64
    /// 
    /// The cache key is generated from the normalized query to ensure
    /// that semantically identical queries (with different formatting)
    /// produce the same cache key.
    pub fn generate_cache_key(&self, query: &str) -> u64 {
        let normalized = self.normalize_query(query);
        farmhash::hash64(normalized.as_bytes())
    }

    /// Generate cache key with additional parameters
    /// 
    /// This creates a cache key that includes query parameters like k, min_score,
    /// and filters to ensure different search configurations are cached separately.
    pub fn generate_cache_key_with_params(
        &self,
        query: &str,
        k: u32,
        min_score: Option<f32>,
        filters: &HashMap<String, String>,
    ) -> u64 {
        let normalized_query = self.normalize_query(query);
        
        // Create a deterministic string representation of all parameters
        let mut key_parts = vec![normalized_query];
        key_parts.push(k.to_string());
        
        if let Some(score) = min_score {
            key_parts.push(format!("score:{:.3}", score));
        }
        
        // Sort filters for deterministic key generation
        let mut filter_pairs: Vec<_> = filters.iter().collect();
        filter_pairs.sort_by_key(|(k, _)| *k);
        
        for (key, value) in filter_pairs {
            key_parts.push(format!("{}:{}", key, value));
        }
        
        let combined = key_parts.join("|");
        farmhash::hash64(combined.as_bytes())
    }

    /// Tokenize text using the loaded tokenizer
    /// 
    /// Returns TokenizedText with input_ids, attention_mask, and token_type_ids
    /// that can be used for ONNX model inference.
    pub fn tokenize(&self, text: &str) -> SearchResult<TokenizedText> {
        match &self.tokenizer {
            Some(tokenizer) => {
                let cleaned_text = self.clean_text(text);
                
                let encoding = tokenizer
                    .encode(cleaned_text, false)
                    .map_err(|e| SearchError::ModelError(format!("Tokenization failed: {}", e)))?;
                
                let input_ids = encoding.get_ids().to_vec();
                let attention_mask = encoding.get_attention_mask().to_vec();
                let token_type_ids = encoding.get_type_ids().to_vec();
                
                Ok(TokenizedText {
                    input_ids,
                    attention_mask,
                    token_type_ids,
                })
            }
            None => {
                // For now, return an error since we don't have a tokenizer loaded
                // This will be implemented when we add ONNX model loading
                Err(SearchError::ModelError(
                    "Tokenizer not loaded - will be implemented with ONNX model integration".to_string()
                ))
            }
        }
    }

    /// Tokenize text and return only token IDs (legacy method)
    pub fn tokenize_ids(&self, text: &str) -> SearchResult<Vec<u32>> {
        let tokenized = self.tokenize(text)?;
        Ok(tokenized.input_ids)
    }

    /// Get the vocabulary size of the loaded tokenizer
    pub fn vocab_size(&self) -> SearchResult<usize> {
        match &self.tokenizer {
            Some(tokenizer) => Ok(tokenizer.get_vocab_size(false)),
            None => Err(SearchError::ModelError("Tokenizer not loaded".to_string())),
        }
    }

    /// Validate query text for common issues
    /// 
    /// Returns validation errors for:
    /// - Empty queries
    /// - Queries that are too long
    /// - Queries with only whitespace/punctuation
    pub fn validate_query(&self, query: &str) -> SearchResult<()> {
        if query.trim().is_empty() {
            return Err(SearchError::InvalidRequest("Query cannot be empty".to_string()));
        }

        if query.len() > 1000 {
            return Err(SearchError::InvalidRequest(
                "Query too long (max 1000 characters)".to_string()
            ));
        }

        let normalized = self.normalize_query(query);
        if normalized.is_empty() {
            return Err(SearchError::InvalidRequest(
                "Query contains no valid text content".to_string()
            ));
        }

        // Check if query has at least some alphanumeric content
        if !normalized.chars().any(|c| c.is_alphanumeric()) {
            return Err(SearchError::InvalidRequest(
                "Query must contain at least some alphanumeric characters".to_string()
            ));
        }

        Ok(())
    }

    /// Helper function to normalize whitespace in text
    fn normalize_whitespace(&self, text: &str) -> String {
        let mut result = String::new();
        let mut prev_was_space = false;
        
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(ch);
                prev_was_space = false;
            }
        }
        
        result.trim().to_string()
    }
}

impl Default for TokenizerService {
    fn default() -> Self {
        Self::new_sync().expect("Failed to create default TokenizerService")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_normalize_query_basic() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Basic normalization
        assert_eq!(tokenizer.normalize_query("  Hello World  "), "hello world");
        assert_eq!(tokenizer.normalize_query("HELLO WORLD"), "hello world");
        assert_eq!(tokenizer.normalize_query("hello\tworld"), "hello world");
        assert_eq!(tokenizer.normalize_query("hello\nworld"), "hello world");
    }

    #[test]
    fn test_normalize_query_whitespace() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Multiple whitespace normalization
        assert_eq!(tokenizer.normalize_query("hello    world"), "hello world");
        assert_eq!(tokenizer.normalize_query("hello\t\t\tworld"), "hello world");
        assert_eq!(tokenizer.normalize_query("hello\n\n\nworld"), "hello world");
        assert_eq!(tokenizer.normalize_query("  hello   world  "), "hello world");
    }

    #[test]
    fn test_normalize_query_control_characters() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Control character removal (except whitespace)
        let input_with_controls = "hello\x00\x01world\x7f";
        assert_eq!(tokenizer.normalize_query(input_with_controls), "helloworld");
        
        // Preserve normal whitespace
        let input_with_whitespace = "hello\tworld\ntest";
        assert_eq!(tokenizer.normalize_query(input_with_whitespace), "hello world test");
    }

    #[test]
    fn test_clean_text() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Basic cleaning
        assert_eq!(tokenizer.clean_text("Hello, World!"), "hello, world!");
        
        // Remove unwanted characters
        let messy_text = "Hello@#$%^&*()World{}[]|\\+=<>";
        assert_eq!(tokenizer.clean_text(messy_text), "helloworld");
        
        // Keep allowed punctuation
        let good_punctuation = "Hello, world! How are you? I'm fine.";
        assert_eq!(tokenizer.clean_text(good_punctuation), "hello, world! how are you? i'm fine.");
    }

    #[test]
    fn test_generate_cache_key() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Same normalized queries should produce same keys
        let key1 = tokenizer.generate_cache_key("  Hello World  ");
        let key2 = tokenizer.generate_cache_key("hello world");
        let key3 = tokenizer.generate_cache_key("HELLO\tWORLD");
        
        assert_eq!(key1, key2);
        assert_eq!(key2, key3);
        
        // Different queries should produce different keys
        let key4 = tokenizer.generate_cache_key("hello universe");
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_generate_cache_key_with_params() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        let mut filters1 = HashMap::new();
        filters1.insert("language".to_string(), "en".to_string());
        filters1.insert("frozen".to_string(), "false".to_string());
        
        let mut filters2 = HashMap::new();
        filters2.insert("frozen".to_string(), "false".to_string());
        filters2.insert("language".to_string(), "en".to_string());
        
        // Same parameters in different order should produce same key
        let key1 = tokenizer.generate_cache_key_with_params("hello world", 10, Some(0.5), &filters1);
        let key2 = tokenizer.generate_cache_key_with_params("hello world", 10, Some(0.5), &filters2);
        assert_eq!(key1, key2);
        
        // Different parameters should produce different keys
        let key3 = tokenizer.generate_cache_key_with_params("hello world", 20, Some(0.5), &filters1);
        assert_ne!(key1, key3);
        
        let key4 = tokenizer.generate_cache_key_with_params("hello world", 10, Some(0.7), &filters1);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_validate_query() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Valid queries
        assert!(tokenizer.validate_query("hello world").is_ok());
        assert!(tokenizer.validate_query("What is machine learning?").is_ok());
        assert!(tokenizer.validate_query("123 test").is_ok());
        
        // Invalid queries
        assert!(tokenizer.validate_query("").is_err());
        assert!(tokenizer.validate_query("   ").is_err());
        assert!(tokenizer.validate_query("!!!???").is_err());
        assert!(tokenizer.validate_query("   \t\n   ").is_err());
        
        // Too long query
        let long_query = "a".repeat(1001);
        assert!(tokenizer.validate_query(&long_query).is_err());
    }

    #[test]
    fn test_normalize_whitespace_helper() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        assert_eq!(tokenizer.normalize_whitespace("hello    world"), "hello world");
        assert_eq!(tokenizer.normalize_whitespace("  hello world  "), "hello world");
        assert_eq!(tokenizer.normalize_whitespace("hello\t\tworld"), "hello world");
        assert_eq!(tokenizer.normalize_whitespace(""), "");
        assert_eq!(tokenizer.normalize_whitespace("   "), "");
    }

    #[test]
    fn test_edge_cases() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Empty string
        assert_eq!(tokenizer.normalize_query(""), "");
        assert_eq!(tokenizer.clean_text(""), "");
        
        // Only whitespace
        assert_eq!(tokenizer.normalize_query("   \t\n   "), "");
        
        // Unicode characters
        assert_eq!(tokenizer.normalize_query("Héllo Wörld"), "héllo wörld");
        
        // Numbers and special characters
        assert_eq!(tokenizer.normalize_query("Test 123 & More"), "test 123 & more");
        assert_eq!(tokenizer.clean_text("Test 123 & More"), "test 123 more");
    }

    #[test]
    fn test_tokenize_without_tokenizer() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Should return error since no tokenizer is loaded
        let result = tokenizer.tokenize("hello world");
        assert!(result.is_err());
        assert!(result.unwrap_err().is_model_error());
    }

    #[test]
    fn test_vocab_size_without_tokenizer() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Should return error since no tokenizer is loaded
        let result = tokenizer.vocab_size();
        assert!(result.is_err());
        assert!(result.unwrap_err().is_model_error());
    }

    #[test]
    fn test_cache_key_consistency() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Test that cache keys are consistent across multiple calls
        let query = "machine learning algorithms";
        let key1 = tokenizer.generate_cache_key(query);
        let key2 = tokenizer.generate_cache_key(query);
        assert_eq!(key1, key2);
        
        // Test with parameters
        let filters = HashMap::new();
        let param_key1 = tokenizer.generate_cache_key_with_params(query, 10, None, &filters);
        let param_key2 = tokenizer.generate_cache_key_with_params(query, 10, None, &filters);
        assert_eq!(param_key1, param_key2);
    }

    #[test]
    fn test_query_normalization_edge_cases() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Test with various Unicode characters
        assert_eq!(tokenizer.normalize_query("café résumé naïve"), "café résumé naïve");
        
        // Test with emojis (should be preserved as they're not control chars)
        assert_eq!(tokenizer.normalize_query("hello 😀 world"), "hello 😀 world");
        
        // Test with mixed case and punctuation
        assert_eq!(tokenizer.normalize_query("Hello, WORLD! How Are You?"), "hello, world! how are you?");
        
        // Test with tabs and newlines mixed
        assert_eq!(tokenizer.normalize_query("line1\n\tline2\r\nline3"), "line1 line2 line3");
    }

    #[test]
    fn test_text_cleaning_comprehensive() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Test removal of various unwanted characters
        let messy_input = "Hello@#$%World!!! This~is*a&test(){}[]|\\+=<>?/";
        let expected = "helloworld!!! thisisatest?";
        assert_eq!(tokenizer.clean_text(messy_input), expected);
        
        // Test preservation of allowed punctuation
        let good_input = "Dr. Smith's book: \"AI & ML\" - A comprehensive guide.";
        let expected = "dr. smith's book: \"ai ml\" - a comprehensive guide.";
        assert_eq!(tokenizer.clean_text(good_input), expected);
    }

    #[test]
    fn test_cache_key_parameter_variations() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        let base_query = "test query";
        let mut filters = HashMap::new();
        filters.insert("lang".to_string(), "en".to_string());
        
        // Different k values should produce different keys
        let key1 = tokenizer.generate_cache_key_with_params(base_query, 10, None, &filters);
        let key2 = tokenizer.generate_cache_key_with_params(base_query, 20, None, &filters);
        assert_ne!(key1, key2);
        
        // Different min_score values should produce different keys
        let key3 = tokenizer.generate_cache_key_with_params(base_query, 10, Some(0.5), &filters);
        let key4 = tokenizer.generate_cache_key_with_params(base_query, 10, Some(0.7), &filters);
        assert_ne!(key3, key4);
        
        // Different filters should produce different keys
        let mut filters2 = HashMap::new();
        filters2.insert("lang".to_string(), "es".to_string());
        let key5 = tokenizer.generate_cache_key_with_params(base_query, 10, None, &filters2);
        assert_ne!(key1, key5);
    }

    #[test]
    fn test_validation_comprehensive() {
        let tokenizer = TokenizerService::new_sync().unwrap();
        
        // Test various invalid inputs
        assert!(tokenizer.validate_query("").is_err());
        assert!(tokenizer.validate_query("   \t\n   ").is_err());
        assert!(tokenizer.validate_query("!@#$%^&*()").is_err());
        assert!(tokenizer.validate_query("???!!!").is_err());
        
        // Test boundary cases
        let exactly_1000_chars = "a".repeat(1000);
        assert!(tokenizer.validate_query(&exactly_1000_chars).is_ok());
        
        let over_1000_chars = "a".repeat(1001);
        assert!(tokenizer.validate_query(&over_1000_chars).is_err());
        
        // Test minimal valid queries
        assert!(tokenizer.validate_query("a").is_ok());
        assert!(tokenizer.validate_query("1").is_ok());
        assert!(tokenizer.validate_query("a?").is_ok());
        assert!(tokenizer.validate_query("hello!").is_ok());
    }
}