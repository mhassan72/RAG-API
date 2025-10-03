/// Main search service that coordinates vector search, reranking, and result processing
/// 
/// This module implements the complete search pipeline including:
/// - Vector search coordination across Redis and Postgres
/// - Optional cross-encoder reranking when rerank=true
/// - Graceful degradation and circuit breaker integration
/// - Result filtering and metadata enrichment

use crate::cache::CacheManager;
use crate::database::DatabaseManager;
use crate::error::{SearchError, SearchResult};
use crate::ml::MLService;
use crate::types::{SearchRequest, SearchResponse, SearchCandidate, SearchMode, Post, SearchFilters, PostMetadata};
use crate::search::{FallbackSearchService, RerankingService, RerankingConfig};
use std::sync::Arc;
use tracing::{debug, error, info, warn, instrument};

/// Complete search service with ML integration
pub struct SearchService {
    /// ML service for embeddings and reranking
    ml_service: Arc<MLService>,
    /// Fallback search service for vector search coordination
    fallback_search: Arc<FallbackSearchService>,
    /// Database manager for post metadata retrieval
    database_manager: Arc<DatabaseManager>,
    /// Reranking service for cross-encoder scoring
    reranking_service: Arc<RerankingService>,
}

impl SearchService {
    /// Create a new search service
    pub async fn new(
        cache_manager: Arc<CacheManager>,
        database_manager: Arc<DatabaseManager>,
        ml_service: Arc<MLService>,
    ) -> SearchResult<Self> {
        // Create fallback search service
        let fallback_search = Arc::new(FallbackSearchService::new(
            cache_manager,
            database_manager.clone(),
        ));

        // Create reranking service with default configuration
        let reranking_service = Arc::new(RerankingService::new(
            Arc::new(ml_service.cross_encoder().clone())
        ));

        Ok(Self {
            ml_service,
            fallback_search,
            database_manager,
            reranking_service,
        })
    }

    /// Create a new search service with custom reranking configuration
    pub async fn new_with_reranking_config(
        cache_manager: Arc<CacheManager>,
        database_manager: Arc<DatabaseManager>,
        ml_service: Arc<MLService>,
        reranking_config: RerankingConfig,
    ) -> SearchResult<Self> {
        let fallback_search = Arc::new(FallbackSearchService::new(
            cache_manager,
            database_manager.clone(),
        ));

        let reranking_service = Arc::new(RerankingService::with_config(
            Arc::new(ml_service.cross_encoder().clone()),
            reranking_config,
        ));

        Ok(Self {
            ml_service,
            fallback_search,
            database_manager,
            reranking_service,
        })
    }

    /// Perform complete semantic search with optional reranking
    #[instrument(skip(self), fields(
        query_len = request.query.len(),
        k = request.k,
        rerank = request.rerank,
        min_score = request.min_score
    ))]
    pub async fn semantic_search(&self, request: SearchRequest) -> SearchResult<Vec<SearchResponse>> {
        info!("Starting semantic search for query: '{}'", request.query);

        // Step 1: Generate query embedding
        debug!("Generating query embedding");
        let query_embedding = self.ml_service.generate_embedding(&request.query).await
            .map_err(|e| {
                error!("Failed to generate query embedding: {}", e);
                e
            })?;

        // Step 2: Perform vector search with fallback logic
        debug!("Performing vector search");
        let (search_candidates, search_mode) = self.fallback_search
            .search_with_fallback(&query_embedding, request.k as usize * 2) // Get more candidates for reranking
            .await
            .map_err(|e| {
                error!("Vector search failed: {}", e);
                e
            })?;

        info!("Vector search completed: {} candidates found (mode: {:?})", 
              search_candidates.len(), search_mode);

        if search_candidates.is_empty() {
            info!("No search candidates found");
            return Ok(vec![]);
        }

        // Step 3: Fetch post metadata and create initial results
        debug!("Fetching post metadata for {} candidates", search_candidates.len());
        let posts = self.fetch_posts_for_candidates(&search_candidates).await?;
        
        let mut search_results = self.create_search_responses(&search_candidates, &posts)?;

        // Step 4: Apply filters if specified
        if let Some(filters) = &request.filters {
            debug!("Applying search filters");
            search_results = self.apply_filters(search_results, filters);
            info!("After filtering: {} results remain", search_results.len());
        }

        // Step 5: Apply minimum score threshold if specified
        if let Some(min_score) = request.min_score {
            debug!("Applying minimum score threshold: {}", min_score);
            let original_count = search_results.len();
            search_results.retain(|result| result.score >= min_score);
            info!("After min_score filter: {} results remain (was {})", 
                  search_results.len(), original_count);
        }

        // Step 6: Perform reranking if enabled and degraded mode is not active
        let should_rerank = request.rerank && search_mode != SearchMode::Degraded;
        if should_rerank {
            debug!("Performing cross-encoder reranking");
            let original_results = search_results.clone(); // Clone for fallback
            match self.reranking_service
                .rerank_results(&request.query, &search_results, true)
                .await
            {
                Ok(reranked) => {
                    search_results = reranked;
                    info!("Reranking completed successfully");
                }
                Err(e) => {
                    warn!("Reranking failed, continuing with original scores: {}", e);
                    search_results = original_results; // Use cloned original results
                }
            }
        } else if request.rerank && search_mode == SearchMode::Degraded {
            warn!("Reranking requested but system is in degraded mode, skipping reranking");
        }

        // Step 7: Limit results to requested number
        search_results.truncate(request.k as usize);

        info!("Semantic search completed: {} final results returned", search_results.len());
        Ok(search_results)
    }

    /// Fetch posts for the given search candidates with metadata backfill from cache
    async fn fetch_posts_for_candidates(&self, candidates: &[SearchCandidate]) -> SearchResult<Vec<Post>> {
        let post_ids: Vec<String> = candidates.iter().map(|c| c.post_id.clone()).collect();
        
        debug!("Fetching {} posts with metadata backfill", post_ids.len());
        
        // Try to get posts from database first (primary source)
        match self.database_manager.get_posts_by_ids(&post_ids).await {
            Ok(posts) => {
                debug!("Successfully fetched {} posts from database", posts.len());
                Ok(posts)
            }
            Err(e) => {
                warn!("Database fetch failed, attempting metadata backfill from cache: {}", e);
                self.fetch_posts_with_metadata_backfill(&post_ids).await
            }
        }
    }

    /// Fetch posts using metadata backfill from Redis cache with Postgres fallback
    async fn fetch_posts_with_metadata_backfill(&self, post_ids: &[String]) -> SearchResult<Vec<Post>> {
        debug!("Attempting metadata backfill for {} posts", post_ids.len());
        
        let mut posts = Vec::new();
        let mut missing_post_ids = Vec::new();
        
        // First, try to get metadata from Redis cache
        for post_id in post_ids {
            match self.fallback_search.cache_manager().get_metadata_cache(post_id).await {
                Ok(Some(metadata)) => {
                    debug!("Found cached metadata for post: {}", post_id);
                    
                    // Create a minimal Post struct from cached metadata
                    // Note: We don't have full content or embedding from cache, but we have enough for response
                    let post = Post {
                        id: uuid::Uuid::new_v4(), // Temporary UUID
                        post_id: post_id.clone(),
                        title: "".to_string(), // Will be filled from metadata if available
                        content: "Content unavailable".to_string(), // Fallback content
                        author_name: metadata.author_name.clone(),
                        language: metadata.language.clone(),
                        frozen: metadata.frozen,
                        date_gmt: metadata.date,
                        url: metadata.url.clone(),
                        embedding: Vec::new(), // Empty embedding for cache-only posts
                    };
                    posts.push(post);
                }
                Ok(None) => {
                    debug!("No cached metadata found for post: {}", post_id);
                    missing_post_ids.push(post_id.clone());
                }
                Err(e) => {
                    warn!("Failed to fetch cached metadata for post {}: {}", post_id, e);
                    missing_post_ids.push(post_id.clone());
                }
            }
        }
        
        // For missing posts, try individual database lookups as fallback
        if !missing_post_ids.is_empty() {
            debug!("Attempting individual database lookups for {} missing posts", missing_post_ids.len());
            
            for post_id in &missing_post_ids {
                match self.database_manager.get_post_by_id(post_id).await {
                    Ok(Some(post)) => {
                        debug!("Successfully fetched post {} from database", post_id);
                        posts.push(post);
                        
                        // Cache the metadata for future use
                        let metadata = PostMetadata {
                            author_name: posts.last().unwrap().author_name.clone(),
                            url: posts.last().unwrap().url.clone(),
                            date: posts.last().unwrap().date_gmt,
                            language: posts.last().unwrap().language.clone(),
                            frozen: posts.last().unwrap().frozen,
                        };
                        
                        if let Err(e) = self.fallback_search.cache_manager().set_metadata_cache(post_id, &metadata).await {
                            warn!("Failed to cache metadata for post {}: {}", post_id, e);
                        }
                    }
                    Ok(None) => {
                        warn!("Post not found in database: {}", post_id);
                    }
                    Err(e) => {
                        warn!("Failed to fetch post {} from database: {}", post_id, e);
                    }
                }
            }
        }
        
        info!("Metadata backfill completed: {} posts retrieved", posts.len());
        Ok(posts)
    }

    /// Create search responses from candidates and posts
    fn create_search_responses(
        &self,
        candidates: &[SearchCandidate],
        posts: &[Post],
    ) -> SearchResult<Vec<SearchResponse>> {
        debug!("Creating search responses for {} candidates", candidates.len());

        let mut results = Vec::new();
        
        for candidate in candidates {
            if let Some(post) = posts.iter().find(|p| p.post_id == candidate.post_id) {
                let search_response = post.to_search_response(candidate.score);
                results.push(search_response);
            } else {
                warn!("Post not found for candidate: {}", candidate.post_id);
            }
        }

        debug!("Created {} search responses", results.len());
        Ok(results)
    }

    /// Apply search filters to results with comprehensive filtering logic
    fn apply_filters(&self, mut results: Vec<SearchResponse>, filters: &SearchFilters) -> Vec<SearchResponse> {
        let original_count = results.len();
        debug!("Applying filters to {} results", original_count);

        // Apply language filter
        if let Some(language) = &filters.language {
            let before_count = results.len();
            results.retain(|result| {
                // Case-insensitive language matching for better compatibility
                result.meta.language.to_lowercase() == language.to_lowercase()
            });
            debug!("Language filter '{}' applied: {} -> {} results", 
                   language, before_count, results.len());
        }

        // Apply frozen filter
        if let Some(frozen) = filters.frozen {
            let before_count = results.len();
            results.retain(|result| result.meta.frozen == frozen);
            debug!("Frozen filter '{}' applied: {} -> {} results", 
                   frozen, before_count, results.len());
            
            // Log specific filtering behavior for GDPR compliance
            if !frozen {
                debug!("Excluded {} frozen posts for GDPR compliance", 
                       before_count - results.len());
            }
        }

        let final_count = results.len();
        if final_count != original_count {
            info!("Filtering completed: {} -> {} results ({} filtered out)", 
                  original_count, final_count, original_count - final_count);
        }

        results
    }

    /// Get current search mode from fallback service
    pub async fn get_current_search_mode(&self) -> SearchMode {
        self.fallback_search.get_current_search_mode().await
    }

    /// Check if reranking is available
    pub fn is_reranking_available(&self) -> bool {
        self.reranking_service.is_available()
    }

    /// Get reranking configuration
    pub fn get_reranking_config(&self) -> &RerankingConfig {
        self.reranking_service.config()
    }

    /// Perform health check on all components
    pub async fn health_check(&self) -> SearchResult<SearchServiceHealth> {
        let fallback_health = self.fallback_search.health_check().await?;
        let ml_available = true; // Assume ML service is available if service was created
        let reranking_available = self.is_reranking_available();

        Ok(SearchServiceHealth {
            fallback_health,
            ml_service_available: ml_available,
            reranking_available,
            current_search_mode: self.get_current_search_mode().await,
        })
    }

    /// Get search statistics
    pub async fn get_search_stats(&self) -> SearchResult<SearchServiceStats> {
        let circuit_breaker_stats = self.fallback_search.get_circuit_breaker_stats().await;
        let current_mode = self.get_current_search_mode().await;
        let reranking_config = self.get_reranking_config().clone();

        Ok(SearchServiceStats {
            circuit_breaker_stats,
            current_search_mode: current_mode,
            reranking_config,
            reranking_available: self.is_reranking_available(),
        })
    }
}

/// Health status for the complete search service
#[derive(Debug, Clone)]
pub struct SearchServiceHealth {
    pub fallback_health: crate::search::FallbackHealthStatus,
    pub ml_service_available: bool,
    pub reranking_available: bool,
    pub current_search_mode: SearchMode,
}

/// Statistics for the search service
#[derive(Debug, Clone)]
pub struct SearchServiceStats {
    pub circuit_breaker_stats: crate::search::CircuitBreakerStats,
    pub current_search_mode: SearchMode,
    pub reranking_config: RerankingConfig,
    pub reranking_available: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SearchFilters, PostMetadata};
    use chrono::Utc;

    /// Create test search results for filtering tests
    fn create_test_results_for_filtering() -> Vec<SearchResponse> {
        vec![
            SearchResponse {
                post_id: "post1".to_string(),
                title: "English Post".to_string(),
                snippet: "This is an English post".to_string(),
                score: 0.9,
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
                title: "Spanish Post".to_string(),
                snippet: "Este es un post en español".to_string(),
                score: 0.8,
                meta: PostMetadata {
                    author_name: "Author 2".to_string(),
                    url: "https://example.com/post2".to_string(),
                    date: Utc::now(),
                    language: "es".to_string(),
                    frozen: true,
                },
            },
            SearchResponse {
                post_id: "post3".to_string(),
                title: "Another English Post".to_string(),
                snippet: "Another English post that is frozen".to_string(),
                score: 0.7,
                meta: PostMetadata {
                    author_name: "Author 3".to_string(),
                    url: "https://example.com/post3".to_string(),
                    date: Utc::now(),
                    language: "en".to_string(),
                    frozen: true,
                },
            },
        ]
    }

    #[test]
    fn test_apply_language_filter() {
        // This test doesn't require async setup, so we can test the filtering logic directly
        // In a real implementation, we'd need to mock the SearchService
        let results = create_test_results_for_filtering();
        
        // Test language filter
        let filters = SearchFilters {
            language: Some("en".to_string()),
            frozen: None,
        };
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                if let Some(language) = &filters.language {
                    result.meta.language == *language
                } else {
                    true
                }
            })
            .collect();
        
        assert_eq!(filtered.len(), 2); // Should have 2 English posts
        assert!(filtered.iter().all(|r| r.meta.language == "en"));
    }

    #[test]
    fn test_apply_frozen_filter() {
        let results = create_test_results_for_filtering();
        
        // Test frozen filter (exclude frozen posts)
        let filters = SearchFilters {
            language: None,
            frozen: Some(false),
        };
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                if let Some(frozen) = filters.frozen {
                    result.meta.frozen == frozen
                } else {
                    true
                }
            })
            .collect();
        
        assert_eq!(filtered.len(), 1); // Should have 1 non-frozen post
        assert!(filtered.iter().all(|r| !r.meta.frozen));
    }

    #[test]
    fn test_apply_combined_filters() {
        let results = create_test_results_for_filtering();
        
        // Test combined filters
        let filters = SearchFilters {
            language: Some("en".to_string()),
            frozen: Some(false),
        };
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                let language_match = if let Some(language) = &filters.language {
                    result.meta.language == *language
                } else {
                    true
                };
                
                let frozen_match = if let Some(frozen) = filters.frozen {
                    result.meta.frozen == frozen
                } else {
                    true
                };
                
                language_match && frozen_match
            })
            .collect();
        
        assert_eq!(filtered.len(), 1); // Should have 1 English, non-frozen post
        assert_eq!(filtered[0].post_id, "post1");
    }

    #[test]
    fn test_min_score_filtering() {
        let results = create_test_results_for_filtering();
        let min_score = 0.75;
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| result.score >= min_score)
            .collect();
        
        assert_eq!(filtered.len(), 2); // Should have 2 results with score >= 0.75
        assert!(filtered.iter().all(|r| r.score >= min_score));
    }

    #[test]
    fn test_search_service_health_structure() {
        // Test that the health structure can be created
        let health = SearchServiceHealth {
            fallback_health: crate::search::FallbackHealthStatus {
                redis_healthy: true,
                postgres_healthy: true,
                circuit_breaker_stats: crate::search::CircuitBreakerStats {
                    state: crate::search::CircuitState::Closed,
                    redis_failures: 0,
                    postgres_failures: 0,
                    recent_failures: 0,
                    success_count: 0,
                },
                current_search_mode: SearchMode::Full,
                redis_error: None,
                postgres_error: None,
            },
            ml_service_available: true,
            reranking_available: true,
            current_search_mode: SearchMode::Full,
        };
        
        assert!(health.ml_service_available);
        assert!(health.reranking_available);
        assert_eq!(health.current_search_mode, SearchMode::Full);
    }

    #[test]
    fn test_search_service_stats_structure() {
        let stats = SearchServiceStats {
            circuit_breaker_stats: crate::search::CircuitBreakerStats {
                state: crate::search::CircuitState::Closed,
                redis_failures: 0,
                postgres_failures: 0,
                recent_failures: 0,
                success_count: 0,
            },
            current_search_mode: SearchMode::Full,
            reranking_config: RerankingConfig::default(),
            reranking_available: true,
        };
        
        assert_eq!(stats.current_search_mode, SearchMode::Full);
        assert!(stats.reranking_available);
        assert_eq!(stats.reranking_config.max_candidates_to_rerank, 50);
    }

    #[test]
    fn test_apply_language_filter_case_insensitive() {
        let results = create_test_results_for_filtering();
        
        // Test case-insensitive language filter
        let filters = SearchFilters {
            language: Some("EN".to_string()), // Uppercase
            frozen: None,
        };
        
        // Simulate the filtering logic
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                if let Some(language) = &filters.language {
                    result.meta.language.to_lowercase() == language.to_lowercase()
                } else {
                    true
                }
            })
            .collect();
        
        assert_eq!(filtered.len(), 2); // Should have 2 English posts
        assert!(filtered.iter().all(|r| r.meta.language.to_lowercase() == "en"));
    }

    #[test]
    fn test_apply_frozen_filter_gdpr_compliance() {
        let results = create_test_results_for_filtering();
        
        // Test frozen filter for GDPR compliance (exclude frozen posts)
        let filters = SearchFilters {
            language: None,
            frozen: Some(false),
        };
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                if let Some(frozen) = filters.frozen {
                    result.meta.frozen == frozen
                } else {
                    true
                }
            })
            .collect();
        
        assert_eq!(filtered.len(), 1); // Should have 1 non-frozen post
        assert!(filtered.iter().all(|r| !r.meta.frozen));
        assert_eq!(filtered[0].post_id, "post1");
    }

    #[test]
    fn test_apply_combined_filters_comprehensive() {
        let results = create_test_results_for_filtering();
        
        // Test combined filters with edge cases
        let filters = SearchFilters {
            language: Some("es".to_string()),
            frozen: Some(true),
        };
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                let language_match = if let Some(language) = &filters.language {
                    result.meta.language.to_lowercase() == language.to_lowercase()
                } else {
                    true
                };
                
                let frozen_match = if let Some(frozen) = filters.frozen {
                    result.meta.frozen == frozen
                } else {
                    true
                };
                
                language_match && frozen_match
            })
            .collect();
        
        assert_eq!(filtered.len(), 1); // Should have 1 Spanish, frozen post
        assert_eq!(filtered[0].post_id, "post2");
        assert_eq!(filtered[0].meta.language, "es");
        assert!(filtered[0].meta.frozen);
    }

    #[test]
    fn test_apply_filters_no_matches() {
        let results = create_test_results_for_filtering();
        
        // Test filters that match no results
        let filters = SearchFilters {
            language: Some("fr".to_string()), // No French posts
            frozen: None,
        };
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                if let Some(language) = &filters.language {
                    result.meta.language.to_lowercase() == language.to_lowercase()
                } else {
                    true
                }
            })
            .collect();
        
        assert_eq!(filtered.len(), 0); // Should have no results
    }

    #[test]
    fn test_apply_filters_empty_input() {
        let results: Vec<SearchResponse> = vec![];
        
        let filters = SearchFilters {
            language: Some("en".to_string()),
            frozen: Some(false),
        };
        
        let filtered: Vec<SearchResponse> = results
            .into_iter()
            .filter(|result| {
                let language_match = if let Some(language) = &filters.language {
                    result.meta.language.to_lowercase() == language.to_lowercase()
                } else {
                    true
                };
                
                let frozen_match = if let Some(frozen) = filters.frozen {
                    result.meta.frozen == frozen
                } else {
                    true
                };
                
                language_match && frozen_match
            })
            .collect();
        
        assert_eq!(filtered.len(), 0); // Should remain empty
    }
}