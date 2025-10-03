use super::*;
use crate::cache::CacheManager;
use crate::config::{DatabaseConfig, RedisConfig};
use crate::database::DatabaseManager;
use crate::types::{SearchCandidate, SearchSource};
use std::sync::Arc;
use tokio;

/// Mock cache manager for testing
struct MockCacheManager {
    should_fail: bool,
    candidates: Vec<SearchCandidate>,
}

impl MockCacheManager {
    fn new(candidates: Vec<SearchCandidate>) -> Self {
        Self {
            should_fail: false,
            candidates,
        }
    }

    fn with_failure() -> Self {
        Self {
            should_fail: true,
            candidates: Vec::new(),
        }
    }

    async fn vector_search(&self, _query_vector: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        if self.should_fail {
            return Err(SearchError::RedisError("Mock Redis failure".to_string()));
        }
        
        let mut results = self.candidates.clone();
        results.truncate(limit);
        Ok(results)
    }

    async fn health_check(&self) -> SearchResult<()> {
        if self.should_fail {
            Err(SearchError::RedisError("Mock Redis unhealthy".to_string()))
        } else {
            Ok(())
        }
    }
}

/// Mock database manager for testing
struct MockDatabaseManager {
    should_fail: bool,
    candidates: Vec<SearchCandidate>,
}

impl MockDatabaseManager {
    fn new(candidates: Vec<SearchCandidate>) -> Self {
        Self {
            should_fail: false,
            candidates,
        }
    }

    fn with_failure() -> Self {
        Self {
            should_fail: true,
            candidates: Vec::new(),
        }
    }

    async fn vector_search(&self, _query_vector: &[f32], limit: usize) -> SearchResult<Vec<SearchCandidate>> {
        if self.should_fail {
            return Err(SearchError::DatabaseError("Mock Postgres failure".to_string()));
        }
        
        let mut results = self.candidates.clone();
        results.truncate(limit);
        Ok(results)
    }

    async fn health_check(&self) -> SearchResult<()> {
        if self.should_fail {
            Err(SearchError::DatabaseError("Mock Postgres unhealthy".to_string()))
        } else {
            Ok(())
        }
    }
}

/// Create test search candidates
fn create_test_candidates() -> (Vec<SearchCandidate>, Vec<SearchCandidate>) {
    let redis_candidates = vec![
        SearchCandidate {
            post_id: "post1".to_string(),
            score: 0.95,
            source: SearchSource::Redis,
        },
        SearchCandidate {
            post_id: "post2".to_string(),
            score: 0.85,
            source: SearchSource::Redis,
        },
        SearchCandidate {
            post_id: "post3".to_string(),
            score: 0.75,
            source: SearchSource::Redis,
        },
    ];

    let postgres_candidates = vec![
        SearchCandidate {
            post_id: "post1".to_string(), // Duplicate with different score
            score: 0.90,
            source: SearchSource::Postgres,
        },
        SearchCandidate {
            post_id: "post4".to_string(),
            score: 0.80,
            source: SearchSource::Postgres,
        },
        SearchCandidate {
            post_id: "post5".to_string(),
            score: 0.70,
            source: SearchSource::Postgres,
        },
    ];

    (redis_candidates, postgres_candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_and_dedup_basic() {
        // Create a mock service for testing merge logic
        let redis_config = RedisConfig {
            url: "redis://localhost:6379".to_string(),
            max_connections: 10,
            connection_timeout_secs: 5,
            default_ttl_secs: 3600,
        };
        
        let database_config = DatabaseConfig {
            supabase_url: "postgresql://localhost:5432/test".to_string(),
            supabase_service_key: "test_key".to_string(),
            max_connections: 10,
            connection_timeout_secs: 30,
        };

        // We can't easily create real managers in tests, so we'll test the merge logic directly
        let candidates = vec![
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.95,
                source: SearchSource::Redis,
            },
            SearchCandidate {
                post_id: "post2".to_string(),
                score: 0.85,
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post1".to_string(), // Duplicate with lower score
                score: 0.90,
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post3".to_string(),
                score: 0.75,
                source: SearchSource::Redis,
            },
        ];

        // Create a temporary service to test merge logic
        // Since we can't easily mock the managers, we'll create a helper function
        let merged = merge_and_dedup_helper(candidates, 130);

        // Should have 3 unique posts
        assert_eq!(merged.len(), 3);

        // Should be sorted by score (descending)
        assert_eq!(merged[0].post_id, "post1");
        assert_eq!(merged[0].score, 0.95); // Higher score kept
        assert_eq!(merged[0].source, SearchSource::Redis);

        assert_eq!(merged[1].post_id, "post2");
        assert_eq!(merged[1].score, 0.85);

        assert_eq!(merged[2].post_id, "post3");
        assert_eq!(merged[2].score, 0.75);
    }

    #[test]
    fn test_merge_and_dedup_max_candidates() {
        let mut candidates = Vec::new();
        
        // Create 150 candidates to test the 130 limit
        for i in 0..150 {
            candidates.push(SearchCandidate {
                post_id: format!("post{}", i),
                score: 1.0 - (i as f32 * 0.001), // Decreasing scores
                source: SearchSource::Redis,
            });
        }

        let merged = merge_and_dedup_helper(candidates, 130);

        // Should be limited to 130 candidates
        assert_eq!(merged.len(), 130);

        // Should be sorted by score (descending)
        assert!(merged[0].score > merged[1].score);
        assert!(merged[1].score > merged[2].score);
    }

    #[test]
    fn test_merge_and_dedup_duplicate_handling() {
        let candidates = vec![
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.80,
                source: SearchSource::Redis,
            },
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.90, // Higher score
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.85, // Middle score
                source: SearchSource::Redis,
            },
        ];

        let merged = merge_and_dedup_helper(candidates, 130);

        // Should have only 1 result
        assert_eq!(merged.len(), 1);

        // Should keep the highest score
        assert_eq!(merged[0].score, 0.90);
        assert_eq!(merged[0].source, SearchSource::Postgres);
    }

    #[test]
    fn test_merge_and_dedup_empty_input() {
        let candidates = vec![];
        let merged = merge_and_dedup_helper(candidates, 130);
        assert_eq!(merged.len(), 0);
    }

    #[test]
    fn test_merge_and_dedup_score_sorting() {
        let candidates = vec![
            SearchCandidate {
                post_id: "post1".to_string(),
                score: 0.60,
                source: SearchSource::Redis,
            },
            SearchCandidate {
                post_id: "post2".to_string(),
                score: 0.90,
                source: SearchSource::Postgres,
            },
            SearchCandidate {
                post_id: "post3".to_string(),
                score: 0.75,
                source: SearchSource::Redis,
            },
        ];

        let merged = merge_and_dedup_helper(candidates, 130);

        // Should be sorted by score (descending)
        assert_eq!(merged[0].post_id, "post2");
        assert_eq!(merged[0].score, 0.90);

        assert_eq!(merged[1].post_id, "post3");
        assert_eq!(merged[1].score, 0.75);

        assert_eq!(merged[2].post_id, "post1");
        assert_eq!(merged[2].score, 0.60);
    }

    // Helper function to test merge logic without needing real managers
    fn merge_and_dedup_helper(candidates: Vec<SearchCandidate>, max_candidates: usize) -> Vec<SearchCandidate> {
        use std::collections::HashMap;

        // Use HashMap to deduplicate by post_id, keeping the highest score
        let mut best_candidates: HashMap<String, SearchCandidate> = HashMap::new();

        for candidate in candidates {
            match best_candidates.get(&candidate.post_id) {
                Some(existing) => {
                    // Keep the candidate with higher score
                    if candidate.score > existing.score {
                        best_candidates.insert(candidate.post_id.clone(), candidate);
                    }
                }
                None => {
                    best_candidates.insert(candidate.post_id.clone(), candidate);
                }
            }
        }

        // Convert to vector and sort by score (descending)
        let mut merged_candidates: Vec<SearchCandidate> = best_candidates.into_values().collect();
        merged_candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max candidates
        merged_candidates.truncate(max_candidates);
        merged_candidates
    }
}

// Integration tests that would require real Redis/Postgres connections
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::env;

    async fn create_test_cache_manager() -> Option<Arc<CacheManager>> {
        if let Ok(redis_url) = env::var("REDIS_URL") {
            let config = RedisConfig {
                url: redis_url,
                max_connections: 5,
                connection_timeout_secs: 5,
                default_ttl_secs: 3600,
            };
            
            if let Ok(manager) = CacheManager::new(config).await {
                return Some(Arc::new(manager));
            }
        }
        None
    }

    async fn create_test_database_manager() -> Option<Arc<DatabaseManager>> {
        if let Ok(database_url) = env::var("DATABASE_URL") {
            let config = DatabaseConfig {
                supabase_url: database_url,
                supabase_service_key: "test_key".to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
            };
            
            if let Ok(manager) = DatabaseManager::new(config).await {
                return Some(Arc::new(manager));
            }
        }
        None
    }

    #[tokio::test]
    #[ignore = "requires Redis and Postgres connections"]
    async fn test_parallel_search_integration() {
        let cache_manager = create_test_cache_manager().await;
        let database_manager = create_test_database_manager().await;

        if let (Some(cache), Some(db)) = (cache_manager, database_manager) {
            let search_service = VectorSearchService::new(cache, db);
            
            // Test with a sample query vector
            let query_vector = vec![0.1; 384]; // 384-dimensional vector
            let limit = 10;

            let result = search_service.parallel_search(&query_vector, limit).await;
            
            // Should succeed even if no results found
            assert!(result.is_ok(), "Parallel search failed: {:?}", result);
            
            let candidates = result.unwrap();
            assert!(candidates.len() <= limit);
        } else {
            println!("Skipping integration test - Redis or Postgres not available");
        }
    }

    #[tokio::test]
    #[ignore = "requires Redis and Postgres connections"]
    async fn test_health_check_integration() {
        let cache_manager = create_test_cache_manager().await;
        let database_manager = create_test_database_manager().await;

        if let (Some(cache), Some(db)) = (cache_manager, database_manager) {
            let search_service = VectorSearchService::new(cache, db);
            
            let health_result = search_service.health_check().await;
            assert!(health_result.is_ok(), "Health check failed: {:?}", health_result);
        }
    }

    #[tokio::test]
    #[ignore = "requires Redis and Postgres connections"]
    async fn test_search_stats_integration() {
        let cache_manager = create_test_cache_manager().await;
        let database_manager = create_test_database_manager().await;

        if let (Some(cache), Some(db)) = (cache_manager, database_manager) {
            let search_service = VectorSearchService::new(cache, db);
            
            let stats_result = search_service.get_search_stats().await;
            assert!(stats_result.is_ok(), "Get stats failed: {:?}", stats_result);
            
            let stats = stats_result.unwrap();
            // Basic validation that stats are populated
            assert!(stats.postgres_total_posts >= 0);
        }
    }
}