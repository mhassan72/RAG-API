# Implementation Plan

- [x] 1. Set up project structure and core interfaces
  - Create Rust project with Cargo.toml dependencies (axum, tokio, tonic, ort, redis, tokio-postgres)
  - Define core data structures and interfaces for SearchRequest, SearchResponse, and error types
  - Set up basic project structure with modules for server, ml, search, cache, and error handling
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Implement basic HTTP server with request validation
  - Create Axum HTTP server with /semantic-search POST endpoint
  - Implement request validation for query length, k parameter limits, and filter validation
  - Add basic error handling with proper HTTP status codes (400, 429, 504)
  - Write unit tests for request validation and error response formatting
  - _Requirements: 1.1, 1.4, 6.4, 6.5_

- [ ] 3. Implement tokenization and text preprocessing
  - Integrate hf-tokenizers for text preprocessing and tokenization
  - Create TokenizerService with query normalization and text cleaning
  - Implement query hashing using farmhash64 for cache keys
  - Write unit tests for tokenization and text preprocessing edge cases
  - _Requirements: 3.1, 3.4_

- [ ] 4. Implement ONNX model loading and inference
  - Create ModelLoader with SHA256 verification for model integrity
  - Implement BiEncoder service using ort (ONNX Runtime) for embedding generation
  - Add model download from GCS with startup verification and crash-on-mismatch behavior
  - Create CrossEncoder service for reranking with scoring capabilities
  - Write unit tests with mock ONNX runtime for model inference
  - _Requirements: 3.1, 3.3, 12.5_

- [ ] 5. Implement Redis connection and vector search
  - Set up Redis client with TLS connection and cluster support using fred
  - Implement vector storage and retrieval with search:vec:<post_id> key pattern
  - Create Redis vector similarity search with HNSW parameters (EF_RUNTIME=200)
  - Add connection pooling and error handling for Redis operations
  - Write unit tests for Redis vector operations and connection handling
  - _Requirements: 3.2, 3.4, 7.1, 12.1_

- [ ] 6. Implement Postgres connection and pgvector search
  - Set up Postgres connection pool using tokio-postgres and deadpool (max 12 connections)
  - Implement IVFFlat vector search with configurable probes and lists parameters
  - Add statement timeout handling (500ms) and connection management
  - Create database schema setup with proper indexes for posts table
  - Write unit tests for Postgres vector search and connection pool behavior
  - _Requirements: 3.2, 3.4, 6.4, 13.5_

- [ ] 7. Implement parallel vector search with merge logic
  - Create parallel search coordinator that queries Redis and Postgres simultaneously
  - Implement candidate merging and deduplication logic with max 130 candidates
  - Add result sorting by cosine similarity scores
  - Handle partial failures where one search source is unavailable
  - Write unit tests for parallel search coordination and merge logic
  - _Requirements: 3.2, 3.4_

- [ ] 8. Implement three-tier caching strategy
  - Create CacheManager with vector cache (permanent LRU), top-k cache (60s TTL), and metadata cache (24h TTL)
  - Implement cache key generation and TTL management
  - Add cache hit/miss tracking and statistics
  - Create cache invalidation logic for GDPR compliance (post deletion)
  - Write unit tests for cache operations, TTL behavior, and invalidation
  - _Requirements: 3.4, 3.5, 10.2, 10.4_

- [ ] 9. Implement circuit breaker and fallback logic
  - Create CircuitBreaker with Redis failure tracking and state management
  - Implement automatic fallback to Postgres-only search when Redis circuit is open
  - Add retry logic with exponential backoff (100ms, 200ms, 400ms, max 3 retries)
  - Create graceful degradation modes (Full, PostgresOnly, CacheOnly, Degraded)
  - Write unit tests for circuit breaker state transitions and fallback scenarios
  - _Requirements: 11.1, 11.2_

- [ ] 10. Implement cross-encoder reranking
  - Add optional reranking using CrossEncoder when rerank=true parameter is set
  - Implement score-based result sorting after cross-encoder inference
  - Add performance optimization to limit reranking to top candidates
  - Handle reranking failures with graceful degradation to similarity scores
  - Write unit tests for reranking logic and fallback behavior
  - _Requirements: 3.3_

- [ ] 11. Implement metadata filtering and result formatting
  - Add filter application for language and frozen status during search
  - Implement metadata backfill from Redis cache with Postgres fallback
  - Create JSON response serialization with proper field formatting
  - Add snippet truncation to 300 characters for GDPR compliance
  - Write unit tests for filtering logic and response formatting
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 10.4_

- [ ] 12. Implement comprehensive observability
  - Add Prometheus metrics for search_total, search_duration_seconds, redis_hit_topk_ratio, pg_tuples_returned, inflight_requests, model_inference_seconds
  - Implement structured JSON logging with trace_id injection
  - Add OpenTelemetry tracing with OTLP export to Jaeger
  - Create health check endpoint for Cloud Run readiness/liveness probes
  - Write unit tests for metrics collection and logging functionality
  - _Requirements: 5.1, 5.3, 5.4_

- [ ] 13. Implement rate limiting and security controls
  - Add rate limiting middleware with configurable burst and sustained limits
  - Implement request size validation (block POST bodies > 32kB)
  - Add CORS headers and security headers for production deployment
  - Create input sanitization to prevent injection attacks
  - Write unit tests for rate limiting behavior and security controls
  - _Requirements: 6.1, 6.5, 12.4_

- [ ] 14. Implement gRPC streaming endpoint
  - Create gRPC service definition with streaming SemanticSearch endpoint using Tonic
  - Implement protobuf message definitions for SearchRequest and SearchResponse
  - Add streaming response logic that sends results as they become available
  - Handle gRPC-specific error codes and status responses
  - Write unit tests for gRPC endpoint and streaming behavior
  - _Requirements: 1.2_

- [ ] 15. Implement GDPR compliance and data deletion
  - Create GdprService with post data deletion workflow
  - Implement Redis cache purge for search:vec and search:meta keys
  - Add audit logging for deletion operations with timestamp tracking
  - Create Pub/Sub subscription for deletion events with <24h SLA processing
  - Write unit tests for GDPR deletion workflow and audit logging
  - _Requirements: 10.1, 10.2, 10.3_

- [ ] 16. Implement secret management and credential rotation
  - Integrate Google Secret Manager for Redis and Postgres credentials
  - Create credential rotation logic that updates connections without service restart
  - Add startup credential validation and connection testing
  - Implement automatic retry on credential refresh failures
  - Write unit tests for credential management and rotation scenarios
  - _Requirements: 9.2, 12.2_

- [ ] 17. Add production deployment configuration
  - Create Dockerfile with distroless base image and static musl binary
  - Add Cloud Run deployment configuration with autoscaling parameters
  - Configure environment variables for production settings (connection limits, timeouts, cache sizes)
  - Add startup health checks and graceful shutdown handling
  - Create deployment scripts for canary releases with traffic splitting
  - _Requirements: 8.4, 8.6_

- [ ] 18. Implement comprehensive error handling and monitoring
  - Add alerting thresholds for p95 latency, error rates, Redis hit ratios, and connection failures
  - Implement error budget tracking and burn rate calculations
  - Create custom error types with proper HTTP status code mapping
  - Add request timeout handling with 504 Gateway Timeout responses
  - Write integration tests for error scenarios and alert triggering
  - _Requirements: 5.2, 6.2, 6.3, 11.5_

- [ ] 19. Add integration tests and end-to-end testing
  - Create integration test suite using testcontainers for Redis and Postgres
  - Implement end-to-end tests covering full search workflow from request to response
  - Add chaos testing scenarios (Redis failures, Postgres timeouts, model inference errors)
  - Create load testing setup with k6 for performance validation
  - Test GDPR deletion workflow end-to-end with cache verification
  - _Requirements: All requirements validation_

- [ ] 20. Implement production readiness gates validation
  - Create automated checks for all production gates (CERT-01, SECRET-01, CONN-01, etc.)
  - Add connection storm testing to validate ≤200 active PG connections at 20k RPS
  - Implement canary deployment validation with error rate and recall diff checking
  - Create disaster recovery testing automation for cross-region failover
  - Add final production checklist validation and sign-off automation
  - _Requirements: 9.1, 9.3, 9.6, 9.7, 11.3, 11.4_