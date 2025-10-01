# Requirements Document

## Introduction

The RAG-Search API is a 100% production-ready cloud microservice that provides semantic search capabilities over a knowledge base of posts. The system transforms natural language queries into vector embeddings and returns the most semantically similar posts with ≤100ms p95 latency and 99.9% monthly availability. The service is built with Rust for performance, uses ONNX-transformed Hugging Face models for ML inference, Upstash Redis for vector caching, and Supabase Postgres with pgvector for the knowledge base. The system includes comprehensive production gates, compliance controls (GDPR + SOC-2), zero-downtime deployments, and full on-call coverage.

## Requirements

### Requirement 1

**User Story:** As a client application, I want to submit semantic search queries via HTTP/gRPC endpoints, so that I can retrieve relevant posts based on natural language queries.

#### Acceptance Criteria

1. WHEN a client sends a POST request to /semantic-search THEN the system SHALL accept the request with query, k, min_score, rerank, and filters parameters
2. WHEN a client connects via gRPC THEN the system SHALL provide a streaming SemanticSearch endpoint
3. WHEN the system processes a query THEN it SHALL return results in JSON/NDJSON format
4. WHEN k parameter is provided THEN the system SHALL limit results to maximum 50 items
5. WHEN min_score filter is applied THEN the system SHALL only return results above the specified threshold

### Requirement 2

**User Story:** As a client application, I want fast semantic search responses with contractual SLO guarantees, so that my users experience consistent performance.

#### Acceptance Criteria

1. WHEN the system is under normal load THEN it SHALL respond with p50 ≤ 35ms, p95 ≤ 100ms, p99 ≤ 150ms (contractual SLO)
2. WHEN the system experiences cold start THEN it SHALL respond within ≤ 2 seconds including model download
3. WHEN processing queries THEN each instance SHALL handle ≥ 3,000 RPS at 80% CPU utilization
4. WHEN the system scales THEN it SHALL support up to 10 instances per region for ≥ 30k RPS total capacity
5. WHEN recall is measured THEN the system SHALL maintain Recall@20 ≥ 0.92 vs brute-force FAISS (monthly evaluation)

### Requirement 3

**User Story:** As a system administrator, I want the service to use efficient vector search and caching, so that performance targets are met consistently.

#### Acceptance Criteria

1. WHEN a query is processed THEN the system SHALL generate embeddings using ONNX bi-encoder (all-MiniLM-L6-v2)
2. WHEN performing vector search THEN the system SHALL query both Redis VSS and Postgres IVFFlat in parallel
3. WHEN rerank=true is specified THEN the system SHALL use ONNX cross-encoder (ms-marco-MiniLM-L-6-v2) for result scoring
4. WHEN caching results THEN the system SHALL store top-k results in Redis with 60s TTL
5. WHEN vector cache is accessed THEN the system SHALL maintain permanent LRU-evicted vector storage

### Requirement 4

**User Story:** As a client application, I want to filter search results by metadata, so that I can narrow results to specific criteria.

#### Acceptance Criteria

1. WHEN filters are provided THEN the system SHALL support language and frozen status filtering
2. WHEN language filter is applied THEN the system SHALL only return posts matching the specified language
3. WHEN frozen filter is false THEN the system SHALL exclude frozen posts from results
4. WHEN multiple filters are applied THEN the system SHALL apply all filters as AND conditions

### Requirement 5

**User Story:** As a system operator, I want comprehensive observability and monitoring, so that I can maintain system health and performance.

#### Acceptance Criteria

1. WHEN the system processes requests THEN it SHALL emit Prometheus metrics for search_total, search_duration_seconds, redis_hit_topk_ratio, pg_tuples_returned, inflight_requests, and model_inference_seconds
2. WHEN performance degrades THEN the system SHALL trigger alerts for p95 > 150ms, error rate > 1%, Redis hit ratio < 30%, Redis connection failures > 5
3. WHEN requests are processed THEN the system SHALL output structured JSON logs with trace_id injection
4. WHEN tracing is enabled THEN the system SHALL send OpenTelemetry traces to managed Jaeger

### Requirement 6

**User Story:** As a system administrator, I want proper error handling and rate limiting, so that the service remains stable under various conditions.

#### Acceptance Criteria

1. WHEN rate limits are exceeded THEN the system SHALL return 429 Too Many Requests
2. WHEN processing takes longer than 500ms THEN the system SHALL return 504 Gateway Timeout
3. WHEN Redis is unavailable THEN the system SHALL fallback to Postgres-only search
4. WHEN Postgres timeouts occur THEN the system SHALL handle gracefully with statement_timeout 500ms
5. WHEN malformed requests are received THEN the system SHALL return appropriate 4xx error codes

### Requirement 7

**User Story:** As a security administrator, I want the service to implement proper security controls, so that data and system access are protected.

#### Acceptance Criteria

1. WHEN connections are established THEN the system SHALL use TLS 1.3 for all communications
2. WHEN query text is processed THEN the system SHALL NOT store query text longer than 60 seconds
3. WHEN accessing data stores THEN the system SHALL use least-privilege role-based access (rag_reader RO)
4. WHEN requests exceed size limits THEN the system SHALL block POST bodies > 32kB
5. WHEN rate limiting is applied THEN the system SHALL enforce 100 RPS burst / 30 RPS sustained per IP

### Requirement 8

**User Story:** As a DevOps engineer, I want automated deployment and scaling capabilities with zero-downtime deployments, so that the service can handle varying loads efficiently without service interruption.

#### Acceptance Criteria

1. WHEN deployed to Cloud Run THEN the system SHALL autoscale with min instances = 1 during business hours (8-20), max instances = 10 per region
2. WHEN traffic increases THEN the system SHALL scale up automatically with quota alarm at 80%
3. WHEN traffic decreases THEN the system SHALL scale down to minimize costs while maintaining availability
4. WHEN deployed THEN the system SHALL use distroless container images ≤ 120MB with static musl binary
5. WHEN rollback is needed THEN the system SHALL support one-button rollback in < 30 seconds
6. WHEN deploying THEN the system SHALL use canary deployments (1% → 5% → 25% → 100%) with KPI diff ≤ 1% error, ≤ 1% recall### Requi
rement 9

**User Story:** As a system administrator, I want comprehensive production readiness gates, so that the service meets all operational requirements before going live.

#### Acceptance Criteria

1. WHEN certificates are configured THEN the system SHALL have ACM auto-renew enabled (CERT-01)
2. WHEN credentials are rotated THEN the rotate-creds.sh script SHALL work without rebuild (SECRET-01)
3. WHEN under connection storm THEN the system SHALL maintain ≤ 200 active PG connections at 20k RPS (CONN-01)
4. WHEN on-call procedures are established THEN runbook and severity matrix SHALL be published and drilled (OC-01)
5. WHEN GDPR compliance is required THEN Redis cache purge script SHALL be tested and logged (GDPR-01)
6. WHEN canary deployments are performed THEN last 3 canaries SHALL show <1% error and recall diff ≤1% (CANARY-01)
7. WHEN disaster recovery is tested THEN cross-region failover SHALL be tested with 24h delay (DR-01)

### Requirement 10

**User Story:** As a compliance officer, I want GDPR and SOC-2 compliance controls, so that the service meets regulatory requirements.

#### Acceptance Criteria

1. WHEN personal data is processed THEN the system SHALL implement data subject deletion workflow completing < 24h SLA
2. WHEN deletion is requested THEN the system SHALL purge post_id from Redis (UNLINK search:vec:<post_id>, search:meta:<post_id>)
3. WHEN audit trails are required THEN the system SHALL export audit logs to GCP Cloud Audit with 365-day retention
4. WHEN data is stored THEN the system SHALL limit to post_id, title, snippet (≤300 chars), author_name, url, date
5. WHEN right to be forgotten is exercised THEN the system SHALL complete deletion workflow with audit log entry

### Requirement 11

**User Story:** As an SRE, I want comprehensive reliability and disaster recovery capabilities, so that the service maintains 99.9% monthly availability.

#### Acceptance Criteria

1. WHEN Redis connections fail >5 times THEN the system SHALL activate circuit breaker and degrade to PG-only path
2. WHEN retries are needed THEN the system SHALL implement exponential backoff with max 3 idempotent retries
3. WHEN cross-region failover is needed THEN the system SHALL use read replica in us-east-1 with RPO < 5 min
4. WHEN backups are required THEN the system SHALL maintain PG daily snapshots retained 30 days and models in GCS multi-region
5. WHEN SLO is monitored THEN the system SHALL maintain 99.9% monthly availability with error budget burn rate alerts
6. WHEN disaster recovery is tested THEN the system SHALL perform quarterly failover simulation

### Requirement 12

**User Story:** As a security administrator, I want enhanced security controls with automatic credential rotation, so that the service maintains security compliance.

#### Acceptance Criteria

1. WHEN connections are established THEN the system SHALL use TLS 1.3 everywhere with mTLS enabled for Cloud Run ↔ Redis
2. WHEN credentials are managed THEN the system SHALL use Google Secret Manager with 90-day automatic rotation
3. WHEN access control is applied THEN the system SHALL use least-privilege service accounts with no static keys
4. WHEN WAF protection is active THEN the system SHALL implement OWASP CRS with rate-limit 100 RPS burst, 30 sustained
5. WHEN models are loaded THEN the system SHALL verify SHA256 hash at startup and crash loop on mismatch

### Requirement 13

**User Story:** As an operations team member, I want comprehensive on-call procedures and incident management, so that service issues are resolved quickly and systematically.

#### Acceptance Criteria

1. WHEN incidents occur THEN the system SHALL classify as SEV-0 (full outage >5 min), SEV-1 (degraded >15 min), SEV-2 (single feature broken)
2. WHEN alerts are triggered THEN the system SHALL notify Primary/Secondary on-call via PagerDuty
3. WHEN incidents are resolved THEN the system SHALL complete blameless post-mortem within 5 days
4. WHEN runbook is accessed THEN it SHALL be available at https://wiki.company/rag-search-runbook
5. WHEN connection pools are managed THEN the system SHALL limit to max 12 per container with Supavisor enabled