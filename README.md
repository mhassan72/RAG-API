# RAG-Search API

A high-performance semantic search microservice built with Rust, providing sub-100ms latency and 99.9% availability.

## Features

- **Fast Semantic Search**: Sub-100ms p95 latency with ONNX-optimized ML models
- **High Availability**: 99.9% monthly SLA with automatic failover
- **Scalable Architecture**: Auto-scaling Cloud Run deployment (0-10 instances)
- **Multi-tier Caching**: Redis vector cache + Postgres knowledge base
- **Production Ready**: GDPR compliance, comprehensive monitoring, zero-downtime deployments

## Tech Stack

- **Language**: Rust 1.78
- **Runtime**: Tokio multi-threaded async
- **HTTP/gRPC**: Axum + Tonic
- **ML**: ONNX Runtime (CPU-optimized)
- **Cache**: Upstash Redis with vector search
- **Database**: Supabase Postgres with pgvector
- **Observability**: Prometheus + Grafana + Jaeger

## Getting Started

This project follows a spec-driven development approach. See the implementation plan in `.kiro/specs/rag-search-api/tasks.md` for detailed development tasks.

## Architecture

```
Client → Cloudflare → Cloud Run (Rust)
                        ├── Redis (Vector Cache)
                        └── Postgres (Knowledge Base)
```

## Performance Targets

- **Latency**: p50 ≤ 35ms, p95 ≤ 100ms, p99 ≤ 150ms
- **Throughput**: ≥ 3,000 RPS per instance
- **Recall**: ≥ 0.92 vs brute-force FAISS
- **Availability**: 99.9% monthly SLA

## Development

1. Follow the implementation tasks in order
2. Each task includes unit test requirements
3. Integration tests validate end-to-end functionality
4. Load testing ensures performance targets

## Compliance

- **GDPR**: Data subject deletion workflow
- **SOC-2**: Audit logging and access controls
- **Security**: TLS 1.3, mTLS, credential rotation