# Semantic Search in Rust

A production-grade semantic search microservice engineered in Rust that transforms natural language queries into meaningful search results through advanced vector similarity matching. This system combines the performance benefits of Rust's zero-cost abstractions with state-of-the-art machine learning models to deliver sub-100ms search latency at scale.

## What is Semantic Search?

Unlike traditional keyword-based search that matches exact terms, semantic search understands the *meaning* and *intent* behind queries. It uses neural networks to convert text into high-dimensional vector representations (embeddings) that capture semantic relationships. This enables finding relevant content even when queries use different words than the target documents.

**Example**: A query for "car maintenance" can find documents about "vehicle servicing," "automobile repair," and "automotive care" because these concepts are semantically similar in vector space.

## System Overview

This microservice implements a complete semantic search pipeline:

1. **Query Processing**: Tokenizes and normalizes input text using HuggingFace tokenizers
2. **Embedding Generation**: Converts queries to 384-dimensional vectors using ONNX-optimized transformer models
3. **Vector Search**: Performs parallel similarity search across Redis (HNSW) and Postgres (IVFFlat) indexes
4. **Result Reranking**: Optional cross-encoder reranking for improved relevance
5. **Response Formatting**: Returns ranked results with metadata and similarity scores

The system is designed for production workloads with sub-100ms latency, 99.9% availability, and horizontal scalability.

## Key Features

### 🚀 **Performance & Scalability**
- **Ultra-Low Latency**: Sub-100ms p95 response times through optimized Rust implementation
- **High Throughput**: 3,000+ requests per second per instance with async processing
- **Auto-Scaling**: Cloud Run deployment scales from 0-10 instances based on demand
- **Memory Efficient**: Rust's zero-cost abstractions minimize memory overhead

### 🧠 **Advanced ML Pipeline**
- **Transformer Models**: ONNX-optimized sentence transformers for embedding generation
- **Dual-Stage Ranking**: Initial vector similarity + optional cross-encoder reranking
- **Model Flexibility**: Supports multiple embedding models (384d MiniLM, 768d BERT variants)
- **CPU-Optimized**: Efficient inference without GPU requirements

### 🔍 **Hybrid Vector Search**
- **Parallel Search**: Simultaneous queries across Redis HNSW and Postgres IVFFlat indexes
- **Smart Fallback**: Automatic degradation when one search backend is unavailable
- **Result Merging**: Intelligent deduplication and score normalization across sources
- **Configurable Recall**: Tunable search parameters for precision/recall trade-offs

### 💾 **Multi-Tier Caching**
- **Vector Cache**: Permanent LRU cache for frequently accessed embeddings
- **Query Cache**: 60-second TTL for repeated search queries
- **Metadata Cache**: 24-hour TTL for post metadata and snippets
- **Cache Invalidation**: GDPR-compliant data deletion workflows

### 🛡️ **Production Hardening**
- **Circuit Breakers**: Automatic failure detection and recovery
- **Rate Limiting**: Configurable request throttling and burst protection  
- **Comprehensive Monitoring**: Prometheus metrics, structured logging, distributed tracing
- **Security**: Input validation, CORS headers, request size limits

## Tech Stack

- **Language**: Rust 1.78
- **Runtime**: Tokio multi-threaded async
- **HTTP/gRPC**: Axum + Tonic
- **ML**: ONNX Runtime (CPU-optimized)
- **Cache**: Upstash Redis with vector search
- **Database**: Supabase Postgres with pgvector
- **Observability**: Prometheus + Grafana + Jaeger

## Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd rag-search-api

# Build the project
cargo build

# Run tests
cargo test

# Start the server
cargo run
```

## Architecture

### High-Level System Design
```
┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
│   Client    │───▶│  Cloudflare  │───▶│    Cloud Run        │
│ Application │    │   (CDN/WAF)  │    │   (Rust Service)    │
└─────────────┘    └──────────────┘    └─────────────────────┘
                                                   │
                                        ┌──────────┴──────────┐
                                        ▼                     ▼
                                ┌──────────────┐    ┌─────────────────┐
                                │    Redis     │    │   Supabase      │
                                │ (Vector Cache│    │  (Postgres +    │
                                │  + HNSW)     │    │   pgvector)     │
                                └──────────────┘    └─────────────────┘
```

### Internal Service Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Rust Microservice                        │
├─────────────────────────────────────────────────────────────────┤
│  HTTP/gRPC Layer (Axum + Tonic)                               │
│  ├── Request Validation & Rate Limiting                        │
│  ├── Authentication & CORS                                     │
│  └── Response Formatting & Error Handling                      │
├─────────────────────────────────────────────────────────────────┤
│  Search Orchestration Layer                                    │
│  ├── Query Processing & Tokenization                          │
│  ├── Parallel Vector Search Coordination                       │
│  ├── Result Merging & Deduplication                           │
│  └── Optional Cross-Encoder Reranking                         │
├─────────────────────────────────────────────────────────────────┤
│  ML Inference Layer (ONNX Runtime)                            │
│  ├── BiEncoder (Query → Vector Embedding)                     │
│  ├── CrossEncoder (Query + Document → Relevance Score)        │
│  └── Model Loading & Caching                                  │
├─────────────────────────────────────────────────────────────────┤
│  Storage & Caching Layer                                       │
│  ├── Redis Client (Vector + Metadata Cache)                   │
│  ├── Postgres Client (Knowledge Base + pgvector)              │
│  └── Multi-Tier Cache Management                              │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                          │
│  ├── Circuit Breakers & Retry Logic                           │
│  ├── Connection Pooling & Resource Management                  │
│  ├── Metrics Collection (Prometheus)                          │
│  └── Distributed Tracing (OpenTelemetry)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Search Flow
1. **Request Processing**: Validate query, apply rate limits, extract parameters
2. **Tokenization**: Normalize text and generate tokens using HuggingFace tokenizers  
3. **Embedding**: Convert query to 384-dimensional vector using ONNX transformer
4. **Parallel Search**: Query Redis HNSW and Postgres IVFFlat indexes simultaneously
5. **Result Merging**: Combine and deduplicate candidates, sort by similarity score
6. **Reranking** (optional): Apply cross-encoder for improved relevance scoring
7. **Response**: Format results with metadata, apply GDPR truncation, return JSON

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