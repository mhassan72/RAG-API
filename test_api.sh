#!/bin/bash

echo "Testing RAG Search API"
echo "====================="

# Test health endpoint
echo "Testing health endpoint..."
curl -X GET http://localhost:8080/health \
  -H "Content-Type: application/json" \
  --connect-timeout 5 \
  --max-time 10

echo -e "\n"

# Test search endpoint with valid request
echo "Testing search endpoint with valid request..."
curl -X POST http://localhost:8080/semantic-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "k": 5,
    "min_score": 0.7,
    "rerank": false,
    "filters": {
      "language": "en",
      "frozen": false
    }
  }' \
  --connect-timeout 5 \
  --max-time 10

echo -e "\n"

# Test search endpoint with invalid request (empty query)
echo "Testing search endpoint with invalid request (empty query)..."
curl -X POST http://localhost:8080/semantic-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "",
    "k": 5
  }' \
  --connect-timeout 5 \
  --max-time 10

echo -e "\n"

# Test search endpoint with invalid k parameter
echo "Testing search endpoint with invalid k parameter..."
curl -X POST http://localhost:8080/semantic-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "k": 0
  }' \
  --connect-timeout 5 \
  --max-time 10

echo -e "\n"
echo "API tests completed!"