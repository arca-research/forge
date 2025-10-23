# Test Wishlist

### Graph Building
- test graph extraction from single document
- test graph extraction from multiple documents
- test graph extraction with oversized documents (chunking)
- test entity deduplication across documents
- test relationship extraction between entities
- test claim attribution to correct sources
- test alias resolution at insertion time
- test handling of entity name case sensitivity
- test circular relationship detection
- test self-referential entity prevention

### Vector DB Building
- test vector indexing from single document
- test vector indexing from multiple documents
- test chunk overlap handling
- test embedding generation with different models
- test incremental index updates
- test rebuild vs update modes
- test handling of empty documents
- test handling of very short documents
- test handling of very long documents

### State Management
- test SQLite index initialization
- test concurrent read/write operations
- test index corruption recovery
- test state persistence across sessions
- test foreign key constraints enforcement
- test WAL mode operation
- test index size growth patterns

### Entity & Relationship Operations
- test lazy loading of entity claims
- test lazy loading of entity relationships
- test lazy loading of relationship claims
- test entity equality with aliases
- test relationship equality (directed vs undirected)
- test entity hashing for sets/dicts
- test relationship strength filtering
- test canonical name resolution
- test handling of missing entities
- test handling of invalid relationship targets

### Query Operations
- test vector similarity search
- test vector search with filters
- test graph traversal by depth
- test graph traversal by relationship type
- test entity lookup by canonical name
- test entity lookup by alias
- test relationship filtering by strength threshold
- test combined vector + graph queries
- test pagination of large result sets

### Consolidation & Maintenance
- test manual entity merging
- test alias creation
- test claim pruning
- test relationship strength recalculation
- test orphaned entity cleanup
- test duplicate claim detection

### Edge Cases & Error Handling
- test empty string entity names
- test special characters in entity names
- test extremely long entity names
- test null/None handling in claims
- test malformed documents
- test missing source attribution
- test extraction failures (LLM errors)
- test embedding failures
- test database lock timeout handling
- test out of memory scenarios with large graphs

### Async/Sync Variants
- test sync graph building
- test async graph building
- test sync vector building
- test async vector building
- test mixed sync/async operations
- test async batch processing
- test async error propagation

### Integration Tests
- test end-to-end: ingest → build → query
- test multi-document workflow
- test incremental updates after initial build
- test switching between embedding models
- test migration between schema versions
- test export/import of graph data
- test interop between vector and graph queries

### Performance & Scaling
- test build time with 100 documents
- test build time with 1000 documents
- test query latency with small graph
- test query latency with large graph
- test memory usage during build
- test index size vs document count
- test concurrent query performance

### Configuration & Setup
- test custom stage_dir location
- test custom data_dir location
- test rebuild flag behavior
- test invalid config handling
- test missing directory creation
- test permission errors on directories