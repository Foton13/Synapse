# Research: Vector Databases Comparison

## Overview

Evaluated three vector databases for the semantic search layer of
Project Helios:

| Database    | License    | Embedding Storage | Query Speed |
|-------------|------------|-------------------|-------------|
| **ChromaDB**    | Apache-2.0 | Local / Cloud     | ~15 ms      |
| **Pinecone**    | Proprietary| Cloud only        | ~8 ms       |
| **Weaviate**    | BSD-3      | Self-hosted       | ~12 ms      |

## Decision

We chose **ChromaDB** because:
1. It is open-source and can run locally without cloud dependencies.
2. The Python SDK integrates seamlessly with **LangChain**.
3. Persistent storage is simple — a single directory on disk.

## Integration Notes

- ChromaDB uses **sentence-transformers** by default for embeddings.
- For production, consider switching to **OpenAI Ada** embeddings
  for higher accuracy on domain-specific text.
