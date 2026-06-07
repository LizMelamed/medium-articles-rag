# Medium Article RAG Assistant

A small RAG project for answering questions over the provided Medium articles CSV.

The app uses:

- Pinecone for vector search
- `4UHRUIN-text-embedding-3-small` for embeddings
- `4UHRUIN-gpt-5-mini` for the final answer
- Vercel serverless API routes

The assistant is instructed to answer only from retrieved Medium article context. If the retrieved data is not enough, it should say that it does not know based on the provided data.

## API

### `GET /api/stats`

Returns the active RAG settings:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 12
}
```

### `POST /api/prompt`

Request:

```json
{
  "question": "List exactly 3 articles about education. Return only the titles."
}
```

Response:

```json
{
  "response": "Final answer from 4UHRUIN-gpt-5-mini.",
  "context": [
    {
      "article_id": "5338",
      "title": "Finding the Whole Child in Education Reform",
      "chunk": "Retrieved article passage...",
      "score": 0.4472
    }
  ],
  "Augmented_prompt": {
    "System": "System prompt sent to the chat model",
    "User": "User prompt with retrieved context"
  }
}
```

## RAG Settings

I used a chunk size of `512` words with `20%` overlap. This keeps chunks safely below the assignment limit of 1024 tokens while preserving some context between chunks.

For retrieval, the public `top_k` is `12`. Internally, the app asks Pinecone for `48` candidates first, deduplicates by `article_id`, and then keeps the best 12 distinct articles. This helps avoid returning many chunks from the same article.

Each embedded chunk includes basic article metadata:

```text
Title: <title> | Authors: <authors> | Tags: <tags>

<chunk_text>
```

This helps topic queries match article titles and tags, not only body text.

## Environment

Create `.env` from the example file:

```bash
cp .env.example .env
```

Required values:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.llmod.ai
EMBEDDING_MODEL=4UHRUIN-text-embedding-3-small
CHAT_MODEL=4UHRUIN-gpt-5-mini

PINECONE_API_KEY=
PINECONE_INDEX_HOST=
PINECONE_API_VERSION=2025-10
PINECONE_NAMESPACE=medium-articles-512

CHUNK_SIZE=512
OVERLAP_RATIO=0.2
TOP_K=12
INTERNAL_TOP_K=48
```

The Pinecone index should use dimension `1536` and cosine similarity.

## Indexing

Small test run:

```bash
npm run index:sample
```

Index 500 articles:

```bash
npm run index:500
```

Index the full CSV:

```bash
npm run index
```

The indexing script uses stable vector IDs such as `article-123-chunk-0`, so rerunning indexing overwrites the same records instead of creating duplicates.

## Local Testing

Install dependencies:

```bash
npm install
```

Run locally:

```bash
npm run dev
```

Test the four assignment-style questions:

```bash
npm run test:direct
```

For a deployed URL:

```bash
BASE_URL=https://your-app.vercel.app npm run test:prod
```

## Current Index

The full dataset has been indexed into Pinecone namespace:

```text
medium-articles-512
```

The latest full index contained about `22,174` vectors.
