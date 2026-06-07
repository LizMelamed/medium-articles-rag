# Medium Article RAG Assistant

This project implements the assignment requirements for a Medium-article RAG assistant:

- `POST /api/prompt`
- `GET /api/stats`
- Pinecone vector search
- OpenAI-compatible embedding and chat calls using the course model names
- strict system prompt that answers only from retrieved Medium article context

## Chosen RAG Hyperparameters

These are exposed by `/api/stats`:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 12
}
```

Rationale: 512 approximate word tokens is safely below the 1024-token assignment limit, 20% overlap preserves continuity across passage boundaries, and top-k 12 gives enough retrieved articles for multi-result questions while staying well below the top-k limit of 30.

The app internally queries Pinecone for `INTERNAL_TOP_K=48` candidates, deduplicates by `article_id`, and then returns the best `TOP_K=12` distinct articles. This improves multi-result questions because repeated chunks from the same article do not crowd out other relevant articles.

Each embedded chunk is prefixed with metadata in this format:

```text
Title: <title> | Authors: <authors> | Tags: <tags>

<chunk_text>
```

For topic-list questions such as "List exactly 3 articles about education", the retrieval query is lightly cleaned before embedding, and retrieved articles with exact title/tag matches for the topic keyword are sorted to the top of the context.

## Environment Variables

Copy `.env.example` to `.env` locally, then fill in the keys:

```bash
cp .env.example .env
```

Required values:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `EMBEDDING_MODEL=4UHRUIN-text-embedding-3-small`
- `CHAT_MODEL=4UHRUIN-gpt-5-mini`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_HOST`
- `PINECONE_API_VERSION=2025-10`
- `PINECONE_NAMESPACE=medium-articles`
- `INTERNAL_TOP_K=48`

Create the Pinecone index with dimension `1536`, because the assignment states that `4UHRUIN-text-embedding-3-small` uses 1536 default dimensions.

## Index The Dataset

Start small to control budget:

```bash
npm run index:sample
```

After verifying retrieval works, index the full dataset:

```bash
npm run index
```

Useful indexing controls:

```bash
INDEX_LIMIT=500 npm run index
INDEX_OFFSET=500 INDEX_LIMIT=500 npm run index
EMBEDDING_BATCH_SIZE=16 UPSERT_BATCH_SIZE=64 npm run index
```

The script uses stable vector IDs like `article-123-chunk-0`, so re-indexing the same articles overwrites the same Pinecone records.

## Run Locally

Install Vercel's local dev dependency:

```bash
npm install
npm run dev
```

Then test:

```bash
curl http://localhost:3000/api/stats
curl -X POST http://localhost:3000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question":"List exactly 3 articles about education. Return only the titles."}'
```

After the full dataset is indexed and the local server is running, test the four assignment question types:

```bash
npm run test:local
```

After deployment:

```bash
BASE_URL=https://your-app.vercel.app npm run test:prod
```

## Deploy To Vercel

1. Push this folder to a public GitHub repository.
2. Import the repository into Vercel.
3. Add the same environment variables in Vercel Project Settings.
4. Deploy and submit the public Vercel URL plus the public GitHub URL.
5. Keep the Pinecone index active until grading is complete.

## Required Response Shape

`POST /api/prompt` accepts:

```json
{
  "question": "Your natural language question here"
}
```

It returns:

```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "article chunk retrieved",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```
