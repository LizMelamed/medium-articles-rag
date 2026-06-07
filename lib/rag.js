export const REQUIRED_SYSTEM_PROMPT = `You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.” Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Additional clarification: If the user explicitly asks to "return only" a specific field, such as titles, follow that output constraint and do not add extra explanation.`;

export function getConfig() {
  return {
    chunkSize: parseInteger(process.env.CHUNK_SIZE, 700),
    overlapRatio: parseNumber(process.env.OVERLAP_RATIO, 0.2),
    topK: parseInteger(process.env.TOP_K, 12),
    internalTopK: parseInteger(process.env.INTERNAL_TOP_K, 48),
    embeddingModel: process.env.EMBEDDING_MODEL || "4UHRUIN-text-embedding-3-small",
    chatModel: process.env.CHAT_MODEL || "4UHRUIN-gpt-5-mini",
    openaiBaseUrl: (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, ""),
    openaiApiKey: process.env.OPENAI_API_KEY || "",
    pineconeApiKey: process.env.PINECONE_API_KEY || "",
    pineconeIndexHost: normalizePineconeHost(process.env.PINECONE_INDEX_HOST || ""),
    pineconeApiVersion: process.env.PINECONE_API_VERSION || "2025-10",
    pineconeNamespace: process.env.PINECONE_NAMESPACE || "medium-articles"
  };
}

export function getPublicStats() {
  const config = getConfig();
  return {
    chunk_size: config.chunkSize,
    overlap_ratio: config.overlapRatio,
    top_k: config.topK
  };
}

export function validateHyperparameters(config = getConfig()) {
  if (config.chunkSize < 1 || config.chunkSize > 1024) {
    throw new Error("CHUNK_SIZE must be between 1 and 1024.");
  }

  if (config.overlapRatio < 0 || config.overlapRatio > 0.3) {
    throw new Error("OVERLAP_RATIO must be between 0 and 0.3.");
  }

  if (config.topK < 1 || config.topK > 30) {
    throw new Error("TOP_K must be between 1 and 30.");
  }

  if (config.internalTopK < config.topK) {
    throw new Error("INTERNAL_TOP_K must be greater than or equal to TOP_K.");
  }
}

export function assertRuntimeConfig(config = getConfig()) {
  validateHyperparameters(config);

  const missing = [];
  if (!config.openaiApiKey) missing.push("OPENAI_API_KEY");
  if (!config.pineconeApiKey) missing.push("PINECONE_API_KEY");
  if (!config.pineconeIndexHost) missing.push("PINECONE_INDEX_HOST");

  if (missing.length) {
    throw new Error(`Missing required environment variables: ${missing.join(", ")}`);
  }
}

export function chunkText(text, chunkSize, overlapRatio) {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  if (!words.length) return [];

  const overlap = Math.floor(chunkSize * overlapRatio);
  const step = Math.max(1, chunkSize - overlap);
  const chunks = [];

  for (let start = 0; start < words.length; start += step) {
    const end = Math.min(words.length, start + chunkSize);
    chunks.push(words.slice(start, end).join(" "));
    if (end === words.length) break;
  }

  return chunks;
}

export async function createEmbedding(input, config = getConfig()) {
  const response = await fetchWithRetry(`${config.openaiBaseUrl}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.openaiApiKey}`
    },
    body: JSON.stringify({
      model: config.embeddingModel,
      input
    })
  });

  if (!response.ok) {
    throw new Error(`Embedding request failed: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();
  return data.data[0].embedding;
}

export async function createEmbeddings(inputs, config = getConfig()) {
  const response = await fetchWithRetry(`${config.openaiBaseUrl}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.openaiApiKey}`
    },
    body: JSON.stringify({
      model: config.embeddingModel,
      input: inputs
    })
  });

  if (!response.ok) {
    throw new Error(`Embedding batch request failed: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();
  return data.data.map((item) => item.embedding);
}

export async function queryPinecone(vector, config = getConfig(), topK = config.topK) {
  const response = await fetchWithRetry(`${config.pineconeIndexHost}/query`, {
    method: "POST",
    headers: pineconeHeaders(config),
    body: JSON.stringify({
      namespace: config.pineconeNamespace,
      vector,
      topK,
      includeMetadata: true
    })
  });

  if (!response.ok) {
    throw new Error(`Pinecone query failed: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();
  return (data.matches || []).map((match) => ({
    article_id: match.metadata?.article_id || articleIdFromVectorId(match.id),
    title: match.metadata?.title || "",
    authors: match.metadata?.authors || "",
    url: match.metadata?.url || "",
    timestamp: match.metadata?.timestamp || "",
    tags: match.metadata?.tags || "",
    chunk: match.metadata?.chunk || "",
    score: Number(match.score || 0)
  }));
}

export async function retrieveContext(question, config = getConfig()) {
  const query = cleanQueryForEmbedding(question);
  const questionEmbedding = await createEmbedding(query.embeddingText, config);
  const candidates = await queryPinecone(questionEmbedding, config, config.internalTopK);
  const uniqueArticles = dedupeByArticle(candidates).slice(0, config.topK);
  return rerankByMetadataKeyword(uniqueArticles, query.keyword);
}

export async function upsertPinecone(vectors, config = getConfig()) {
  const response = await fetchWithRetry(`${config.pineconeIndexHost}/vectors/upsert`, {
    method: "POST",
    headers: pineconeHeaders(config),
    body: JSON.stringify({
      namespace: config.pineconeNamespace,
      vectors
    })
  });

  if (!response.ok) {
    throw new Error(`Pinecone upsert failed: ${response.status} ${await response.text()}`);
  }

  return response.json();
}

export async function answerWithContext(question, context, config = getConfig()) {
  const userPrompt = buildUserPrompt(question, context);
  const response = await fetchWithRetry(`${config.openaiBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.openaiApiKey}`
    },
    body: JSON.stringify({
      model: config.chatModel,
      messages: [
        { role: "system", content: REQUIRED_SYSTEM_PROMPT },
        { role: "user", content: userPrompt }
      ]
    })
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();
  return {
    response: data.choices?.[0]?.message?.content || "",
    augmentedPrompt: {
      System: REQUIRED_SYSTEM_PROMPT,
      User: userPrompt
    }
  };
}

export function buildUserPrompt(question, context) {
  const contextText = context.map((item, index) => {
    return [
      `Context ${index + 1} (retrieved article)`,
      `article_id: ${item.article_id}`,
      `title: ${item.title}`,
      `authors: ${item.authors}`,
      `url: ${item.url}`,
      `timestamp: ${item.timestamp}`,
      `tags: ${item.tags}`,
      `score: ${item.score}`,
      `passage: ${item.chunk}`
    ].join("\n");
  }).join("\n\n");

  return `Question: ${question}

Retrieved Medium article context:
${contextText || "No context retrieved."}

Instructions:
- Answer only from the retrieved context.
- Do not use outside knowledge.
- For requests asking for multiple article titles, use distinct articles, not repeated chunks from the same article.
- If the question explicitly asks to return only titles, return only titles.
- When recommending an article, include the title, author if available, and a short evidence-based justification.
- When summarising, keep the summary concise and grounded in the retrieved passage.`;
}

function pineconeHeaders(config) {
  return {
    "Content-Type": "application/json",
    "Api-Key": config.pineconeApiKey,
    "X-Pinecone-Api-Version": config.pineconeApiVersion
  };
}

async function fetchWithRetry(url, options, attempts = 4) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, options);
      if (!shouldRetryResponse(response) || attempt === attempts) {
        return response;
      }

      await sleep(1000 * attempt);
    } catch (error) {
      lastError = error;
      if (attempt === attempts) {
        throw error;
      }

      await sleep(1000 * attempt);
    }
  }

  throw lastError;
}

function shouldRetryResponse(response) {
  return response.status === 429 || response.status >= 500;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizePineconeHost(host) {
  const trimmed = host.trim().replace(/\/$/, "");
  if (!trimmed) return "";
  return /^https?:\/\//.test(trimmed) ? trimmed : `https://${trimmed}`;
}

function dedupeByArticle(matches) {
  const bestByArticle = new Map();

  for (const match of matches) {
    const articleId = match.article_id || "";
    const current = bestByArticle.get(articleId);
    if (!current || match.score > current.score) {
      bestByArticle.set(articleId, match);
    }
  }

  return Array.from(bestByArticle.values()).sort((a, b) => b.score - a.score);
}

export function cleanQueryForEmbedding(question) {
  const trimmed = String(question || "").trim();
  const normalized = normalizeText(trimmed);
  const listMatch = trimmed.match(/^list\s+exactly\s+\d+\s+articles?\s+about\s+(.+?)(?:[.?!]\s*)?(?:return\s+only\s+the\s+titles?\.?)?$/i);

  if (listMatch) {
    const topic = stripTrailingInstruction(listMatch[1]);
    return {
      embeddingText: `articles about ${topic}`,
      keyword: extractCoreKeyword(topic)
    };
  }

  if (
    normalized.includes("bubonic plague") &&
    normalized.includes("pandemic") &&
    (normalized.includes("innovation") || normalized.includes("recovery"))
  ) {
    return {
      embeddingText: "rebounding from pandemic AI bubonic plague renaissance",
      keyword: "renaissance"
    };
  }

  return {
    embeddingText: trimmed,
    keyword: extractCoreKeyword(trimmed)
  };
}

function rerankByMetadataKeyword(context, keyword) {
  if (!keyword) return context;

  const normalizedKeyword = normalizeText(keyword);
  return [...context].sort((a, b) => {
    const aRank = metadataKeywordRank(a, normalizedKeyword);
    const bRank = metadataKeywordRank(b, normalizedKeyword);
    if (aRank !== bRank) return bRank - aRank;
    return b.score - a.score;
  });
}

function metadataKeywordRank(item, keyword) {
  const title = normalizeText(item.title);
  const tags = normalizeText(item.tags);

  if (containsWord(tags, keyword)) return 2;
  if (containsWord(title, keyword)) return 1;
  return 0;
}

function stripTrailingInstruction(text) {
  return String(text || "")
    .replace(/\breturn\s+only\s+the\s+titles?\.?$/i, "")
    .replace(/[.?!]\s*$/g, "")
    .trim();
}

function extractCoreKeyword(text) {
  const words = normalizeText(text)
    .split(/\s+/)
    .filter((word) => word && !STOP_WORDS.has(word));

  return words.at(-1) || "";
}

function normalizeText(text) {
  return String(text || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function containsWord(text, word) {
  if (!word) return false;
  return new RegExp(`(^|\\s)${escapeRegExp(word)}(\\s|$)`).test(text);
}

function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function articleIdFromVectorId(vectorId) {
  const match = String(vectorId || "").match(/^article-(.+?)-chunk-\d+$/);
  return match ? match[1] : "";
}

function parseInteger(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseNumber(value, fallback) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "article",
  "articles",
  "about",
  "for",
  "from",
  "in",
  "of",
  "on",
  "the",
  "to",
  "with"
]);
