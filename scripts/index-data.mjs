import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  assertRuntimeConfig,
  chunkText,
  createEmbeddings,
  getConfig,
  upsertPinecone
} from "../lib/rag.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");

loadEnv(path.join(projectRoot, ".env"));

const csvPath = process.env.CSV_PATH || path.join(projectRoot, "medium-english-50mb.csv");
const limit = parseOptionalInteger(process.env.INDEX_LIMIT);
const offset = Number.parseInt(process.env.INDEX_OFFSET || "0", 10);
const embeddingBatchSize = Number.parseInt(process.env.EMBEDDING_BATCH_SIZE || "16", 10);
const upsertBatchSize = Number.parseInt(process.env.UPSERT_BATCH_SIZE || "64", 10);
const config = getConfig();

assertRuntimeConfig(config);

console.log(`Reading ${csvPath}`);
const rows = parseCsv(fs.readFileSync(csvPath, "utf8"));
const [header, ...records] = rows;
const selectedRecords = records.slice(offset, limit ? offset + limit : undefined);
const columnIndex = Object.fromEntries(header.map((name, index) => [name, index]));

console.log(`Preparing ${selectedRecords.length} articles from offset ${offset}`);
let pendingInputs = [];
let pendingMetadata = [];
let pendingVectors = [];
let embeddedChunks = 0;

for (let localArticleIndex = 0; localArticleIndex < selectedRecords.length; localArticleIndex += 1) {
  const record = selectedRecords[localArticleIndex];
  const articleIndex = offset + localArticleIndex;
  const article = {
    article_id: String(articleIndex),
    title: value(record, columnIndex.title),
    text: value(record, columnIndex.text),
    url: value(record, columnIndex.url),
    authors: value(record, columnIndex.authors),
    timestamp: value(record, columnIndex.timestamp),
    tags: value(record, columnIndex.tags)
  };

  const chunks = chunkText(article.text, config.chunkSize, config.overlapRatio);
  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    const chunk = chunks[chunkIndex];
    const id = `article-${article.article_id}-chunk-${chunkIndex}`;
    pendingInputs.push(formatEmbeddingInput(article, chunk));
    pendingMetadata.push({
      id,
      metadata: {
        article_id: article.article_id,
        chunk_index: chunkIndex,
        title: article.title,
        authors: article.authors,
        url: article.url,
        timestamp: article.timestamp,
        tags: article.tags,
        chunk
      }
    });

    if (pendingInputs.length >= embeddingBatchSize) {
      await embedAndQueueVectors();
    }
  }
}

if (pendingInputs.length) {
  await embedAndQueueVectors();
}

if (pendingVectors.length) {
  await flushVectors();
}

console.log(`Done. Embedded and upserted ${embeddedChunks} chunks into namespace "${config.pineconeNamespace}".`);

async function embedAndQueueVectors() {
  const inputs = pendingInputs;
  const metadata = pendingMetadata;
  pendingInputs = [];
  pendingMetadata = [];

  const embeddings = await createEmbeddings(inputs, config);
  for (let index = 0; index < embeddings.length; index += 1) {
    pendingVectors.push({
      id: metadata[index].id,
      values: embeddings[index],
      metadata: metadata[index].metadata
    });
  }

  embeddedChunks += embeddings.length;
  if (pendingVectors.length >= upsertBatchSize) {
    await flushVectors();
  }

  if (embeddedChunks % 100 === 0) {
    console.log(`Embedded ${embeddedChunks} chunks...`);
  }
}

async function flushVectors() {
  const vectors = pendingVectors;
  pendingVectors = [];
  await upsertPinecone(vectors, config);
  console.log(`Upserted ${vectors.length} chunks`);
}

function loadEnv(envPath) {
  if (!fs.existsSync(envPath)) return;

  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const equalsIndex = trimmed.indexOf("=");
    if (equalsIndex === -1) continue;

    const key = trimmed.slice(0, equalsIndex).trim();
    let val = trimmed.slice(equalsIndex + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }

    if (!process.env[key]) {
      process.env[key] = val;
    }
  }
}

function parseOptionalInteger(valueToParse) {
  if (!valueToParse) return undefined;
  const parsed = Number.parseInt(valueToParse, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function value(record, index) {
  return index === undefined ? "" : String(record[index] || "");
}

function formatEmbeddingInput(article, chunk) {
  return `Title: ${article.title} | Authors: ${article.authors} | Tags: ${article.tags}\n\n${chunk}`;
}

function parseCsv(csvText) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let index = 0; index < csvText.length; index += 1) {
    const char = csvText[index];
    const next = csvText[index + 1];

    if (inQuotes) {
      if (char === '"' && next === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }

  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}
