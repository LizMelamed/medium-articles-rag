import fs from "node:fs";

const BASE_URL = (process.env.BASE_URL || "http://localhost:3000").replace(/\/$/, "");
const DIRECT_API = process.env.DIRECT_API === "1";

const QUESTIONS = [
  {
    type: "Precise fact retrieval",
    question: "Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author."
  },
  {
    type: "Multi-result topic listing",
    question: "List exactly 3 articles about education. Return only the titles."
  },
  {
    type: "Key idea summary extraction",
    question: "Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument."
  },
  {
    type: "Recommendation with evidence-based justification",
    question: "I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?"
  }
];

async function main() {
  loadEnv();
  console.log(`Target: ${DIRECT_API ? "direct API handlers" : BASE_URL}\n`);
  await testStats();

  for (const item of QUESTIONS) {
    console.log(`\n=== ${item.type} ===`);
    console.log(`Q: ${item.question}`);

    try {
      const result = await postPrompt(item.question);
      console.log("\nResponse:");
      console.log(result.response);

      const context = Array.isArray(result.context) ? result.context : [];
      const uniqueArticles = new Set(context.map((entry) => entry.article_id));
      console.log(`\nContext: ${context.length} chunks from ${uniqueArticles.size} unique articles`);
      for (const entry of context) {
        const score = Number(entry.score || 0).toFixed(4);
        console.log(`- [${score}] ${entry.title} (${entry.article_id})`);
      }

      const prompt = result.Augmented_prompt || {};
      console.log(`Augmented prompt present: ${Boolean(prompt.System && prompt.User)}`);
    } catch (error) {
      console.error(`Error: ${error.message}`);
    }
  }
}

async function testStats() {
  console.log("=== GET /api/stats ===");
  try {
    if (DIRECT_API) {
      console.log(JSON.stringify(await callHandler("./../api/stats.js", { method: "GET" }), null, 2));
      return;
    }

    const response = await fetch(`${BASE_URL}/api/stats`);
    const text = await response.text();
    console.log(text);
  } catch (error) {
    console.error(`Stats error: ${error.message}`);
  }
}

async function postPrompt(question) {
  if (DIRECT_API) {
    return callHandler("./../api/prompt.js", { method: "POST", body: { question } });
  }

  const response = await fetch(`${BASE_URL}/api/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  return response.json();
}

async function callHandler(modulePath, request) {
  const { default: handler } = await import(modulePath);
  let payload;
  const response = {
    statusCode: 200,
    headers: {},
    setHeader(key, value) {
      this.headers[key] = value;
    },
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(value) {
      payload = value;
      return this;
    }
  };

  await handler(request, response);
  if (response.statusCode >= 400) {
    throw new Error(`HTTP ${response.statusCode}: ${JSON.stringify(payload)}`);
  }
  return payload;
}

function loadEnv() {
  if (typeof process.loadEnvFile === "function") {
    try {
      process.loadEnvFile(".env");
      return;
    } catch {
      // Fall back to the small parser below.
    }
  }

  try {
    for (const line of fs.readFileSync(".env", "utf8").split(/\r?\n/)) {
      if (!line || line.startsWith("#") || !line.includes("=")) continue;
      const separator = line.indexOf("=");
      const key = line.slice(0, separator);
      const value = line.slice(separator + 1).trim();
      if (!process.env[key]) process.env[key] = value;
    }
  } catch {
    // The HTTP mode may not need local env values.
  }
}

main().catch((error) => {
  console.error(`Fatal: ${error.message}`);
  process.exit(1);
});
