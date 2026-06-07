import {
  answerWithContext,
  assertRuntimeConfig,
  getConfig,
  retrieveContext
} from "../lib/rag.js";

export default async function handler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Method not allowed" });
  }

  try {
    const question = String(request.body?.question || "").trim();
    if (!question) {
      return response.status(400).json({ error: "Request body must include a non-empty question." });
    }

    const config = getConfig();
    assertRuntimeConfig(config);

    const context = await retrieveContext(question, config);
    const answer = await answerWithContext(question, context, config);
    const responseContext = context.map((item) => ({
      article_id: item.article_id,
      title: item.title,
      chunk: item.chunk,
      score: item.score
    }));

    return response.status(200).json({
      response: answer.response,
      context: responseContext,
      Augmented_prompt: answer.augmentedPrompt
    });
  } catch (error) {
    return response.status(500).json({ error: error.message });
  }
}
