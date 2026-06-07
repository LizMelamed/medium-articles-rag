import { getPublicStats, validateHyperparameters } from "../lib/rag.js";

export default function handler(request, response) {
  if (request.method !== "GET") {
    response.setHeader("Allow", "GET");
    return response.status(405).json({ error: "Method not allowed" });
  }

  try {
    validateHyperparameters();
    return response.status(200).json(getPublicStats());
  } catch (error) {
    return response.status(500).json({ error: error.message });
  }
}
