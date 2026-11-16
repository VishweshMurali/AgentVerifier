// Key-holding proxy for the hosted agent (Cloudflare Worker, free plan).
// It keeps GROQ_API_KEY secret and forwards chat requests from the engine to Groq.
// Secrets (Worker settings > Variables and Secrets, type "Secret", never in this file):
//   GROQ_API_KEY  your Groq key
//   PROXY_TOKEN   a long random string; the engine sends it in the x-proxy-token header
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const MODELS = new Set(["openai/gpt-oss-20b"]); // only the model the site uses
const MAX_BYTES = 200 * 1024;

function same(a, b) {
  // constant-time comparison of two strings
  if (a.length !== b.length) return false;
  let r = 0;
  for (let i = 0; i < a.length; i++) r |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return r === 0;
}

function plain(status, message) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return new Response("ok"); // no secrets involved: tells you the Worker is deployed
    }
    if (request.method !== "POST" || url.pathname !== "/v1/chat/completions") {
      return plain(404, "not found");
    }
    if (!env.GROQ_API_KEY || !env.PROXY_TOKEN) return plain(500, "proxy is not configured");
    const sent = request.headers.get("x-proxy-token") || "";
    if (!same(sent, env.PROXY_TOKEN)) return plain(401, "missing or wrong x-proxy-token");

    const body = await request.text();
    if (body.length > MAX_BYTES) return plain(413, "request too large");
    let model;
    try {
      model = JSON.parse(body).model;
    } catch {
      return plain(400, "send JSON");
    }
    if (!MODELS.has(model)) return plain(400, "model not allowed");

    const upstream = await fetch(GROQ_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${env.GROQ_API_KEY}`,
      },
      body,
    });
    // Groq's answer, status and rate-limit headers are returned unchanged (the engine paces on them)
    const headers = new Headers();
    for (const [k, v] of upstream.headers) {
      if (k === "content-type" || k.startsWith("x-ratelimit") || k === "retry-after") {
        headers.set(k, v);
      }
    }
    return new Response(upstream.body, { status: upstream.status, headers });
  },
};
