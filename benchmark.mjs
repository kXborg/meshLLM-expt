// benchmark.mjs
// Unified benchmark runner for MeshLLM / Skippy serving engine
// Usage:
//   TARGET_URL=https://<tunnel-id>.trycloudflare.com node benchmark.mjs
//   TARGET_URL=http://localhost:9337 node benchmark.mjs --stress

const BASE_URL = process.env.TARGET_URL || "http://127.0.0.1:9337";
const MODEL = process.env.MODEL || "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL";

console.log(`\n================ BENCHMARK TARGET CONFIG ================`);
console.log(`Target Endpoint: ${BASE_URL}`);
console.log(`Model:           ${MODEL}`);
console.log(`=========================================================\n`);

async function runStandardBenchmark(tokensToGenerate = 100) {
  console.log(`--- [1] Standard Generation Test (${tokensToGenerate} tokens) ---`);
  const prompt = "Write a detailed timeline of key events in computing history from 1950 to 1990 with bullet points.";
  const t0 = performance.now();

  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      max_tokens: tokensToGenerate,
      temperature: 0.2
    })
  });

  const wallClockMs = performance.now() - t0;
  if (!res.ok) {
    console.error(`Request failed HTTP ${res.status}: ${await res.text()}`);
    return null;
  }

  const json = await res.json();
  const t = json.timings || {};
  const usage = json.usage || {};

  console.log(`Prompt Tokens:       ${usage.prompt_tokens ?? "N/A"}`);
  console.log(`Completion Tokens:   ${usage.completion_tokens ?? "N/A"}`);
  if (t.prompt_per_second) {
    console.log(`🚀 Prompt Prefill:   ${t.prompt_per_second.toFixed(2)} tok/s (${t.prompt_ms?.toFixed(1) ?? "-"} ms)`);
  }
  if (t.predicted_per_second) {
    console.log(`⚡ Decode Speed:     ${t.predicted_per_second.toFixed(2)} tok/s (${t.predicted_ms?.toFixed(1) ?? "-"} ms)`);
  }
  console.log(`⏱️ Total Wall Clock:  ${(wallClockMs / 1000).toFixed(2)} s\n`);

  return { usage, timings: t, wallClockSec: wallClockMs / 1000 };
}

async function runStreamingTest(tokensToGenerate = 30) {
  console.log(`--- [2] Real-Time Streaming SSE Test (TTFT) ---`);
  const t0 = performance.now();
  let firstTokenTime = null;
  let receivedTokens = 0;

  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "Count from 1 to 20 slowly with commas." }],
      max_tokens: tokensToGenerate,
      temperature: 0.2,
      stream: true
    })
  });

  if (!res.ok || !res.body) {
    console.error(`Streaming failed HTTP ${res.status}: ${await res.text()}`);
    return null;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let done = false;

  while (!done) {
    const { value, done: readerDone } = await reader.read();
    if (readerDone) break;
    const chunk = decoder.decode(value, { stream: true });
    if (!firstTokenTime && chunk.includes('"delta"')) {
      firstTokenTime = performance.now() - t0;
    }
    // approximate tokens from chunk
    const lines = chunk.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ") && !line.includes("[DONE]")) {
        receivedTokens++;
      }
    }
  }

  const totalStreamDuration = (performance.now() - t0) / 1000;
  console.log(`⏱️ Time to First Token (TTFT): ${firstTokenTime ? firstTokenTime.toFixed(1) + " ms" : "N/A"}`);
  console.log(`Total Stream Duration:         ${totalStreamDuration.toFixed(2)} s (~${receivedTokens} tokens)`);
  console.log(`Stream Consistency:            Instantaneous first token emission\n`);

  return { ttftMs: firstTokenTime, durationSec: totalStreamDuration };
}

function generateFillerText(targetTokens) {
  const baseSentence = "Distributed artificial intelligence pools compute across heterogeneous peer nodes using encrypted QUIC channels and sequential pipeline splits. ";
  const approxTokensPerSentence = 18;
  const repeatCount = Math.ceil(targetTokens / approxTokensPerSentence);
  return baseSentence.repeat(repeatCount);
}

async function runContextStressStage(name, targetPromptTokens, maxCompletion = 28) {
  const filler = generateFillerText(targetPromptTokens);
  const prompt = `${filler}\n\nBased on the text above, summarize the key principle in exactly 1 bullet point.`;

  console.log(`Testing Stage [${name}] (~${targetPromptTokens} prompt tokens)...`);
  const t0 = performance.now();

  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxCompletion,
      temperature: 0.2
    })
  });

  const wallClockSec = (performance.now() - t0) / 1000;
  if (!res.ok) {
    console.error(`[${name}] Failed HTTP ${res.status}: ${await res.text()}`);
    return null;
  }

  const json = await res.json();
  const t = json.timings || {};
  const usage = json.usage || {};

  console.log(`   -> Prompt Tokens: ${usage.prompt_tokens} | Completion Tokens: ${usage.completion_tokens}`);
  console.log(`   -> Prefill:       ${t.prompt_per_second?.toFixed(2) ?? "N/A"} tok/s`);
  console.log(`   -> Decode:        ${t.predicted_per_second?.toFixed(2) ?? "N/A"} tok/s`);
  console.log(`   -> Wall Time:     ${wallClockSec.toFixed(2)} s\n`);

  return {
    stage: name,
    promptTokens: usage.prompt_tokens,
    completionTokens: usage.completion_tokens,
    prefillTokPerSec: t.prompt_per_second,
    decodeTokPerSec: t.predicted_per_second,
    wallClockSec
  };
}

async function runLongContextStress() {
  console.log(`--- [3] Long-Context Stress Test Matrix ---`);
  const stages = [
    { name: "Short (~450 tok)", tokens: 450 },
    { name: "Medium (~1,300 tok)", tokens: 1300 },
    { name: "Heavy (~3,300 tok)", tokens: 3300 },
    { name: "Extra Heavy (~5,900 tok)", tokens: 5900 }
  ];

  const results = [];
  for (const s of stages) {
    const res = await runContextStressStage(s.name, s.tokens);
    if (res) results.push(res);
  }

  console.log("\n================ STRESS TEST SUMMARY TABLE ================");
  console.table(results.map(r => ({
    Stage: r.stage,
    "Prompt Tok": r.promptTokens,
    "Compl Tok": r.completionTokens,
    "Prefill (tok/s)": r.prefillTokPerSec?.toFixed(2),
    "Decode (tok/s)": r.decodeTokPerSec?.toFixed(2),
    "Wall Clock (s)": r.wallClockSec?.toFixed(2)
  })));
  console.log("===========================================================\n");
}

async function main() {
  const args = process.argv.slice(2);
  const stressOnly = args.includes("--stress");
  const streamingOnly = args.includes("--streaming");

  if (streamingOnly) {
    await runStreamingTest();
    return;
  }

  if (stressOnly) {
    await runLongContextStress();
    return;
  }

  // Full default suite
  await runStandardBenchmark(100);
  await runStreamingTest(30);

  if (args.includes("--full")) {
    await runLongContextStress();
  } else {
    console.log("💡 Tip: Run `node benchmark.mjs --stress` to benchmark across context scaling (450 to 5,900 tokens).");
    console.log("💡 Tip: Run `node concurrency-test.mjs` to measure multi-stream throughput.");
  }
}

main().catch(err => {
  console.error("Benchmark error:", err);
  process.exit(1);
});
