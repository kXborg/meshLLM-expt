// concurrency-test.mjs
const BASE_URL = process.env.TARGET_URL || "http://127.0.0.1:9337";
const MODEL = "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL";

async function sendRequest(id, prompt, maxTokens = 50) {
  const t0 = performance.now();
  console.log(`[Req ${id}] Dispatched: "${prompt}"`);

  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature: 0.2
    })
  });

  const wallClock = (performance.now() - t0) / 1000;
  if (!res.ok) {
    console.error(`[Req ${id}] Error HTTP ${res.status}: ${await res.text()}`);
    return null;
  }

  const json = await res.json();
  const t = json.timings;
  console.log(`[Req ${id}] Finished in ${wallClock.toFixed(2)}s | Generated: ${json.usage.completion_tokens} tok | Queue Wait: ${t.queue_wait_ms.toFixed(1)}ms | Decode: ${t.predicted_per_second.toFixed(2)} tok/s`);
  return {
    id,
    wallClock,
    completionTokens: json.usage.completion_tokens,
    promptTokens: json.usage.prompt_tokens,
    timings: t
  };
}

async function runConcurrency(n, maxTokens = 50) {
  console.log(`\n================ RUNNING ${n} CONCURRENT STREAMS ================`);
  const prompts = [
    "Write a concise 3-bullet point summary of quantum computing principles.",
    "Write a concise 3-bullet point summary of artificial neural network architectures.",
    "Write a concise 3-bullet point summary of distributed ledger consensus mechanisms.",
    "Write a concise 3-bullet point summary of CRISPR Cas9 genetic engineering techniques."
  ];

  const tStart = performance.now();
  const tasks = [];
  for (let i = 0; i < n; i++) {
    tasks.push(sendRequest(i + 1, prompts[i % prompts.length], maxTokens));
  }

  const results = await Promise.all(tasks);
  const totalWallSec = (performance.now() - tStart) / 1000;

  const valid = results.filter(Boolean);
  const totalTokens = valid.reduce((acc, r) => acc + r.completionTokens, 0);
  const avgQueueMs = valid.reduce((acc, r) => acc + r.timings.queue_wait_ms, 0) / valid.length;
  const avgDecodeSpeed = valid.reduce((acc, r) => acc + r.timings.predicted_per_second, 0) / valid.length;
  const aggregateTokPerSec = totalTokens / totalWallSec;

  console.log(`\n---------------- METRICS FOR ${n} CONCURRENT STREAMS ----------------`);
  console.log(`Streams Executed:          ${valid.length} / ${n}`);
  console.log(`Total Generated Tokens:    ${totalTokens} tokens`);
  console.log(`Total Elapsed Wall Time:   ${totalWallSec.toFixed(2)} seconds`);
  console.log(`Average Queue Wait Time:   ${avgQueueMs.toFixed(1)} ms`);
  console.log(`Average Stream Speed:      ${avgDecodeSpeed.toFixed(2)} tok/s per stream`);
  console.log(`AGGREGATE CLUSTER TOK/SEC: ${aggregateTokPerSec.toFixed(2)} tok/s`);
  console.log(`-------------------------------------------------------------------\n`);
  return { n, totalTokens, totalWallSec, avgQueueMs, avgDecodeSpeed, aggregateTokPerSec };
}

async function main() {
  console.log("Starting Concurrency Evaluation on 3-Node Cluster...");
  console.log("Model: " + MODEL);
  console.log("Target: " + BASE_URL);

  // Test 1: 2 Concurrent Streams
  const r2 = await runConcurrency(2, 50);

  // Test 2: 4 Concurrent Streams (Full saturation of 4 parallel slots)
  const r4 = await runConcurrency(4, 50);

  console.log("\n================ SUMMARY COMPARISON ================");
  console.log(`1 Stream (Baseline):  ~5.71 tok/s aggregate`);
  console.log(`2 Streams Concurrent: ${r2.aggregateTokPerSec.toFixed(2)} tok/s aggregate (${r2.totalWallSec.toFixed(2)}s wall time)`);
  console.log(`4 Streams Concurrent: ${r4.aggregateTokPerSec.toFixed(2)} tok/s aggregate (${r4.totalWallSec.toFixed(2)}s wall time)`);
  console.log("====================================================\n");
}

main();
