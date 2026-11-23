# MeshLLM: Findings, Architectural Analysis & Discussion Queries

**Date:** 2026-11-23  
**Cluster Architecture:** Heterogeneous Distributed Mesh (Windows 11 RTX 3060 12GB + macOS M-series 16GB + Arch Linux RTX 3060 6GB)  
**Evaluated Model:** `unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL` (27B Parameters, Q4_K_XL, ~17.6 GB)

---

## 1. VRAM Allocation Mechanics & Headroom Analysis

### A. Is VRAM Reservation Fixed or Dynamic?
In the `mesh-llm` native runtime (`llama.cpp` + Skippy ABI):

1. **Static KV Cache Pre-Allocation**:
   - The KV cache for the specified context length (`--ctx-size 8192`) is **allocated upfront in VRAM at initialization time**.
   - As observed in the startup logs:
     ```text
     kv_runtime_state: kv_cache_initialization_started
     kv_runtime_state: kv_cache_initialization_completed numeric_summary=native_numeric_summary_0=8192
     ```
   - **Conclusion**: VRAM usage **does NOT expand dynamically** as context grows from 100 to 3,300+ tokens. The memory for all 8,192 context positions is already reserved.
   - When processing requests of varying lengths (e.g. 465 tokens vs 3,297 tokens), the runtime merely writes into pre-allocated memory slots in the existing buffer.

2. **Ephemeral Scratch Buffers**:
   - The only dynamic memory components during active inference are the intermediate tensor activation buffers (scratchpads for matrix multiplications during forward passes).
   - These ephemeral buffers are lightweight (~50–150 MB) and are freed immediately after each layer computation.

---

### B. Live `nvidia-smi` Headroom Breakdown (Windows Coordinator)

| Component | VRAM Footprint | Description |
|---|---|---|
| **Total Physical VRAM** | `12,288 MiB` (12.0 GB) | Hardware capacity of RTX 3060 |
| **Model Weights (Layers 0–36)** | `~8,200 MiB` | Static GGUF layer weights & embeddings |
| **Pre-allocated KV Cache (8k)** | `~1,600 MiB` | Upfront allocation for 8,192 tokens |
| **OS & Desktop Processes** | `~933 MiB` | Windows Terminal, Chrome, Antigravity IDE, DWM |
| **Total Used Memory** | **`10,733 MiB`** (~10.48 GB) | Reported by `nvidia-smi` |
| **Remaining Free Headroom** | **`1,555 MiB` (1.52 GB)** | Safe buffer against OOM crashes |

> [!NOTE]
> Because the full 8,192-token KV cache is already accounted for within the `10,733 MiB` footprint, the **1.52 GB free headroom remains completely stable** even during long-context prompts up to 8k tokens.

---

## 2. Key Empirical Findings

### 1. Prefill Throughput vs. Decode Speed
- **Prompt Prefill Speed**: Ingested context at **111.5 to 135.7 tokens/second**. Because prefill processes all prompt tokens in parallel tensor batches, GPU compute utilization is high.
- **Decode Autoregressive Speed**: Generation stabilized at **3.61 to 5.75 tokens/second**. In pipeline parallelism, decode throughput is constrained by sequential layer execution and the round-trip latency of passing activation tensors between stages over QUIC.

### 2. Network Efficiency (Skippy Contiguous Splits vs Tensor Parallelism)
- **Tensor Parallelism (TP)**: Requires all-reduce synchronization across every single attention head and MLP layer (hundreds of round-trips per token), making it unviable over consumer Wi-Fi/LAN.
- **Skippy Contiguous Stage Splits (PP)**: Passes activation tensors **only once per token** at stage boundaries (e.g., between Layer 36 on Windows and Layer 37 on Mac).
- **Network Load**:
  - Decode: Only **~10.7 KB per token** transmitted across the network.
  - 100-token generation: Total network payload was only **1,019.1 KB** (~1.0 MB).
  - LAN QUIC RTT remained between **7 ms and 17 ms**, contributing virtually zero perceived delay.

---

## 3. Findings & Discussion Queries

### Query 1: What happens if an incoming prompt exceeds the 8,192 context boundary?
* **Behavior**: If a prompt exceeds the configured context window (e.g. 9,000 tokens), the local OpenAI API returns a `400 Bad Request: context length exceeded` error or truncates earlier tokens depending on the client application settings.
* **Why not default to 256k?**: The native architecture of Qwen 3.6 supports up to 262,144 tokens. However, evaluating at native context requires ~37.8 GB of VRAM solely for the KV cache. Pining context to `8192` strikes the optimal balance between real-world agent task capacity and consumer GPU memory limits.

### Query 2: How will the Coordinator incorporate Omarchy (3060 6GB)?
* **Current State**: Windows (Layers 0–36) and Mac (Layers 36–64) are handling the entire 64-layer pipeline.
* **Potential 3-Way Partitioning**:
  - Node 1 (Windows 12GB): Layers 0–24 (~6.5 GB) + Embeddings
  - Node 2 (Mac 16GB): Layers 24–48 (~6.5 GB)
  - Node 3 (Omarchy 6GB): Layers 48–64 (~4.3 GB) + Output Norm/Logits
* **Trade-off Analysis**:
  - *Advantage*: Reduces VRAM usage on both Windows and Mac, freeing up VRAM for longer context or concurrent requests.
  - *Disadvantage*: Introduces a second network hop (Windows $\to$ Mac $\to$ Omarchy $\to$ Windows), which will slightly lower decode tokens/sec due to additional RTT latency.

### Query 3: Static Pre-allocation vs. PagedAttention (vLLM style)
* **Current Limitation**: `llama.cpp` statically reserves the entire 8k KV cache buffer at startup, regardless of whether a query uses 50 tokens or 5,000 tokens.
* **Future Optimization**: Implementing PagedAttention or chunked dynamic KV blocks would allow nodes to dynamically allocate memory pages on demand, potentially enabling larger nominal context windows without upfront VRAM reservation.

### Query 4: Streaming Responsiveness for Coding Agents
* **TTFT of 40.6 ms**: Shows that time-to-first-token is negligible on LAN clusters.
* **Agent Suitability**: Tools like Claude Code, Goose, and OpenCode require low latency and high consistency. At ~5.8 tokens/sec, distributed inference on 27B dense models provides a viable, zero-cloud-cost alternative to API providers for coding and reasoning tasks.

---

## 4. Concurrency Evaluation & Multi-Stream Mechanics

### A. The Concurrency Test (Dispatched Simultaneous Streams)
When testing concurrent request bursts against the 3-node cluster configured with `--ctx-size 16384`:
* **Dispatched**: 2 simultaneous requests, then 4 simultaneous requests.
* **Error Encountered**:
  ```json
  HTTP 502: {"error":{"message":"persistent downstream lane did not become ready","type":"server_error","code":"service_unavailable"}}
  ```
* **Underlying Cause**:
  - In `mesh-llm`, every concurrent stream requires its own independent KV cache buffer and dedicated QUIC multiplexed stream (`lane_count`).
  - At **16k context**, a single lane requires **~5.1 GB of KV cache** across the 3 nodes. Allocating 2 or 4 lanes would require 10–20 GB of KV cache, exceeding the total 34 GB cluster limit.
  - To prevent OOM crashes, the planner constrained the pipeline to **`lane_count: 1`**.
  - When two requests arrived at the exact same millisecond, Request 2 attempted to acquire a non-existent second downstream lane, causing a socket buffer overflow (`failed to fill whole buffer`).

### B. Concurrency Strategy Recommendations
1. **FIFO Request Queuing (For 16k Context)**:
   In realistic team use, requests rarely arrive at the identical millisecond. A lightweight reverse proxy with `max_concurrency = 1` queues incoming prompts cleanly.
2. **True Parallel Multiplexing (`lane_count: 4`)**:
   Lowering context to `--ctx-size 4096` reduces KV cache footprint to ~1.2 GB per stream, allowing the planner to allocate **4 simultaneous execution lanes** without queuing.

---

## 5. Node Churn & Resilience: What Happens if Mac Drops?

### A. Immediate Pipeline Invalidation
Because Mac hosts **Layers 31–63**, dropping the Mac node physically severs the sequential execution pipeline. In-flight queries fail with a transport disconnect (`peer disconnected`).

### B. Auto-Rebalance Attempt & Physical Capacity Shortfall
The coordinator's gossip protocol immediately attempts to re-plan the 64 layers across the remaining alive nodes (Windows + Omarchy):
* **Windows Usable VRAM**: 11.6 GB
* **Omarchy Usable VRAM**: 4.0 GB
* **Combined VRAM**: **15.6 GB**
* **Qwen 27B Weights (Q4_K_XL)**: **17.6 GB**

Because $15.6\text{ GB} < 17.6\text{ GB}$, the raw model weights cannot fit into physical memory even at zero context. The coordinator logs `split_capacity_shortfall: no topology can distribute all layers` and enters a waiting standby state.

### C. Automatic Self-Healing on Reconnect
`mesh-llm` does not require restarting the coordinator. The moment Mac turns back on and rejoins:
1. The coordinator detects the restored ~34 GB pool.
2. It automatically re-plans the 3-stage partition.
3. Downstream stages reload and signal readiness.
4. The cluster returns to `Serving` status on its own.

---

## 6. Upcoming Experiment: NVIDIA-Only Architecture (Windows + Omarchy)

### Hypothesis: Does a Pure CUDA Pipeline Beat Heterogeneous CUDA + Metal?
* **Concept**: Does eliminating Apple Silicon (Metal/Accelerate) in favor of a homogeneous NVIDIA-only pipeline (Windows CUDA + Linux CUDA) yield higher decode throughput?
* **Factors to Measure**:
  1. **Kernel Homogeneity**: Uniform cuBLAS matrix multiplication kernels vs. bridging CUDA activation frames to Apple Metal.
  2. **Memory Bandwidth Differences**: Apple Silicon unified memory (M-series ~150–200 GB/s) vs. Desktop RTX 3060 (340 GB/s) vs. Laptop RTX 3060 (325 GB/s).
  3. **Network Latency**: Direct Windows-to-Linux LAN latency (~2–7 ms).

### Important Sizing Note for the NVIDIA-Only Test
Because Windows + Omarchy provide **15.6 GB usable VRAM**:
* **27B Model**: Cannot fit on Windows + Omarchy alone (requires 17.6 GB for weights).
* **Recommended Models for Pure NVIDIA Test**:
  - **`Qwen2.5-Coder-14B-Instruct-GGUF` (~8.5 GB weights)**: Leaves **~7 GB of VRAM** for a massive 16k context window and multiple concurrent lanes across the two NVIDIA GPUs.
  - **`DeepSeek-R1-Distill-Qwen-14B-GGUF` (~9.0 GB weights)**: Ideal for testing pure CUDA reasoning throughput.
