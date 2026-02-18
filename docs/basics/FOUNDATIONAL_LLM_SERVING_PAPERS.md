# A Plain-English Guide to Six Foundational LLM Serving Papers

These six papers form the backbone of how modern large language models are deployed and served at scale. Together they solve a chain of interconnected problems — from how attention is computed, to how memory is managed, to how requests are scheduled, to how we can squeeze out more speed at inference time. Reading them in this guide will give you a clear mental model of the engineering behind every major LLM API you use today.

---

## Table of Contents

1. [FlashAttention (2022)](#1-flashattention-2022)
2. [Orca: Continuous Batching (2022)](#2-orca-a-distributed-serving-system-2022)
3. [vLLM: PagedAttention (2023)](#3-vllm-pagedattention-2023)
4. [Splitwise: Phase Splitting (2023)](#4-splitwise-efficient-generative-llm-inference-using-phase-splitting-2023)
5. [Sarathi-Serve: Chunked Prefill (2024)](#5-sarathi-serve-chunked-prefill-for-llm-serving-2024)
6. [SpecInfer: Speculative Inference (2023)](#6-specinfer-speculative-inference-for-llms-2023)

---

## Background: How LLMs Generate Text

Before diving into the papers, you need a mental model of what actually happens when an LLM generates a response.

When you send a prompt to an LLM, the process splits into two distinct phases:

**Prefill** — The model reads your entire input prompt all at once. It processes every token in parallel and builds up a structure called the KV Cache (explained below). This phase is compute-heavy and produces the very first output token.

**Decode** — After the first token is produced, the model enters a loop. At each step it reads the previous token and generates the next one — one at a time, sequentially. This continues until the model outputs a stop token. This phase is memory-heavy and produces every token after the first.

The time it takes to get your first token back is called **TTFT** (Time to First Token). The time between each subsequent token is called **TPOT** or **TBT** (Time Per Output Token / Time Between Tokens). Both matter to users. The papers below are all in some way trying to make one or both of these faster, while also serving as many users simultaneously as possible.

### What is the KV Cache?

Attention — the core mechanism in transformers — works by having every token "look at" all previous tokens to decide what to pay attention to. It does this using three vectors per token: a Query (Q), a Key (K), and a Value (V).

During decoding, when generating token number 500, the model needs to compare that token's Query against the Keys of all 499 previous tokens. If you didn't cache those Keys and Values, you would have to recompute them from scratch every single step — a massive waste. The KV Cache stores the K and V vectors of all previously processed tokens so they can be reused.

The problem: the KV cache is enormous and grows with every token. For a large model serving many users simultaneously, KV cache management becomes the central bottleneck. Four of these six papers are fundamentally about that problem.

---

## 1. FlashAttention (2022)

**Paper:** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

### The Problem It Solves

Standard attention has a dirty secret: even though the actual math isn't that expensive, the way it was implemented on GPUs was extremely wasteful in how it moved data around. FlashAttention fixes that.

To understand why, you need to know a bit about GPU memory. A GPU has two main memory areas:

- **HBM (High Bandwidth Memory):** This is the big pool of GPU RAM you hear about (e.g. "40GB A100"). It's large but relatively slow.
- **SRAM (on-chip memory):** This is tiny (a few hundred KB), but it's 10–100x faster than HBM. It sits right next to the compute cores.

Every GPU operation follows the same pattern: load data from HBM → compute → write result back to HBM. The bottleneck isn't always the computation — it's often the back-and-forth trips to slow HBM.

### What Standard Attention Was Doing Wrong

The standard attention computation involves several steps: multiply Q and K, apply a softmax, multiply the result by V. In the original PyTorch implementation, each of those steps was a separate operation. That meant:

1. Load Q and K from HBM, compute the Q×K matrix, **write it back to HBM**
2. Load that matrix from HBM again, compute the softmax, **write it back to HBM**
3. Load it again, multiply by V, write the final result back

For a long sequence of N tokens, the Q×K matrix has N×N entries. This matrix gets read and written multiple times. For long sequences, this becomes a serious bottleneck — and the memory to store it grows quadratically with sequence length.

### What FlashAttention Does

FlashAttention's core insight is deceptively simple: **never write the big N×N attention matrix to HBM at all.** Instead, do all the math in small tiles that fit entirely inside the fast on-chip SRAM, and only write the final result back to HBM.

Concretely, FlashAttention:

1. Splits the Q, K, and V matrices into small blocks that fit in SRAM
2. Loads one block at a time, computes partial attention results, and combines them correctly using a running normalization trick
3. Produces the exact same result as standard attention — no approximation — but without ever materializing the full N×N matrix

This technique is called **tiling**, and it's well-known in high-performance computing. FlashAttention's contribution was applying it cleverly to the attention mechanism, where the softmax operation makes tiling mathematically tricky (you need to know the sum of the entire row to normalize, but you're computing in blocks).

The solution is to maintain running statistics — tracking the maximum value and running sum needed for softmax as you process each block — so you can compute the correct normalized result block-by-block without seeing the whole row at once.

### Why This Matters

- **Up to 7.6x faster** than standard PyTorch attention for long sequences
- **Memory usage drops from quadratic to linear** in sequence length — you no longer need to store the N×N matrix
- **Enables much longer contexts** because the memory bottleneck is lifted. The first transformers to handle 64K token sequences used FlashAttention
- **No accuracy loss** — it produces mathematically identical outputs to standard attention

FlashAttention is now the de facto standard for attention in virtually every serious LLM inference system. When a serving system claims to use "fast kernels," FlashAttention (or its successors FlashAttention-2 and FlashAttention-3) is usually what they mean.

---

## 2. Orca: A Distributed Serving System (2022)

**Paper:** *Orca: A Distributed Serving System for Transformer-Based Generative Models*
**Authors:** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, et al.

### The Problem It Solves

Before Orca, LLM serving systems processed requests in **static batches**: you'd wait until you had a group of requests, send them all through the model together, and wait until every single one finished before starting the next batch.

This sounds reasonable but has a catastrophic flaw: requests have wildly different output lengths. If you batch 8 requests and 7 of them finish generating after 50 tokens but the 8th keeps going to 500 tokens, those 7 finished requests sit completely idle while you wait for the 8th. The GPU is doing nothing useful for 7 out of 8 slots — huge waste.

Imagine a restaurant that seats a table of 8 and doesn't let anyone leave (or new guests sit down) until the entire table is done eating. That's static batching.

### The Core Innovation: Iteration-Level Scheduling (Continuous Batching)

Orca changed the fundamental scheduling unit from **request** to **iteration**. An iteration is one step — generating a single token. Instead of committing to a fixed set of requests for the duration of their generation, Orca makes a fresh scheduling decision at every single token step.

Here's what that means in practice:

- After each token generation step, the scheduler checks: did any request just finish? If so, remove it from the batch and immediately add a waiting request in its place
- New requests join mid-batch as soon as a slot opens up
- The batch is continuously replenished — hence the term "continuous batching"

Back to the restaurant analogy: now each diner can leave as soon as they're done eating, and new guests can immediately take their seat. The tables are never empty.

### The Selective Batching Trick

There's a technical complication: during the decode phase, all sequences in the batch have different lengths — one request might be at token 47, another at token 183. Standard GPU batch operations require all inputs to be the same shape.

Orca's solution is **selective batching**: for operations that don't depend on sequence length (like linear projections and normalization layers), batch everything together. For the attention operation — which does depend on each request's specific history — process each request's attention separately, then recombine. This preserves the efficiency gains of batching where it matters most while correctly handling the variable-length nature of decoding.

### Why This Matters

Continuous batching was a landmark improvement in LLM serving. It became the foundation that every subsequent serving system was built on. By keeping the GPU continuously busy rather than waiting for stragglers:

- GPU utilization goes from typically 20-40% to much higher
- Latency for waiting requests improves dramatically (they don't have to wait for a full batch to complete)
- Throughput increases significantly — the paper reports meaningful improvements over prior systems

Every major LLM serving system today (vLLM, TGI, TensorRT-LLM, SGLang) uses continuous batching as a baseline. Orca invented it.

---

## 3. vLLM: PagedAttention (2023)

**Paper:** *Efficient Memory Management for Large Language Model Serving with PagedAttention*
**Authors:** Woosuk Kwon, Zhuohan Li, et al. (UC Berkeley)

### The Problem It Solves

After Orca's continuous batching, serving systems could keep GPUs busier. But this created a new bottleneck: memory. With more requests running simultaneously, more KV caches need to live in GPU memory at once, and the KV cache for a long conversation can be enormous.

Previous systems allocated KV cache memory the naive way: reserve one giant contiguous block for each request upfront, sized for the *maximum possible* output length. This leads to three kinds of waste:

- **Internal fragmentation:** You reserve space for 2048 tokens but the request only uses 200. The other 1848 slots are reserved but empty, wasting memory
- **External fragmentation:** After many requests finish and free their memory, the GPU's memory looks like Swiss cheese — lots of free space, but in scattered pieces too small to fit a new large request
- **No sharing:** Multiple requests with the same prefix (e.g. the same system prompt) each store their own copy of the KV cache for that prefix, even though it's identical

The result: studies found that 60-80% of allocated KV cache memory was being wasted. This directly limited how many requests could run simultaneously.

### The Core Innovation: Paging (Like an Operating System)

The vLLM team looked at this problem and realized it was identical to a problem operating systems solved decades ago: memory fragmentation. The OS solution was **virtual memory with paging** — instead of allocating one big contiguous block for each process, you split memory into small fixed-size pages and assign them as needed, tracking which physical pages correspond to which logical addresses.

vLLM applied exactly this idea to the KV cache. They introduced **PagedAttention**:

- The KV cache for each request is split into small **blocks** (typically 16 tokens each)
- These blocks can be stored anywhere in GPU memory — they don't need to be contiguous
- A **block table** for each request maps "logical block 1 → physical location 47, logical block 2 → physical location 12..." etc.
- Blocks are allocated on demand as the request generates more tokens, and freed immediately when the request completes

This eliminates external fragmentation entirely (any freed block can be reused by any request) and reduces internal fragmentation to at most one partially-filled block per request (at most 15 wasted token slots for a block size of 16, instead of potentially thousands).

### KV Cache Sharing

Because blocks are tracked by a table rather than embedded in a contiguous layout, vLLM can share blocks across requests. If two requests start with the same system prompt, their KV cache blocks for that prompt can point to the same physical memory — no duplication. This is called **prefix caching** and it's now a standard feature. When you send the same system prompt with every API call, providers like Anthropic and OpenAI can reuse the computed KV cache for that prefix, making your calls faster and cheaper.

### Why This Matters

- **Near-zero KV cache waste** (under 4% vs. 60-80% before)
- **2-4x higher throughput** than systems using Orca-style continuous batching but naive memory management
- **Enables much larger batch sizes** because memory is used efficiently
- **Prefix caching** — the foundation of "prompt caching" features you see in commercial LLM APIs today
- **Graceful handling of memory pressure** — when memory runs low, vLLM can evict and recompute KV blocks rather than crashing

vLLM became one of the most widely deployed open-source LLM inference engines, and PagedAttention is now used or adapted by virtually every major system.

---

## 4. Splitwise: Efficient Generative LLM Inference Using Phase Splitting (2023)

**Paper:** *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*
**Authors:** Pratyush Patel, Esha Choukse, et al. (Microsoft Research)

### The Problem It Solves

Recall that LLM generation has two phases: prefill and decode. These two phases are fundamentally different kinds of computation:

- **Prefill** processes all input tokens in parallel. It does a lot of computation at once. GPU utilization is high. It's **compute-bound** (the bottleneck is raw compute speed)
- **Decode** processes one token at a time. Each step does very little work but has to load the entire model and KV cache from memory. It's **memory-bound** (the bottleneck is memory bandwidth, not compute)

When prefill and decode run on the same GPU, they interfere with each other. A long prefill for a new request blocks ongoing decode steps for other requests, causing latency spikes. A short decode step that gets mixed with a long prefill wastes compute because decode underutilizes the GPU's compute capacity.

Moreover, the optimal way to configure a multi-GPU deployment (how many GPUs in parallel, what parallelism strategy to use) is different for prefill vs. decode. But if they run together, you're forced to make a single compromise configuration that's suboptimal for both.

### The Core Innovation: Disaggregation

Splitwise's answer is radical separation: **run prefill and decode on completely different GPUs** (or groups of GPUs).

Here's how a request flows through a Splitwise deployment:

1. A new request arrives and gets sent to a **prefill worker** — a GPU dedicated to processing input prompts
2. The prefill worker processes the entire prompt, generates the first output token, and produces the KV cache
3. The KV cache is transferred over the network (via high-speed interconnects like InfiniBand or NVLink) to a **decode worker**
4. The decode worker takes over and generates all subsequent tokens using its own memory

Since prefill workers only handle compute-intensive work and decode workers only handle memory-bound work, each can be optimized independently:

- Prefill workers benefit from compute-focused parallelism (tensor parallelism)
- Decode workers benefit from memory-focused parallelism (pipeline parallelism, larger batch sizes)
- You can even use different GPU types — more powerful H100s for prefill where compute matters, cheaper A100s for decode

### The SLO Angle

A key concept in Splitwise is **SLOs (Service Level Objectives)** — contractual guarantees about latency. TTFT and TPOT have different requirements for different applications:

- A coding assistant needs fast TPOT (tokens appearing quickly so you can see the code stream in real time)
- A document summarization tool might tolerate slow TPOT but needs low TTFT (users want to see something quickly)
- A chatbot needs both to be reasonable

With colocation (same GPU for both phases), optimizing for one metric hurts the other. With disaggregation, you can scale prefill capacity and decode capacity independently to hit both SLO targets simultaneously.

### Why This Matters

- Eliminates interference between prefill and decode, improving both TTFT and TPOT
- Enables **independent scaling** of each phase — spin up more prefill workers when you have long prompts, more decode workers when you have long outputs
- Allows **phase-specific hardware choices** for cost efficiency
- Introduced the concept of **goodput** — not just raw throughput, but the rate of requests that actually satisfy latency SLOs — as the right metric for evaluating LLM serving systems

Prefill-decode disaggregation went from a research idea in 2023 to the dominant architecture in production LLM serving by 2025. Nearly every major inference framework (vLLM, SGLang, TensorRT-LLM) now supports it.

---

## 5. Sarathi-Serve: Chunked Prefill for LLM Serving (2024)

**Paper:** *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
**Authors:** Amey Agrawal, Nitin Kedia, et al.

### The Problem It Solves

Here's a painful scenario in LLM serving with continuous batching: you have 20 decode requests happily generating tokens at a good rate, with low latency. Then a new request arrives with a very long prompt — say, 8000 tokens. Under naive continuous batching, that new request gets scheduled for prefill immediately, and that prefill takes a long time to complete.

During that entire time, your 20 ongoing decode requests are **completely blocked**. They can't generate a single token until the long prefill finishes. Users experience a sudden stall — they were getting tokens every 50ms, and suddenly there's a multi-second freeze. This is called a **generation stall**, and it causes nasty latency spikes in TBT.

Meanwhile, the inverse problem exists too: if you prioritize decodes and delay long prefills, users with long prompts wait a long time for their first token (high TTFT).

There seems to be a fundamental tradeoff: good TPOT OR good TTFT, but not both at the same time.

### The Core Innovation: Chunked Prefill + Stall-Free Batching

Sarathi-Serve's insight is that a long prefill doesn't have to be processed all at once. A prompt of 8000 tokens can be split into chunks — say, 512 tokens each — and processed across multiple iterations, interleaved with decode steps.

**Chunked prefill** works like this: instead of spending one entire iteration processing a 8000-token prompt, you process 512 tokens of it, then let decodes run, then process the next 512 tokens of it, and so on. The prefill completes in 16 steps spread across 16 iterations instead of dominating one.

**Stall-free batching** is the scheduling policy built on top of this:

1. Each iteration, first fill the batch with all ongoing decode requests
2. If there's remaining "budget" (compute capacity), add a chunk of any in-progress prefill
3. Only add brand new requests if there's still budget left

This means decodes are always prioritized — they never get blocked. Prefills make progress incrementally, sharing iterations with decodes rather than monopolizing them.

A key parameter is the **token budget** per iteration — the maximum number of tokens processed in one step. It's chosen based on your TBT target: smaller budgets mean lower latency, but too small and you spend excessive time re-loading KV caches for the same prefill across many chunks.

### The Tradeoff

Chunked prefill has a small cost: because you're loading the partial KV cache of each chunk repeatedly across multiple iterations, there's some extra memory bandwidth usage. The paper shows this overhead is small in practice.

There's also a slight TTFT increase for new requests because their prefill is spread across multiple iterations. But this is usually acceptable — the TPOT improvement for ongoing users more than compensates.

### Why This Matters

- Eliminates the generation stalls that caused latency spikes under continuous batching with mixed-length workloads
- Achieves high throughput AND low tail latency simultaneously — something that was previously a tradeoff
- Works well for single-GPU or small-cluster deployments where disaggregation (Splitwise) might be too complex
- Became a standard technique adopted by vLLM, SGLang, and most other serving frameworks
- Especially important for pipeline-parallel deployments (multiple GPUs in a chain), where the uniform compute per iteration reduces "pipeline bubbles" — idle time caused by uneven workload distribution

Sarathi-Serve sits between Orca and Splitwise in the conceptual hierarchy: Orca gave us continuous batching, Splitwise gave us full disaggregation, and Sarathi-Serve gave us a pragmatic middle ground that achieves most of the benefits without the complexity of disaggregation.

---

## 6. SpecInfer: Speculative Inference for LLMs (2023)

**Paper:** *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification*
**Authors:** Xupeng Miao, Gabriele Oliaro, et al.

### The Problem It Solves

Every technique so far has attacked the problem from the serving infrastructure side — how to schedule requests, manage memory, or allocate GPU resources. SpecInfer attacks a different, more fundamental bottleneck: the sequential nature of token generation itself.

Standard LLM decoding is inherently sequential. To generate token 47, you must first have token 46. To have token 46, you need token 45. Each token requires one full forward pass through the entire model. Even if your GPU is doing that efficiently, you can't parallelize across tokens — it's a hard dependency chain.

This means your big, expensive model runs one step at a time, one token at a time. For a 70-billion-parameter model, each step takes a significant fraction of a second. The GPU sits partially idle between steps because a single token-step doesn't use all the GPU's compute capacity.

### The Core Innovation: Draft-Then-Verify Speculation

Speculative decoding breaks the sequential bottleneck by introducing a two-model system:

1. **The draft model** — a small, fast model (maybe 1-7B parameters, 10-20x smaller than the main model). It rapidly generates several "speculative" tokens at once
2. **The target model** — your actual large model. It verifies all the draft tokens in a single parallel forward pass

Here's the key insight: verifying N tokens in parallel takes barely more time than generating 1 token, because the verification computation across multiple tokens can be parallelized.

The process works like this:

1. The draft model generates, say, 5 candidate tokens rapidly
2. The large model takes those 5 tokens as input and computes what it *would* have generated at each position — all in one parallel pass
3. You compare: if the large model agrees with draft token 1, accept it. Check token 2. And so on.
4. Accept the longest correct prefix. If the large model disagrees at token 3, accept tokens 1 and 2, discard 3, 4, and 5, and resample token 3 from the large model
5. Repeat

The crucial mathematical guarantee: this process produces exactly the same distribution of outputs as just running the large model normally. There's no quality loss — it's a lossless speedup.

### Why This Works

The speedup comes from the observation that many tokens are "easy" — the next word in "The quick brown fox jumps over the..." is not going to surprise a large model. Highly predictable tokens can be guessed correctly by a small model almost every time. Instead of spending one expensive large-model step per obvious token, you generate several obvious tokens in one draft pass, then verify them all at once.

In typical conversational text, the draft model gets 3-5 consecutive tokens right before the large model disagrees. Each round of draft-then-verify produces around 3-4 tokens instead of 1, yielding roughly a 2-3x speedup in wall-clock time.

### The Acceptance Rate Problem

The speedup depends heavily on how often the draft model's predictions are accepted. For predictable content (boilerplate text, structured outputs, code with obvious next tokens), acceptance rates are very high and speedups approach 3x. For creative or unpredictable outputs, the draft model gets rejected more often and the speedup diminishes.

This is why choosing the right draft model matters. A draft model that's a smaller version of the target model (same architecture, less parameters) typically achieves high acceptance rates. Models from a different family may not align well.

More recent variants like **EAGLE** sidestep the separate-model problem entirely: instead of a separate draft model, a tiny prediction head is attached directly to the large model's internal layers. It reuses the large model's own representations to make predictions, achieving higher acceptance rates with minimal extra memory.

### Why This Matters

- **2-3x faster token generation** without any change to output quality — users get responses faster
- Particularly effective for **latency-sensitive applications** like chatbots and code completion, where reducing inter-token latency matters most
- **No accuracy tradeoff** — the mathematical guarantee of losslessness is fundamental to the technique
- Works well at low batch sizes (single-user or lightly-loaded systems) where the GPU has spare capacity the draft model can use
- Now natively supported in vLLM, TensorRT-LLM, and most major inference frameworks

The main limitation: speculative decoding is most effective at low batch sizes. When serving many users simultaneously at high batch sizes, the GPU is already fully utilized during the target model's verification pass, and the overhead of running the draft model becomes a cost rather than a benefit.

---

## How These Papers Fit Together

These six papers are best understood as a progression of increasingly targeted solutions:

**FlashAttention** solves the low-level kernel problem — attention itself was being computed wastefully. Every other paper depends on FlashAttention or its successors being in place.

**Orca** introduced the scheduling revolution — continuous batching keeps GPUs busy rather than waiting for slow requests to finish. This unlocked the next level of problems.

**vLLM** solved the memory crisis that Orca's continuous batching created — now that you have more requests running simultaneously, you need to manage their KV caches efficiently. PagedAttention is the solution.

**Splitwise** recognized that the two phases of generation are fundamentally different workloads and should be served by different resources. This is the architectural insight that powers production-scale serving today.

**Sarathi-Serve** addressed the day-to-day latency spikes that occur when long prompts and short-prompt users share the same serving system. Chunked prefill is the practical fix that works without requiring full disaggregation infrastructure.

**SpecInfer** attacks a completely different angle — instead of optimizing how we schedule and manage resources, it makes the generation process itself faster by generating multiple tokens per large-model pass.

A modern state-of-the-art LLM serving system uses all of these ideas simultaneously: FlashAttention kernels for efficient attention computation, continuous batching for GPU utilization, PagedAttention for KV cache management, chunked prefill to avoid stalls, prefill-decode disaggregation for large-scale deployments, and speculative decoding for latency-sensitive workloads.

---

## Quick Reference Glossary

| Term | Meaning |
|------|---------|
| **KV Cache** | Stored key-value vectors from previous tokens, avoiding recomputation |
| **Prefill** | Processing the input prompt in parallel; produces the first token |
| **Decode** | Generating output tokens one at a time |
| **TTFT** | Time to First Token — latency until the first output token appears |
| **TPOT/TBT** | Time Per Output Token / Time Between Tokens — inter-token latency |
| **HBM** | High Bandwidth Memory — the main GPU RAM, large but relatively slow |
| **SRAM** | On-chip memory — tiny but very fast, sits near compute cores |
| **Continuous batching** | Adding/removing requests from the batch at every token step |
| **Paging** | Splitting KV cache into small fixed-size blocks that can go anywhere in memory |
| **Disaggregation** | Running prefill and decode on separate hardware |
| **Chunked prefill** | Splitting a long prompt's prefill across multiple iterations |
| **Speculative decoding** | Using a small draft model to propose tokens that the large model verifies |
| **SLO** | Service Level Objective — a latency guarantee (e.g. P99 TTFT < 500ms) |
| **Goodput** | Rate of requests that satisfy their SLO targets (more meaningful than raw throughput) |
| **Draft model** | Small fast model used to speculate tokens in speculative decoding |
| **Acceptance rate** | Fraction of speculative tokens the large model agrees with |
| **Tiling** | Processing data in small chunks that fit in fast memory |
| **Kernel fusion** | Combining multiple GPU operations into one to reduce memory round-trips |