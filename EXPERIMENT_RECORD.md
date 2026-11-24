# Experiment 2 record

**KV cache offloading for long-context inference on a 4 GB GPU**: the trade-off between generation speed (tok/s) and
maximum context length when the KV cache is offloaded to CPU RAM, versus a truncated (compressed) cache kept
strictly in GPU VRAM.

Written from the log files in this folder, not from memory. Paths are relative to `~/kv-offload/`.
Every check has a script in `scripts/checks/` and a log in `logs/setup/`. Times are the log files' modification times.
All checks ran on 2026-11-19. Sections are added one at a time.

---

## 0. What is being compared

| Method | Where the KV cache lives | How much of it is kept | Implementation |
|---|---|---|---|
| GPU-only (baseline) | GPU VRAM | all tokens | transformers `DynamicCache` |
| CPU offloading | CPU RAM (pinned), copied to the GPU layer by layer | all tokens | own `PinnedOffloadCache` (`scripts/checks/kvoffload.py`), see check 11 for why |
| StreamingLLM | GPU VRAM | first 4 "sink" tokens + most recent tokens, up to a budget | kvpress `StreamingLLMPress` |
| SnapKV | GPU VRAM | tokens the model attended to most, up to a budget | kvpress `SnapKVPress` (window 64, kernel 5) |

Considered and dropped:
- **H2O**: needs the full attention-weight matrix, which the memory-efficient attention kernel used here never builds.
- **InfiniGen**: research code, not runnable on this setup.

Planned dataset: **RULER** (synthetic tasks with exact-match answers; any length can be generated).
Planned metrics: decode tok/s, prefill time (time to first token), maximum context length, peak VRAM, peak RAM,
RULER accuracy.

---

## 1. Hardware and software

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1650 (laptop), 4096 MiB, compute capability 7.5 (sm_75, no Tensor Cores) | `c1_system.log`, `c2_versions_gpu.log` |
| Driver | 581.83 (Windows), CUDA 13.0; `nvidia-smi` 580.108 inside WSL | `c1_system.log` |
| CPU | 11th Gen Intel Core i5-11400H, 12 threads | `c1_system.log` |
| RAM | 8 GB physical; WSL limited to 6 GB (`memory=6GB`) plus 12 GB swap file on D: (`.wslconfig`) | `c1_system.log`, `.wslconfig` |
| OS | Windows 11 + WSL2 Ubuntu 22.04, Python 3.10.12 | `c1_system.log` |
| Libraries | torch 2.11.0+cu128, transformers 5.2.0, accelerate 1.15.0, bitsandbytes 0.50.2, kvpress 0.5.5 | `c2_versions_gpu.log` (kvpress printed "?" there; version confirmed afterwards with `pip show kvpress` = 0.5.5) |
| Disk | venv 7.2 GB, model files 3.8 GB | `du -sh` |

## 2. Model

| Item | Value |
|---|---|
| Model | Qwen/Qwen3-1.7B, snapshot `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Weights | loaded in 4-bit NF4 (bitsandbytes); checkpoint itself is bf16 (3441.2 + 622.3 MB files) |
| Shape | 28 layers, 16 query heads, 8 KV heads, head_dim 128, hidden 2048, MLP 6144, vocab 151936 |
| Position limit | 40960 tokens (`max_position_embeddings`), rope_theta 1e6, no rope scaling, no sliding window |
| KV cache size | 2 (K,V) × 28 layers × 8 heads × 128 × 2 bytes = **114,688 B = 112 KiB per token** |
| | 4096 tok → 0.44 GiB, 8192 → 0.88 GiB, 16384 → 1.75 GiB, 24576 → 2.62 GiB, 32768 → 3.50 GiB |

Source: `c5_model.log`.

---

## 3. Setup checks, results and fixes

Each check below says **why** it was run, **what** was measured, the **result**, and **what was changed**.

### Check 1–3: system, installation, library source

- Scripts: `c1_system.sh`, `c2_install.sh`, `c2_verify.sh`, `c2_test.py`, `c3_libs.sh`, `c3b_read.sh`, `c3c_read.sh`
- Logs: `c1_system.log`, `c2_versions_gpu.log`, `c3_libs.log`, `c3b_read.log`, `c3c_read.log`

Results:
- The GPU is visible to PyTorch. torch is compiled for sm_75. An FP16 matmul and a bitsandbytes 4-bit layer both run
  on the GPU with finite output.
- **transformers 5.2 has no separate `OffloadedCache` class any more.** Offloading is a flag:
  `DynamicCache(offloading=True)` / `StaticCache(..., offloading=True)`.
  - Each layer is moved to the CPU with `.to("cpu", non_blocking=True)`.
  - The next layer is brought back on a separate "prefetch" CUDA stream.
  - Source: `cache_utils.py` lines 57–67 and 740–800, printed in `c3b_read.log`.
- kvpress 0.5.5 has `StreamingLLMPress` and `SnapKVPress`, and explicitly supports `Qwen3Attention` (`kvpress/utils.py`).
- **How kvpress compresses (read from source, `c3c_read.log`):**
  - Presses act only during prefill (`is_prefilling`).
  - The pipeline compresses the context **first**, then appends the question without compressing it.
  - So SnapKV's choice of tokens does **not** see the question. This is how kvpress implements it and must be stated
    in the paper.
- StreamingLLM in kvpress keeps `n_sink = 4` first tokens. To renumber positions exactly as in the StreamingLLM paper,
  kvpress says to wrap it in `KeyRerotationPress` (`c3b_read.log`).

### Check 4: CPU↔GPU transfer (PCIe) speed and pinned-memory limit

- Why: offloading moves the KV cache over PCIe every generated token. Its speed bounds offloading speed.
- Scripts: `c4_pcie.py`, `c4b_pinlimit.py`
- Logs: `c4_pcie.log`, `c4b_pinlimit.log`

| Copy (128 MiB) | GB/s |
|---|---|
| CPU→GPU, pageable memory | 6.78 |
| CPU→GPU, pinned memory | 9.01 |
| GPU→CPU, pageable memory | 4.97 |
| GPU→CPU, pinned memory | 9.53 |

- Pinned memory: a single 1 GiB and a single 2 GiB block worked. **A single 3 GiB block failed**
  (`CUDA error: out of memory`).
- In 64 MiB pieces, a total of **3392 MiB (3.31 GiB)** could be pinned before failure.
- Inference: an offloaded KV cache can be at most about 3.3 GiB, which is roughly 30k tokens.
  In practice it is lower, because the 6 GB WSL RAM also holds the model process (see check 11).

### Check 5–6: model loads and answers; duplicate output layer

- Scripts: `c5_model.py`, `c6_load_answer.py`, `c6b_memory.py`, `c6c_tiecheck.py`, `c6d_tie_verify.py`
- Logs: matching names

Check 6, first load (`c6_load_answer.log`):
- Load time 60.3 s. 1944 MiB of weights on the GPU. Attention implementation: `sdpa`.
- Answer check, greedy decoding:

| Question | Answer | Result |
|---|---|---|
| Capital of France | "Paris" | PASS |
| 17 × 23 | "17 × 23 = 391" | PASS |
| Largest planet | "Jupiter" | PASS |
| 2, 4, 8, 16, … next number | starts explaining "The sequence is: … geometric sequence …" and never gives 32 | FAIL |

- **3/4 correct.** The failure is an instruction-following failure of the model (it explains instead of answering),
  not a setup fault. It is reported as is.

Check 6b, memory breakdown (`c6b_memory.log`):
- 4-bit attention/MLP weights 672 MiB, `embed_tokens` 594 MiB, `lm_head` 594 MiB.
- transformers printed: "both are present in the checkpoints, so we will NOT tie them".
- So the embedding table was stored twice on the GPU.

Check 6c (`c6c_tiecheck.log`): the two tensors were read straight from the checkpoint files.
- Shape (151936, 2048), dtype bf16 for both.
- **Bit-identical: True.**

Check 6d (`c6d_tie_verify.log`): sharing one tensor for both (tying).
- GPU memory allocated: 1944 → **1359 MiB** (−585 MiB).
- Output tokens identical untied vs tied for all 4 questions: **[True, True, True, True]**.
- **Fix 4 adopted:** `model.lm_head.weight = model.model.embed_tokens.weight`.

### Check 7: first capacity test, and the attention kernel

First capacity test (`c7_capacity.py` → `c7_capacity.log`, 11:50):

| Cache | Tokens | Result |
|---|---|---|
| GPU-only | 4096 | OK, but prefill only 34 tok/s; **peak GPU 4401 MiB on a 4096 MiB card** |
| GPU-only | 8192 | out of memory: one step tried to allocate **4.00 GiB** |
| transformers offload | 4096 | OK, 54 tok/s prefill, decode 1.12 tok/s, peak GPU 3985 MiB |
| transformers offload | 8192 | out of memory |

Two problems were found:
1. A 4.00 GiB allocation at 8192 tokens = 16 heads × 8192 × 8192 × 4 bytes, i.e. the **full attention matrix**.
2. A peak above the card's size means memory was **spilling into system RAM** (check 8a).

Check 7b, which attention kernels run on this GPU (`c7b_sdpa.py` → `c7b_sdpa.log`).
Test: one attention call, 4096 tokens, 16 query heads / 8 KV heads.

| PyTorch SDPA kernel | `enable_gqa=True` (8 KV heads passed as is) | `enable_gqa=False` (KV heads repeated to 16) |
|---|---|---|
| FLASH_ATTENTION | not available (needs sm80+) | not available |
| EFFICIENT_ATTENTION | not available | **OK, extra peak 16 MiB** |
| CUDNN_ATTENTION | not available (needs sm80+) | not available |
| MATH | OK, extra peak 2536 MiB | OK, extra peak 2496 MiB |

- transformers' `sdpa_attention.py` (line 28, `use_gqa_in_sdpa`) passes `enable_gqa=True` on torch ≥ 2.8.
- On sm_75 that forces the MATH kernel, which builds the full attention matrix.

Check 7c, the fix (`c7c_gqa_fix_verify.log`):
- `use_gqa_in_sdpa` is made to return False. transformers then repeats the KV heads itself (`repeat_kv`) and the
  memory-efficient kernel is used.
- Extra peak for a 1000-token forward: 316 → 175 MiB.
- Next-token scores: max abs difference 0.0234, mean 0.00445; **top-1 token same: True**.
  The top-5 order differed (rounding differences between kernels).
- 4 answers token-identical: **[True, True, True, True]**.
- **Fix 2 adopted.**

Check 7 rerun with the attention fix (`c7_capacity_run2.log`, 15:26):
- GPU-only "worked" up to 16384 tokens, with peak GPU **3991 MiB**.
- But decode fell to 0.91 tok/s and prefill to 63 tok/s.
- This result is **not valid**: the GPU only has about 3296 MiB free (check 8a), so part of this ran from system RAM.

### Check 8: memory spilling and slow FP16 maths

Check 8a, fill the GPU until allocation fails (`c8a_gpufill.py` → `c8a_gpufill.log`):
- Driver reported **3296 MiB free of 4096 MiB** (the rest is used by Windows and the CUDA context).
- PyTorch nevertheless allocated **7104 MiB** in 111 chunks before failing.
- Then the read-speed measurement itself crashed (out of memory).
- Inference: the NVIDIA driver's **"CUDA – Sysmem Fallback Policy"** is active inside WSL. Allocations beyond VRAM
  silently go to system RAM over PCIe.
  - The user set "Prefer No Sysmem Fallback" in the NVIDIA Control Panel.
  - The behaviour inside WSL did not change.
  - So a software limit was needed (check 9a).

Check 8b, FP16 vs FP32 compute (`c8b_speed.py` → `c8b_speed.log`).

Raw matrix multiply 2048×2048 @ 2048×6144:

| Precision | Time | Speed |
|---|---|---|
| FP16 | 158.98 ms | 0.32 TFLOPS |
| FP32 | 47.07 ms | 1.09 TFLOPS |

The GTX 1650 (TU117) is slower in FP16 than in FP32 for these GEMMs.

Model with NF4 weights, 2048-token prompt:

| Compute dtype | Prefill | Decode | Answers correct |
|---|---|---|---|
| float16 | 19.5 s (105 tok/s) | 6.41 tok/s | 3/3 |
| float32 | **6.6 s (311 tok/s)** | 6.58 tok/s | 3/3 |

- **Fix 3 adopted:** `bnb_4bit_compute_dtype=torch.float32`.
- Weights stay 4-bit and the KV cache stays FP16; only the matrix maths runs in FP32.

### Check 9: hard VRAM limit, capacity, and kvpress on Qwen3

Check 9a, hard limit (`c9a_limit_fill.py` → `c9a_limit_fill.log`):
- `torch.cuda.set_per_process_memory_fraction` is set to (free VRAM at start − 256 MiB).
  Here: free 3296 MiB → limit **3040 MiB**.
- The 256 MiB headroom is for memory outside PyTorch's allocator (cuBLAS and bitsandbytes workspaces).
- The fill test then stopped at **3008 MiB**, with "driver free now 280 MiB".
- Read speed of the last chunk: 132.2 GB/s, i.e. real VRAM, not PCIe.
- Verdict: **PASS, no spill. Fix 1 adopted.** All later numbers use this limit.

Check 9b, capacity with fixes 1–4 (`c9b_capacity.py` → `c9b_capacity.log`):

| Cache | Tokens | Result |
|---|---|---|
| GPU-only | 4096 | OK, prefill 18.5 s (222 tok/s), decode 4.22 tok/s, peak 2097 MiB |
| GPU-only | 6144 | OK, prefill 23.5 s (261 tok/s), decode 3.54 tok/s, peak 2467 MiB |
| GPU-only | 8192 | out of memory (limit 3040 MiB) |
| transformers offload | 4096 | OK, decode 1.11 tok/s, but available RAM fell to 734 MiB |
| transformers offload | 8192 | **crashed**: available RAM 62 MiB, swap 1166 MiB, then CUDA out of memory and process abort |

These two failures led to checks 10 and 11.

Check 9c, kvpress presses on this model (`c9c_presses.py` → `c9c_presses.log`):
- Context: 4039 tokens with 3 facts hidden at 10 %, 50 % and 90 % depth. 3 questions.
  Greedy decoding, max 24 new tokens.

| Press | KV tokens kept | Fact at 10 % | Fact at 50 % | Fact at 90 % | Time | Peak GPU |
|---|---|---|---|---|---|---|
| none (full cache) | 4042 | OK | OK | OK | 34.1 s | 2089 MiB |
| StreamingLLM keep 50 % | 2021 | **MISS** ("Biscuit") | OK | OK | 26.4 s | 1868 MiB |
| StreamingLLM + KeyRerotation, keep 50 % | 2021 | **MISS** ("Biscuit") | OK | OK | 26.4 s | 1867 MiB |
| SnapKV keep 50 % | 2021 | OK | OK | OK | 25.3 s | 1868 MiB |
| SnapKV keep 25 % | 1010 | OK | **MISS** ("not provided in the text") | OK | 26.2 s | 1757 MiB |

- Both presses run on Qwen3 and keep exactly the requested number of tokens.
- Their misses are the expected behaviour:
  - StreamingLLM drops the middle-to-early part of the text, which contained the 10 % fact. That fact sits after the
    4 sink tokens.
  - SnapKV at 25 % dropped the middle fact.
- Rerotation made no difference here.

### Check 10: why GPU-only stopped at 6k, and why transformers' offloading uses so much RAM

Check 10a, MLP in pieces (`c10a_mlp_chunk.py` → `c10a_mlp_chunk.log`):
- During prefill the MLP of each layer is run on 1024-token pieces instead of all tokens at once.
- The MLP works on each token separately, so the result is mathematically the same.

| 6144 tokens | Peak GPU | Prefill | Decode |
|---|---|---|---|
| without | 2467 MiB | 25.0 s | 3.57 tok/s |
| with | **2275 MiB** | 27.3 s | 3.61 tok/s |

- Last-token scores: max difference 0.0343, top-1 same: True. **16 generated tokens identical: True.**
- With the fix, GPU-only: 8192 tokens OK (peak 2579 MiB, decode 2.90 tok/s); **10240 out of memory.**
- **Fix 5 adopted.** Cost: about 9 % slower prefill at 6k.

Check 10d, where the memory goes at 8192 tokens (`c10d_where.py` → `c10d_where_8192.log`):
- Peak temporary memory per step is small (attention +256 MiB, MLP +120 MiB).
- After prefill, PyTorch had **allocated 2255 MiB but reserved 2724 MiB**: a gap of **469 MiB** held but unusable.
- Inference: **memory fragmentation** of PyTorch's caching allocator, caused by the KV cache growing with `torch.cat`.

Check 10e, fragmentation fix and static cache. Runner `c10e_runall.sh`; logs `c10e_dynamic_expandable.log`,
`c10e_static.log`, `c10e_static_expandable.log`.

| Cache | `PYTORCH_CUDA_ALLOC_CONF` | 10240 tokens | 12288 tokens |
|---|---|---|---|
| dynamic | `expandable_segments:True` | **OK**, peak allocated 2883, reserved 3004 MiB, prefill 60.5 s, decode 1.87 tok/s | out of memory |
| static (pre-allocated) | none | out of memory | – |
| static | `expandable_segments:True` | out of memory | – |

- At 4096 tokens the static cache gave tokens identical to dynamic. Static decode was faster:
  4.19 vs 3.54 tok/s, and 4.36 vs 3.04 tok/s.
- **Fix 6 adopted:** `expandable_segments:True` (set in `run_py.sh`).
- **Final GPU-only limit: 10240 tokens OK, 12288 out of memory.**
- Decision: **the dynamic cache is used for all methods**, because kvpress works on the dynamic cache and it reaches
  a longer length here.

Check 10c, transformers' built-in offloading, RAM and correctness. Script `c10c_offload_ram.py`.
Logs: `c10c_offload_dynamic.log`, `c10c_offload_dynamic_empty.log`, `c10c_offload_static.log`.
Setup: 4096 tokens (KV size 448 MiB), 16 generated tokens.

| Variant | Pinned RAM after prefill | after 16 tokens | Decode | GPU-only decode | Tokens identical to GPU-only? |
|---|---|---|---|---|---|
| `DynamicCache(offloading=True)` | 448 MiB | **1552 MiB** | 0.55 tok/s | 3.40 | **False** |
| same + `torch._C._host_emptyCache()` each step | 448 MiB | 896 MiB | 0.64 tok/s | 3.16 | **False** |
| `StaticCache(offloading=True)` | 896 MiB | 1280 MiB | 1.96 tok/s | 3.35 | **False** |

Pinned RAM grows to 2–3.5× the KV size, because new pinned buffers are allocated at every step.
**More importantly, the outputs are wrong.**

Check 10g, cause of the wrong outputs (`c10g_offload_race.py` → `c10g_offload_race.log`):

| Run | Generated tokens | Decoded |
|---|---|---|
| GPU-only | [582, 728, 13, 2619, 323, …] | " we go. There and back again. The grass is green. The sky is" |
| transformers offload, as shipped | [582, 728, **1438**, 6684, 46451, …] | " we go break�现芳/*+ [[',ialressing\|无论_" |
| transformers offload + `torch.cuda.synchronize()` after every copy | identical to GPU-only | identical |

- First-token scores are identical (difference 0.0000), so prefill is fine. Generation goes wrong from the 3rd token.
- Forcing a synchronize fixes it.
- Inference: a **race condition** in transformers 5.2's offloading. The GPU→CPU copy is `non_blocking`, and the next
  prefetch, on another CUDA stream, can read the buffer before that copy has finished.
- This is a bug in the library, not in our setup. **transformers' offloading is therefore not used.**

### Check 11: own offloaded KV cache (`kvoffload.py`)

Why: transformers' offloading is wrong (check 10g) and wastes RAM (check 10c). A correct offloaded cache is needed
for the comparison.

Design (first version):
- One CPU buffer holding the whole KV cache, allocated once at its exact size and pinned in place with
  `cudaHostRegister`. This avoids PyTorch's pinned allocator, which rounds up to powers of two.
- One reusable GPU staging buffer.
- Every copy and every kernel runs on the same CUDA stream, so a read can never overtake a write.
  Correct by construction.
- Only new tokens are copied GPU→CPU. A layer's stored tokens are copied CPU→GPU when the layer runs.
- No overlap of copying and computing (no prefetch), so the transfer cost is fully visible.

Script `c11_offload_verify.py` → `c11_offload_verify.log`:

(1) Correctness at 4096 tokens:

| Cache | Prefill | Decode | Peak GPU | RAM taken (KV size 448 MiB) |
|---|---|---|---|---|
| GPU-only | 20.3 s (202 tok/s) | 3.82 tok/s | 1976 MiB | 265 MiB |
| own offload | 20.0 s (205 tok/s) | 0.75 tok/s | 1552 MiB | 737 MiB |

- **16 generated tokens identical: True.** First-token scores max abs difference **0.000000**.
- Both texts: " we go. There and back again. The grass is green. The sky is".

(2) Maximum length with own offload:

| Tokens | Prefill | Decode | Peak GPU | RAM taken | KV size | Valid? |
|---|---|---|---|---|---|---|
| 8192 | 46.8 s (175 tok/s) | 1.09 tok/s | 1746 MiB | 969 MiB | 896 MiB | yes |
| 12288 | 88.4 s (139 tok/s) | 0.43 tok/s | 1940 MiB | 1357 MiB | 1344 MiB | yes |
| 16384 | 143.9 s (114 tok/s) | 0.23 tok/s | 2134 MiB | 1828 MiB | 1792 MiB | yes |
| 20480 | 204.3 s (100 tok/s) | 0.20 tok/s | 2328 MiB | 2093 MiB | 2240 MiB | yes (swap +93 MiB, under the 256 MiB threshold) |
| 24576 | 270.3 s (91 tok/s) | 0.46 tok/s | 2522 MiB | 2163 MiB | 2688 MiB | **NO: swap grew 1462 MiB** |

- RAM taken ≈ 1× the KV size, compared with 2–3.5× for transformers' offloading.
- At 20480 tokens "RAM taken" is less than the KV size because Linux freed file cache at the same time.
  The measurement is "drop in available RAM", not an exact count.
- **Offload limit: 20480 tokens.** It is set by the 6 GB WSL RAM, not by the GPU.
  The 24576 numbers are not used, because swapping makes them meaningless.
- **Gap in this log:** part (3), "GPU-only with all fixes: final limit", printed its header but **no result line**.
  The run ended before it finished. The GPU-only limit therefore comes from check 10e (10240 OK, 12288 out of
  memory), which used the same fixes.
- Problem found: decode was very slow (0.20–1.09 tok/s).
  - The layout was head-major (`[layer][K/V][head][token][dim]`), so a layer's first *n* tokens were not one
    contiguous block in memory.
  - PyTorch therefore repacked them on the CPU before each copy.

### Check 12: token-major layout for own offload

- Change: storage order `[layer][token][K/V][head][dim]`. A layer's first *n* tokens are one contiguous block, copied
  in a single transfer. Nothing else changed.
- Script `c12_offload_speed.py` → `c12_offload_speed.log`
- Correctness at 4096 tokens: **tokens identical to GPU-only: True**, first-token scores max abs difference **0.000000**.

| Tokens | GPU-only decode | Own offload, new layout | Own offload, old layout (check 11) | Time to copy the full KV cache once |
|---|---|---|---|---|
| 4096 | 4.35 tok/s | **3.66 tok/s** | 0.75 tok/s | 0.060 s (≤ 16.8 tok/s from copying alone) |
| 8192 | 2.69 tok/s | **2.17 tok/s** | 1.09 tok/s | 0.115 s (≤ 8.7 tok/s from copying alone) |

- Offload is now 16–19 % slower than GPU-only at these lengths. Before it was 2.5–5× slower.
- **Gap:** the 16384-token line was not measured. The extra helper that times a raw full-cache copy
  (`transfer_bound`) tried to put a 1.75 GiB test buffer on the GPU next to the model and ran out of memory.
  - This is a fault in the measuring helper, not in the cache.
  - **The new layout has therefore not yet been run at 16384 or 20480 tokens.** The memory it uses is the same as
    the old layout (same buffers, only a different order).

---

## 4. Final configuration used by the experiment

All in `scripts/checks/kvsetup.py` (imported by every script) and `scripts/checks/run_py.sh`:

| # | Fix | Verified in |
|---|---|---|
| 1 | PyTorch hard VRAM limit = free VRAM − 256 MiB (3040 MiB here) | 9a |
| 2 | `repeat_kv` instead of `enable_gqa` → memory-efficient attention kernel | 7b, 7c |
| 3 | NF4 weights, FP32 compute | 8b |
| 4 | `lm_head` tied to `embed_tokens` (bit-identical in the checkpoint) | 6c, 6d |
| 5 | MLP in 1024-token pieces during prefill | 10a |
| 6 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 10e |
| 7 | own `PinnedOffloadCache`, token-major layout, instead of transformers' offloading | 10g, 11, 12 |

Measured limits on this laptop:
- **GPU-only: 10240 tokens** (12288 fails).
- **Own offload: 20480 tokens** (24576 swaps).

## 5. Status before the experiment: verified and not yet verified

Verified:
- Model loads and answers.
- Every fix keeps outputs identical, or the same top-1 token (fix 2).
- The GPU memory limit holds.
- Own offload produces outputs identical to GPU-only (checks 11 and 12).
- Both kvpress presses run on Qwen3 at about 4k tokens.

Not yet run (to be covered by a 1-sample pilot of the experiment runner before the full run):
1. RULER data has not been generated yet.
2. The experiment runner has not been written yet.
3. Own offload with the new layout at 16384 and 20480 tokens (check 12 gap).
4. StreamingLLM and SnapKV above 4039 tokens. kvpress compresses each layer during prefill, so their maximum length
   may differ from GPU-only. Not measured.
5. GPU-only at 10240 tokens has been run only once (check 10e). Peak reserved was 3004 MiB against a 3040 MiB limit,
   so it is close to the edge.

---

## 6. Experiment preparation

### 6a. RULER code and data

- Code: official NVIDIA RULER, `third_party/RULER`, commit `db43f92ff634bbce1c33d1fb15858aec4bba509a` (2026-11-03).
  Scripts used unmodified. Steps logged by `scripts/exp/e1_get_ruler.sh`, `e1b_read.sh`, `e1c_read.sh`.
- Extra packages (`logs/exp/e2_ruler_deps.log`): nltk 3.10.4 (+ punkt, punkt_tab), wonderwords 3.0.1,
  html2text 2025.4.15, beautifulsoup4 4.15.0, tenacity 9.2.1.
  - torch and transformers versions checked unchanged before and after.
- Haystack text, Paul Graham essays (`logs/exp/e2b_finish_essays.log`):
  - RULER's downloader fetched all 169 paulgraham.com pages.
  - It fetched the 49 GitHub raw files at about 1 per minute, so it was stopped after 10.
  - The other 39 were fetched with `curl -L` from the same URLs.
  - The JSON was then assembled with the same code as the downloader's lines 51–62.
  - Result: 218 essays, 3,048,891 characters, 525,469 words. sha256 `8d31e1b6…d3dd`.
- Data (`scripts/exp/e3_gen_ruler.sh` → `logs/exp/e3_gen_ruler.log`):
  - Generated with RULER's `prepare.py`, `--tokenizer_type hf` (Qwen3-1.7B tokenizer),
    `--model_template_type base`, `--num_samples 5`, default seed 42.
  - Tasks: niah_single_2, niah_multikey_1, niah_multivalue, vt. Lengths 4096, 8192, 16384, 20480.
    80 samples, 4.3 MB, in `data/ruler/<length>/<task>/validation.jsonl`.
  - Check: every reference answer is present in its input text (True for all 16 groups).
  - Input tokens per sample (Qwen3 tokenizer) are below the length, because RULER's length includes its
    tokens_to_generate (128 for niah, 30 for vt) and RULER sizes the haystack in steps.
    For example, 4096 niah_single_2 = 3330–3334 tokens and 20480 vt = 20402–20409 tokens.
    The runner records the actual count of every sample.
  - With the same seed at every length, some needles repeat across lengths (for example 8973915 in niah_multivalue at
    8192, 16384 and 20480). The haystack around them differs.

### 6b. Runner (`scripts/exp/run_ruler.py`, started by `scripts/exp/run_exp.sh`)

- Prompt: Qwen3 chat template with thinking off. Context and question are split as in kvpress' pipeline
  (`kvpress/pipeline.py`, read in `e1c_read.sh`).
  - Context part: instruction + haystack.
  - Question part: RULER question + end of user turn + assistant start + RULER answer prefix.
- Steps per sample:
  1. Prefill the context with `model.model` (no output layer). The press is active only in this step.
  2. Question pass with positions starting at the original context length, as kvpress does
     (the compressed length for KeyRerotation).
  3. Greedy decoding until an end token or RULER's tokens_to_generate.
- StreamingLLM is wrapped in `KeyRerotationPress`, as kvpress advises for the paper's behaviour.
- Budgets: 1024 and 2048 kept context tokens. `compression_ratio = 1 − budget / context tokens`.
- Score: RULER's `string_match_all` and `postprocess_pred`, copied from `scripts/eval/`.
- Saved per run:
  - prefill time, time to first token, decode tok/s, tokens generated, end token reached
  - peak GPU memory allocated and reserved
  - peak process RAM (RSS), lowest available RAM, swap growth (sampled every 0.1 s)
  - output text and score
- Resumable. A method that runs out of memory at a length is skipped at that length and longer.

### 6c. Pilot run 1: stopped because the GPU overheated

`results/exp/pilot1_throttled.jsonl`, `logs/exp/pilot1_throttled.log`. Task niah_single_2, sample 0, 2026-11-19 20:33–20:46.

| Length | Method | Context tokens | Kept | Prefill | Decode | Tokens generated | Score |
|---|---|---|---|---|---|---|---|
| 4096 | gpu | 3319 | 3319 | 16.6 s | 3.14 tok/s | 83 | 1 |
| 4096 | offload | 3319 | 3319 | 19.3 s | 3.74 tok/s | 83 | 1 |
| 4096 | stream_1024 | 3319 | 1024 | 23.0 s | 4.88 tok/s | 37 | 1 |
| 4096 | stream_2048 | 3319 | 2048 | 22.9 s | 3.43 tok/s | 11 | 1 |
| 4096 | snapkv_1024 | 3319 | 1024 | 29.5 s | 4.08 tok/s | 45 | 0 (answered "5663" for 5663623) |
| 4096 | snapkv_2048 | 3319 | 2048 | 36.1 s | 2.57 tok/s | 128 | 1 |
| 8192 | gpu | 8034 | 8034 | **138.7 s** | **0.80 tok/s** | 91 | 1 |

- **Finding:**
  - GPU-only at 8192 was about 3× slower than in check 10a (41.6 s prefill, 2.90 tok/s).
  - `nvidia-smi` sampled every second for 20 s during the run:
    **temperature 87–88 °C, SM clock mostly 300 MHz (maximum 1785), 100 % utilisation**,
    clock-event reasons `0x24` = software thermal slowdown (0x20) + software power cap (0x04).
  - The laptop was on AC power (96 %), Windows "Balanced" power plan.
  - After stopping, the GPU was still at 80 °C idle about 2 minutes later.
- **Conclusion:** the GPU throttles because of heat. The speed numbers of this pilot are not valid.
  Prefill time of the same 3319-token input also rose with each run (16.6 → 36.1 s), which points the same way.
  Answers and scores are not affected by clock speed.
- **Fix in the runner:**
  - Before every run, wait until the GPU is ≤ 60 °C (at most 10 min); the wait and the start temperature are saved.
  - During every run, `nvidia-smi` is sampled every 0.5 s. Saved per run: maximum temperature, average and minimum SM
    clock, average power, and the fraction of samples with thermal slowdown (bits 0x20, 0x40, 0x08) and with
    power cap (0x04).
  - Runs with thermal slowdown can then be identified and reported, not hidden.
- **User action:** the user improved the laptop's cooling (fan/performance mode, airflow).
  Idle GPU temperature then measured 45 °C, steady over 1 minute (it was 75–80 °C before).

### 6d. Pilot run 2: valid

`results/exp/pilot2.jsonl`, `logs/exp/pilot2.log`, 2026-11-19 23:29 to 2026-11-20 00:14.
Same sample as pilot 1 (niah_single_2, sample 0), with the cool-down gate (≤ 60 °C) and GPU monitoring.

| Length | Method | Context | Kept | Prefill | Decode | Generated | Peak GPU | Peak RSS | Avg SM clock | Max temp | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | gpu | 3319 | 3319 | 13.2 s | 5.08 tok/s | 83 | 1875 MiB | 3086 MiB | 1359 MHz | 62 °C | 1 |
| 4096 | offload | 3319 | 3319 | 13.8 s | 4.35 | 83 | 1525 | 3469 | 1401 | 65 | 1 |
| 4096 | stream_1024 | 3319 | 1024 | 15.2 s | 8.43 | 37 | 1624 | 3174 | 1018 | 66 | 1 |
| 4096 | stream_2048 | 3319 | 2048 | 13.9 s | 6.62 | 11 | 1736 | 3174 | 1012 | 66 | 1 |
| 4096 | snapkv_1024 | 3319 | 1024 | 14.4 s | 8.42 | 45 | 1624 | 3207 | 1098 | 69 | 0 ("5663") |
| 4096 | snapkv_2048 | 3319 | 2048 | 14.2 s | 6.55 | 128 | 1736 | 3207 | 1229 | 67 | 1 |
| 8192 | gpu | 8034 | 8034 | 43.0 s | 2.90 | 91 | 2555 | 3213 | 1260 | 69 | 1 |
| 8192 | offload | 8034 | 8034 | 43.7 s | 2.29 | 91 | 1739 | 4111 | 1331 | 71 | 1 |
| 8192 | stream_1024 | 8034 | 1023 | 44.2 s | 8.22 | 71 | 1815 | 3213 | 1076 | 71 | 0 ("10") |
| 8192 | stream_2048 | 8034 | 2047 | 44.0 s | 6.39 | 116 | 1923 | 3213 | 1157 | 72 | 0 ("10") |
| 8192 | snapkv_1024 | 8034 | 1023 | 44.6 s | 8.22 | 48 | 1815 | 3213 | 1062 | 71 | 0 ("7214") |
| 8192 | snapkv_2048 | 8034 | 2047 | 44.3 s | 6.43 | 78 | 1923 | 3213 | 1120 | 71 | 1 |
| 16384 | gpu | 16225 | – | **out of GPU memory** (tried to allocate 128 MiB; 354 MiB free) | | | | 3219 | 1164 | 76 | – |
| 16384 | offload | 16225 | 16225 | 118.2 s | 1.28 | 58 | 2127 | 4885 | 1305 | 74 | 1 |
| 16384 | stream_1024 | 16225 | 1024 | 118.3 s | 7.99 | 67 | 2171 | 3091 | 1185 | 73 | 0 ("10") |
| 16384 | stream_2048 | 16225 | 2048 | 118.7 s | 6.30 | 72 | 2279 | 3091 | 1194 | 74 | 0 ("10") |
| 16384 | snapkv_1024 | 16225 | 1024 | 119.2 s | 8.04 | 40 | 2171 | 3091 | 1180 | 74 | 0 ("10") |
| 16384 | snapkv_2048 | 16225 | 2048 | 119.4 s | 6.31 | 65 | 2279 | 3091 | 1193 | 75 | 0 ("5") |
| 20480 | gpu | – | – | skipped (failed at 16384) | | | | | | | – |
| 20480 | offload | 20323 | 20323 | 174.1 s | 0.86 | 67 | 2322 | 5087 | 1313 | 78 | 1 |
| 20480 | stream_1024 | 20323 | 1023 | 170.5 s | 7.93 | 52 | 2349 | 2846 | 1194 | 76 | 0 ("10") |
| 20480 | stream_2048 | 20323 | 2047 | 171.4 s | 6.15 | 83 | 2457 | 2846 | 1206 | 78 | 0 ("10") |
| 20480 | snapkv_1024 | 20323 | 1023 | 172.4 s | 7.87 | 44 | 2349 | 2847 | 1181 | 77 | 0 ("10") |
| 20480 | snapkv_2048 | 20323 | 2047 | 172.2 s | 6.17 | 35 | 2457 | 2847 | 1189 | 77 | 0 ("10") |

Swap growth was 0–5 MiB in every run.

Findings:
- **Thermal fix works.**
  - Prefill time for the same input is now steady: 13.2–15.2 s at 4096, 43.0–44.6 s at 8192.
  - GPU-only at 8192 (43.0 s, 2.90 tok/s) matches check 10a (41.6 s, 2.90 tok/s).
- **Throttle flags.** Even at 60–78 °C the driver reports software thermal slowdown (0x20) and power cap (0x04)
  in almost every sample. The SM clock moves between about 690 and 1785 MHz.
  - This is the laptop's own power/thermal policy (the GPU and CPU share one cooling system), not overheating.
  - So `thermal_throttle_frac` is close to 100 % in every run and cannot separate good runs from bad ones.
  - **The average SM clock is the number to report.** It was about 1000–1400 MHz here, against about 300 MHz in
    pilot 1.
- **Answers are deterministic.** All 4096 answers equal pilot 1's, including the wrong "5663".
- **Gaps from section 5 are now closed:**
  - The token-major offload runs at 16384 and 20480. Its answers are correct and RAM stays without swap
    (peak RSS 5087 MiB at 20480).
  - Both presses run at every length up to 20480.
  - GPU-only fails at 16384, consistent with the 10240/12288 limit from check 10e.
- **Prefill time is the same for all methods at a length.** Compression is done during prefill and saves no prefill
  time. It only saves decode time and memory.
- **Kept tokens were 1023/2047 instead of 1024/2048** at 8192 and 20480.
  - Cause: kvpress keeps `int(L × (1 − ratio))` tokens, and `ratio = 1 − b/L` lands just below b in floating point.
  - Fix for the main run: `ratio = 1 − (b + 0.5)/L`, so the floor is exactly b.
- **Time for one sample across all methods and lengths:** about 45 minutes; details below.

### 6e. Main run: two failed starts and their fixes

**Attempt 1** (`logs/exp/main_attempt1_load_oom.log`, 2026-11-20 00:15):
- Out of GPU memory while *loading* the model, before any test: "2.95 GiB allowed by PyTorch".
- Cause:
  - transformers 5.2 (`core_model_loading.py` line 781: `GLOBAL_WORKERS = min(4, cpu_count)`) moves weights to the
    GPU with 4 threads before 4-bit quantisation.
  - Unquantised tensors can pile up, depending on timing.
  - 17 earlier loads had all ended at 1350–1351 MiB; this was the first failure.
- The failed process did not exit by itself (it kept 3.1 GiB of GPU memory and 5 GB RSS) and was stopped.
- **Fix 8:** `kvsetup.load_model()` sets `transformers.core_model_loading.GLOBAL_WORKERS = 1` (sequential loading,
  same weights).
- Verified:
  - The next load was 1350 MiB on the GPU.
  - The first main-run output (gpu, niah_single_2, 4096, sample 0) is **identical in full text** to pilot 2's
    (83 tokens each; `scripts/exp/e12_compare.sh`).

**Attempt 2** (`logs/exp/main_attempt2_cuda_unknown.log`, 00:18):
- The model loaded (1350 MiB). The first prefill then failed with `CUDA error: unknown error` (cudaErrorUnknown),
  inside rotary embedding `torch.cat`.
- The runner then also failed on its own bug (`UnboundLocalError`: no result variable for non-memory errors), and the
  process again did not exit by itself.
- Diagnosis (`scripts/exp/e10_diag.sh`):
  - Windows System log: no errors or warnings in the previous 70 min, so no driver reset was logged.
  - WSL kernel log: no GPU messages.
  - GPU health test afterwards: 200 × (2048×2048 matmul) OK, max difference vs CPU 0.00055; 2.5 GiB allocation and
    fill OK.
- Conclusion: a one-time failure in the Windows–WSL GPU path. Not reproducible; the GPU was healthy.
- **Fix in the runner:**
  - Any other exception is recorded as `status: "error"`, its output line is written and flushed, and the process
    exits with `os._exit(3)` (a normal exit hung).
  - "error" records are not counted as done, so they are redone.
  - `run_exp.sh` restarts the runner up to 5 times, 60 s apart; it resumes from the results file.
  - Every attempt is logged.

**Attempt 3** (`logs/exp/main.log`, from 00:23) is the main run.

---

## 7. Main run 1: results (1 sample per task)

- Run: `results/exp/main.jsonl`, `logs/exp/main.log`, 2026-11-20 00:23–04:08.
- Summary made by `scripts/exp/summarize.py` → `results/exp/main_summary.txt` and `.csv`.
- 97 records:

| Status | Count | Meaning |
|---|---|---|
| ok | 88 | completed |
| error | 1 | redone after automatic restart |
| oom | 1 | GPU-only at 16384, out of memory |
| skipped_after_failure | 7 | GPU-only at 16384 and 20480 |

- **The single error:** 00:41:29, 4096 niah_multivalue snapkv_1024, `CUDA error: unknown error` again (second time,
  after attempt 2 in section 6e).
  - The runner recorded it, exited in 3 s, `run_exp.sh` restarted after 60 s, and the run was redone OK at 00:45.
  - No other errors in 3 h 45 min.
- All press runs kept exactly 1024 or 2048 tokens (rounding fix confirmed).
- Swap growth was ≤ 16 MiB in every run.

### RULER score (string_match_all × 100; 1 sample per task)

| Length | Method | single | multikey | multivalue | vt | Mean |
|---|---|---|---|---|---|---|
| 4096 | gpu | 100 | 100 | 100 | 100 | 100.0 |
| 4096 | offload | 100 | 100 | 100 | 100 | 100.0 |
| 4096 | stream_1024 | 100 | 0 | 50 | 80 | 57.5 |
| 4096 | stream_2048 | 100 | 0 | 75 | 80 | 63.8 |
| 4096 | snapkv_1024 | 0 | 0 | 0 | 0 | 0.0 |
| 4096 | snapkv_2048 | 100 | 0 | 0 | 0 | 25.0 |
| 8192 | gpu | 100 | 100 | 100 | 100 | 100.0 |
| 8192 | offload | 100 | 100 | 100 | 100 | 100.0 |
| 8192 | stream_1024 | 0 | 0 | 50 | 40 | 22.5 |
| 8192 | stream_2048 | 0 | 0 | 50 | 60 | 27.5 |
| 8192 | snapkv_1024 | 0 | 0 | 0 | 0 | 0.0 |
| 8192 | snapkv_2048 | 100 | 0 | 0 | 0 | 25.0 |
| 16384 | gpu | out of memory | skip | skip | skip | – |
| 16384 | offload | 100 | 100 | 75 | 80 | 88.8 |
| 16384 | stream_1024 | 0 | 0 | 25 | 0 | 6.2 |
| 16384 | stream_2048 | 0 | 0 | 25 | 60 | 21.2 |
| 16384 | snapkv_1024 | 0 | 0 | 0 | 0 | 0.0 |
| 16384 | snapkv_2048 | 0 | 0 | 0 | 0 | 0.0 |
| 20480 | gpu | skip | skip | skip | skip | – |
| 20480 | offload | 100 | 100 | 75 | 100 | 93.8 |
| 20480 | stream_1024 | 0 | 0 | 25 | 40 | 16.2 |
| 20480 | stream_2048 | 0 | 0 | 25 | 60 | 21.2 |
| 20480 | snapkv_1024 | 0 | 0 | 0 | 0 | 0.0 |
| 20480 | snapkv_2048 | 0 | 0 | 0 | 0 | 0.0 |

### Speed and memory (mean over the 4 tasks; peaks are the maximum)

| Length | Method | Prefill (s) | Decode tok/s (min–max) | Peak GPU (MiB) | Peak RSS (MiB) | Avg SM clock (MHz) |
|---|---|---|---|---|---|---|
| 4096 | gpu | 17.6 | 4.47 (4.12–4.60) | 1959 | 3159 | 1148 |
| 4096 | offload | 18.4 | 3.63 (3.48–3.75) | 1548 | 3598 | 1184 |
| 4096 | stream_1024 / 2048 | 18.5 / 17.9 | 7.73 / 6.30 | 1640 / 1748 | 3151 | ~1080 |
| 4096 | snapkv_1024 / 2048 | 18.2 / 17.6 | 7.36 / 6.29 | 1640 / 1748 | 3157 | ~1130 |
| 8192 | gpu | 44.3 | 2.85 (2.79–2.98) | 2564 | 3155 | 1138 |
| 8192 | offload | 45.0 | 2.25 (2.18–2.33) | 1742 | 4052 | 1189 |
| 8192 | stream_1024 / 2048 | 44.1 / 44.1 | 8.02 / 6.31 | 1818 / 1926 | 3155 | ~1090 |
| 8192 | snapkv_1024 / 2048 | 44.4 / 44.3 | 8.07 / 6.33 | 1818 / 1926 | 3155 | ~1070 |
| 16384 | gpu | out of memory | – | – | – | – |
| 16384 | offload | 128.7 | 1.20 (1.19–1.22) | 2130 | 4737 | 1201 |
| 16384 | stream_1024 / 2048 | 124.1 / 123.5 | 7.85 / 6.17 | 2174 / 2282 | 2943 | ~1140 |
| 16384 | snapkv_1024 / 2048 | 124.5 / 124.3 | 7.77 / 6.18 | 2174 / 2282 | 2943 | ~1135 |
| 20480 | gpu | skipped | – | – | – | – |
| 20480 | offload | 181.9 | 0.96 (0.94–0.98) | 2324 | 5114 | 1227 |
| 20480 | stream_1024 / 2048 | 178.2 / 177.7 | 7.72 / 6.08 | 2352 / 2460 | 2861 | ~1155 |
| 20480 | snapkv_1024 / 2048 | 178.2 / 177.5 | 7.74 / 6.09 | 2352 / 2460 | 2862 | ~1160 |

### Observations (facts from the data; interpretation is marked)

- **Maximum context on this laptop:**
  - GPU-only fails at 16384 (out of memory).
  - Offload and both presses complete 20480.
  - Offload's limit is RAM: peak RSS 5114 MiB of the 6 GB WSL limit at 20480.
  - The presses' GPU peak at 20480 is only 2352–2460 MiB.
- **Speed vs length:**
  - Offload decode falls with length: 3.63 → 2.25 → 1.20 → 0.96 tok/s. It is 19–21 % slower than GPU-only where both
    run.
  - Press decode is flat at about 7.7–8.1 (1024 kept) and 6.1–6.3 (2048 kept) tok/s at every length.
- **Prefill time is the same for all methods at a given length** (for example 177.5–181.9 s at 20480).
  Compression happens during prefill and saves no prefill time.
- **Accuracy:**
  - Offload equals GPU-only where both run. They are computationally identical: check 11/12 showed identical tokens.
  - Offload's drop at 16384/20480 (multivalue 75, vt 80) is therefore the model's own accuracy with the full cache,
    not an offloading effect.
  - The presses lose accuracy fast with length.
- **SnapKV scored below StreamingLLM here.** Interpretation, to be stated as such:
  - In kvpress' pipeline (followed exactly) the context is compressed *before* the question is seen
    (question-agnostic; section 3, check 1–3).
  - SnapKV's observation window (last 64 tokens) is therefore the end of the haystack, not the question, so it cannot
    pick the needle.
  - Its outputs often contain a truncated needle (for example "5663" for 5663623, "7214" for 7210606).
  - This is a known limitation of question-agnostic compression, not a bug. It must be stated in the paper.
- **Limitation:** 1 sample per task and length.
  - The scores are single runs with a coarse step (0/25/50/75/100) and must not be read as precise accuracies.
  - Speed and memory numbers are stable across the 4 tasks (min–max ranges above).
  (4096: 2.5 min, 8192: 7.5, 16384: 14.5, 20480: 20; this includes cool-down waits of 0–51 s per run).
