# KV cache offloading vs a truncated GPU-only cache on a 4 GB GTX 1650 — experiment 2

The trade-off between generation speed (tok/s) and the longest usable context when the KV cache is offloaded to
CPU RAM, compared with a truncated (compressed) cache kept strictly in GPU VRAM.

- **Model:** Qwen/Qwen3-1.7B (NF4 weights, FP32 compute), HF snapshot `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- **Methods:** GPU-only (full cache), own pinned CPU offload (`PinnedOffloadCache`), StreamingLLM and SnapKV
  (kvpress) with 1024 and 2048 kept tokens
- **Benchmark:** RULER (niah_single_2, niah_multikey_1, niah_multivalue, vt) at 4096 / 8192 / 16384 / 20480 tokens
- **Hardware:** GTX 1650 laptop 4 GB, i5-11400H, 8 GB RAM, Windows 11 + WSL2 Ubuntu 22.04

**`EXPERIMENT_RECORD.md` is the full record:** every setup check with its result and fix, the final configuration,
the pilot and main runs, and the results. It was written from the log files in this repository. Paths in it are
relative to the project folder (`~/kv-offload/` on the test laptop), which is the root of this repository.

## White paper

`paper/kv_cache_offloading_4gb_gpu.pdf`: *KV Cache Offloading vs Truncated Caches on a 4 GB GPU* (17 pages, with
both figures). Every number in it was re-checked on 2026-11-24 against `results/exp/main.jsonl` and
`EXPERIMENT_RECORD.md`; the two scripts used for that are in `scripts/paper_checks/`
(`verify_paper.py`: run status, output identity GPU-only vs offloading, input sizes, clocks, temperatures, waits,
per-task speed spread, peak memory, SnapKV outputs; `verify_needles.py`: where each answer sits in the input and which
ones StreamingLLM returned). The references were checked against arXiv, NeurIPS, ICLR and USENIX pages.

## Layout

| Folder | What it is |
|---|---|
| `scripts/checks/` | Setup checks (c1–c12), each with a log in `logs/setup/`. `kvsetup.py` holds the final configuration that every script imports; `kvoffload.py` is the own offloaded KV cache; `run_py.sh` runs a check and saves its log. |
| `scripts/exp/` | Experiment: RULER download and data generation (`e1_*`, `e2_*`, `e3_*`), runner `run_ruler.py` started by `run_exp.sh`, summary `summarize.py`, helper scripts used during the run. |
| `results/exp/` | Measured data: `main.jsonl` (one record per run), `main_summary.txt` / `.csv` (tables), and the two pilot runs (`pilot1_throttled.jsonl` was stopped because the GPU overheated, see record 6c). |
| `logs/setup/` | Console output of every setup check (cited in the record). |
| `paper/` | The white paper (PDF). |
| `scripts/paper_checks/` | Scripts that recompute the paper's numbers from the raw results. |
| `logs/exp/` | Console output of RULER preparation, pilots and the main run, including the two failed starts (record 6e). |

## Not included

- **Model weights, Python venv:** downloaded/installed by the scripts (`c2_install.sh`, `c5_model.py`).
- **RULER source** (`third_party/RULER`): official NVIDIA RULER, commit `db43f92ff634bbce1c33d1fb15858aec4bba509a`,
  Apache-2.0, used unmodified. Fetched by `scripts/exp/e1_get_ruler.sh`.
- **Generated RULER data** (`data/ruler/`, 80 samples): it contains Paul Graham essay text, so it is not
  redistributed. It is regenerated with `scripts/exp/e3_gen_ruler.sh` (RULER `prepare.py`, Qwen3 tokenizer,
  5 samples per task, seed 42), after the essays are fetched as described in record section 6a.
