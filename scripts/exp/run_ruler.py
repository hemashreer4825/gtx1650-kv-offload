"""Experiment 2 runner: RULER samples x KV-cache methods. One JSON line per (method, task, length, sample) run.

Methods (all share kvsetup.load_model(): fixes 1-6; same prompt, same greedy decoding):
  gpu          - full KV cache on GPU (transformers DynamicCache)
  offload      - full KV cache in pinned CPU RAM (own PinnedOffloadCache, check 11/12)
  stream_B     - StreamingLLM (kvpress, wrapped in KeyRerotationPress as kvpress advises for the paper's behaviour),
                 context KV cache compressed to B tokens, kept on GPU
  snapkv_B     - SnapKV (kvpress), context KV cache compressed to B tokens, kept on GPU

Per sample the steps copy kvpress' own pipeline (kvpress/pipeline.py, read in step e1c):
  1. prompt = Qwen3 chat template (enable_thinking=False); split into context part and question part
     (question part = RULER question + end of user turn + assistant start + RULER answer prefix)
  2. prefill context with model.model (no lm_head) - press active only here
  3. question pass with position_ids starting at the ORIGINAL context length (compressed length if rerotation)
  4. greedy decoding until EOS or max_new_tokens (RULER's tokens_to_generate)
Score = RULER string_match_all (fraction of reference answers contained in the output, case-insensitive).

Resumable: results already in the output file are skipped. If a method runs out of GPU memory at a length,
its remaining samples at that length (and longer) are recorded as skipped - that is its maximum context.
"""
import argparse, contextlib, gc, json, os, re, subprocess, sys, threading, time
import psutil, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checks"))
from kvsetup import load_model, MiB                    # noqa: E402
from kvoffload import PinnedOffloadCache                # noqa: E402
from transformers import DynamicCache                   # noqa: E402
from kvpress import StreamingLLMPress, SnapKVPress, KeyRerotationPress   # noqa: E402

QUESTION_MARK = {"niah": "\nWhat ", "vt": "\nQuestion:"}      # where RULER's question starts in "input"
MAX_NEW = {"niah": 128, "vt": 30}                             # RULER tokens_to_generate


def string_match_all(pred, refs):                             # RULER scripts/eval/synthetic/constants.py
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def postprocess(pred):                                        # RULER scripts/eval/evaluate.py postprocess_pred
    return re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip()


class Sampler:
    """Background thread: peak process RSS, minimum available RAM, peak swap used (every 0.1 s)."""
    def __init__(self):
        self.p = psutil.Process(); self.stop = False
        self.swap0 = psutil.swap_memory().used
        self.rss = self.p.memory_info().rss; self.avail = psutil.virtual_memory().available; self.swap = self.swap0
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()
    def run(self):
        while not self.stop:
            self.rss = max(self.rss, self.p.memory_info().rss)
            self.avail = min(self.avail, psutil.virtual_memory().available)
            self.swap = max(self.swap, psutil.swap_memory().used)
            time.sleep(0.1)
    def done(self):
        self.stop = True; self.t.join()
        return dict(peak_rss_mib=round(self.rss / MiB), min_avail_ram_mib=round(self.avail / MiB),
                    swap_growth_mib=round((self.swap - self.swap0) / MiB))


SMI = ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active",
       "--format=csv,noheader,nounits"]
THERMAL_BITS = 0x20 | 0x40 | 0x08          # sw thermal slowdown | hw thermal slowdown | hw slowdown
POWER_BIT = 0x04                           # sw power cap


def gpu_now():
    t, c, p, r = subprocess.run(SMI, capture_output=True, text=True).stdout.strip().split(", ")
    return int(t), int(c), float(p), int(r, 16)


class GpuMon:
    """Pilot run 1 found heavy thermal throttling (87-88 C, SM clock 300 of 1785 MHz, reasons 0x24).
    Samples nvidia-smi every 0.5 s during a run; the stats are saved with every result."""
    def __init__(self):
        self.p = subprocess.Popen(SMI + ["-lms", "500"], stdout=subprocess.PIPE, text=True)
        self.rows = []
        self.t = threading.Thread(target=self.read, daemon=True); self.t.start()
    def read(self):
        for line in self.p.stdout:
            try:
                t, c, p, r = line.strip().split(", "); self.rows.append((int(t), int(c), float(p), int(r, 16)))
            except ValueError:
                pass
    def done(self):
        self.p.terminate(); self.p.wait(); self.t.join(timeout=2)
        rows = self.rows or [gpu_now()]
        n = len(rows)
        return dict(gpu_samples=n, gpu_temp_max=max(r[0] for r in rows),
                    gpu_sm_clock_avg=round(sum(r[1] for r in rows) / n), gpu_sm_clock_min=min(r[1] for r in rows),
                    gpu_power_avg_w=round(sum(r[2] for r in rows) / n, 1),
                    thermal_throttle_frac=round(sum(1 for r in rows if r[3] & THERMAL_BITS) / n, 3),
                    power_cap_frac=round(sum(1 for r in rows if r[3] & POWER_BIT) / n, 3))


def cool_down(limit_c, max_wait_s=600):
    t0 = time.time(); temp = gpu_now()[0]
    while temp > limit_c and time.time() - t0 < max_wait_s:
        time.sleep(5); temp = gpu_now()[0]
    return dict(cool_wait_s=round(time.time() - t0), gpu_temp_start=temp)


def build(tok, sample, task):
    inp = sample["input"]
    cut = inp.rfind(QUESTION_MARK[task])
    assert cut > 0, "question marker not found"
    context, question = inp[:cut], inp[cut:]
    sep = "#" * 20
    templ = tok.apply_chat_template([{"role": "user", "content": context + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    ctx_text, suffix = templ.split(sep)
    q_text = question + suffix + sample["answer_prefix"].strip()
    ctx = tok.encode(ctx_text, add_special_tokens=False, return_tensors="pt")
    q = tok.encode(q_text, add_special_tokens=False, return_tensors="pt")
    return ctx, q


def make(method, model, ctx_len, total_len):
    """returns (cache, press)"""
    if method == "gpu":
        return DynamicCache(config=model.config), None
    if method == "offload":
        return PinnedOffloadCache(model.config, total_len), None
    name, b = method.rsplit("_", 1); b = int(b)
    # kvpress keeps int(L * (1 - ratio)) tokens; with ratio = 1 - b/L float error gave b-1 (pilot 2: 1023, 2047).
    # +0.5 makes the floor land exactly on b.
    ratio = 1.0 - (b + 0.5) / ctx_len
    if name == "stream":
        return DynamicCache(config=model.config), KeyRerotationPress(press=StreamingLLMPress(compression_ratio=ratio))
    if name == "snapkv":
        return DynamicCache(config=model.config), SnapKVPress(compression_ratio=ratio)
    raise ValueError(method)


def run_one(model, tok, ctx, q, method, max_new, eos):
    ctx, q = ctx.cuda(), q.cuda()
    L = ctx.shape[1]
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cache, press = make(method, model, L, L + q.shape[1] + max_new + 8)
    smp = Sampler(); mon = GpuMon()
    try:
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with press(model) if press is not None else contextlib.nullcontext():
                model.model(input_ids=ctx, past_key_values=cache)
            torch.cuda.synchronize(); t_prefill = time.perf_counter() - t0
            kept = cache.get_seq_length()
            start = kept if isinstance(press, KeyRerotationPress) else L
            pos = torch.arange(start, start + q.shape[1], device="cuda").unsqueeze(0)
            out = model(input_ids=q, past_key_values=cache, position_ids=pos, logits_to_keep=1)
            ids = [out.logits[0, -1].argmax()]
            torch.cuda.synchronize(); t_first = time.perf_counter() - t0
            pos = pos[:, -1:] + 1
            t1 = time.perf_counter()
            for i in range(max_new - 1):
                if ids[-1].item() in eos:
                    break
                out = model(input_ids=ids[-1].view(1, 1), past_key_values=cache, position_ids=pos + i)
                ids.append(out.logits[0, -1].argmax())
            torch.cuda.synchronize(); t_dec = time.perf_counter() - t1
        n_dec = len(ids) - 1                                     # tokens produced by the decode loop
        text = tok.decode(torch.stack(ids), skip_special_tokens=True)
        r = dict(status="ok", ctx_tokens=L, q_tokens=q.shape[1], kv_kept=kept, prefill_s=round(t_prefill, 3),
                 ttft_s=round(t_first, 3), gen_tokens=len(ids), hit_eos=ids[-1].item() in eos,
                 decode_s=round(t_dec, 3), decode_tok_s=round(n_dec / t_dec, 3) if n_dec > 0 else None,
                 peak_gpu_alloc_mib=round(torch.cuda.max_memory_allocated() / MiB),
                 peak_gpu_reserved_mib=round(torch.cuda.max_memory_reserved() / MiB), output=text)
    except torch.OutOfMemoryError as e:
        r = dict(status="oom", ctx_tokens=L, error=str(e).splitlines()[0][:200])
    except Exception as e:      # main run 2: one "CUDA error: unknown error" (GPU healthy afterwards); record and exit
        r = dict(status="error", ctx_tokens=L, error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", fatal=True)
    finally:
        r.update(smp.done()); r.update(mon.done())
        try:
            if method == "offload":
                cache.close()
            del cache; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            r["cleanup_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"; r["fatal"] = True
    if r["status"] == "ok" and r["swap_growth_mib"] >= 256:
        r["status"] = "invalid_swap"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/kv-offload/data/ruler"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="niah_single_2,niah_multikey_1,niah_multivalue,vt")
    ap.add_argument("--lengths", default="4096,8192,16384,20480")
    ap.add_argument("--methods", default="gpu,offload,stream_1024,stream_2048,snapkv_1024,snapkv_2048")
    ap.add_argument("--n", type=int, default=1, help="samples per task and length")
    ap.add_argument("--cool", type=int, default=60, help="wait until GPU temperature <= this (C) before each run")
    a = ap.parse_args()

    model, tok, limit = load_model()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    done, failed = set(), {}
    if os.path.exists(a.out):
        for line in open(a.out):
            d = json.loads(line)
            if d["status"] == "error":          # crashed runs are redone after a restart
                continue
            done.add((d["method"], d["task"], d["length"], d["index"]))
            if d["status"] in ("oom", "invalid_swap"):
                failed[d["method"]] = min(failed.get(d["method"], 10**9), d["length"])
    print(f"[run] already done: {len(done)} | eos ids {sorted(eos)} | out {a.out}", flush=True)

    # warm-up (not recorded): CUDA kernels / allocator first-use costs
    with torch.no_grad():
        model.model(input_ids=torch.arange(1000, 1512, device="cuda").unsqueeze(0), past_key_values=DynamicCache(config=model.config))
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

    methods = a.methods.split(",")
    with open(a.out, "a") as f:
        for length in map(int, a.lengths.split(",")):
            for task in a.tasks.split(","):
                fam = "vt" if task == "vt" else "niah"
                samples = [json.loads(l) for l in open(f"{a.data}/{length}/{task}/validation.jsonl")][: a.n]
                for s in samples:
                    ctx, q = build(tok, s, fam)
                    for m in methods:
                        key = (m, task, length, s["index"])
                        if key in done:
                            continue
                        base = dict(method=m, task=task, length=length, index=s["index"], refs=s["outputs"])
                        if m in failed and length >= failed[m]:
                            r = dict(status="skipped_after_failure", failed_at=failed[m])
                        else:
                            cw = cool_down(a.cool)
                            r = run_one(model, tok, ctx, q, m, MAX_NEW[fam], eos); r.update(cw)
                            if r["status"] in ("oom", "invalid_swap"):
                                failed[m] = min(failed.get(m, 10**9), length)
                        if r["status"] == "ok":
                            r["score"] = string_match_all(postprocess(r["output"]), s["outputs"])
                        rec = {**base, **r, "time": time.strftime("%F %T")}
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        print(f"{rec['time']} {length:6d} {task:16s} #{s['index']} {m:12s} {r['status']:8s} "
                              + (f"ctx {r['ctx_tokens']} kept {r['kv_kept']} | prefill {r['prefill_s']:.1f}s ttft {r['ttft_s']:.1f}s "
                                 f"| decode {r['decode_tok_s']} tok/s ({r['gen_tokens']} tok) | GPU {r['peak_gpu_alloc_mib']} MiB "
                                 f"| RSS {r['peak_rss_mib']} MiB swap+{r['swap_growth_mib']} "
                                 f"| temp {r['gpu_temp_start']}->{r['gpu_temp_max']}C clk avg {r['gpu_sm_clock_avg']} MHz "
                                 f"throttled {r['thermal_throttle_frac']:.0%} (waited {r['cool_wait_s']}s) | score {r['score']:.2f} | {r['output'][:70]!r}"
                                 if r["status"] == "ok" else json.dumps({k: v for k, v in r.items() if k != 'output'})),
                              flush=True)
                        if r.get("fatal"):
                            print("[run] fatal GPU error - exiting so run_exp.sh can restart and resume", flush=True)
                            f.flush(); os.fsync(f.fileno())
                            os._exit(3)     # normal exit hung after both main-run errors (CUDA teardown); skip it


if __name__ == "__main__":
    main()
