"""Summarise results/exp/main.jsonl -> results/exp/main_summary.txt and main_summary.csv (one row per method x length).
Only status "ok" rows enter averages; "error" rows (redone after restart) are ignored; oom/skipped are reported."""
import json, csv, collections, statistics as st, os, sys
src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/kv-offload/results/exp/main.jsonl")
rows = [json.loads(l) for l in open(src)]
M = ["gpu", "offload", "stream_1024", "stream_2048", "snapkv_1024", "snapkv_2048"]
Ls = sorted({r["length"] for r in rows}); T = ["niah_single_2", "niah_multikey_1", "niah_multivalue", "vt"]
ok = [r for r in rows if r["status"] == "ok"]
out = []
def p(s=""): out.append(s); print(s)
p(f"source: {src} | records {len(rows)} | " + str(dict(collections.Counter(r['status'] for r in rows))))
p("errors (redone after restart): " + "; ".join(f"{r['time']} {r['length']} {r['task']} {r['method']}: {r['error']}" for r in rows if r["status"] == "error"))
p("\n== RULER score (string_match_all, 0-100), per task, 1 sample each ==")
p(f"{'length':>6} {'method':12} " + " ".join(f"{t:>16}" for t in T) + f" {'mean':>6}")
table = []
for L in Ls:
    for m in M:
        sc = []; cells = []
        for t in T:
            rr = [r for r in rows if r["length"] == L and r["method"] == m and r["task"] == t and r["status"] != "error"]
            if not rr: cells.append("-"); continue
            r = rr[-1]
            if r["status"] == "ok": sc.append(100 * r["score"]); cells.append(f"{100*r['score']:.0f}")
            else: cells.append(r["status"].replace("skipped_after_failure", "skip"))
        g = [r for r in ok if r["length"] == L and r["method"] == m]
        mean = st.mean(sc) if sc else None
        p(f"{L:6d} {m:12} " + " ".join(f"{c:>16}" for c in cells) + f" {'-' if mean is None else round(mean, 1):>6}")
        if g:
            avg = lambda k: round(st.mean(r[k] for r in g if r.get(k) is not None), 2)
            table.append(dict(length=L, method=m, n=len(g), score_mean=round(mean, 1), ctx_tokens=avg("ctx_tokens"), kv_kept=avg("kv_kept"),
                              prefill_s=avg("prefill_s"), ttft_s=avg("ttft_s"), decode_tok_s=avg("decode_tok_s"),
                              decode_tok_s_min=min(r["decode_tok_s"] for r in g if r["decode_tok_s"]),
                              decode_tok_s_max=max(r["decode_tok_s"] for r in g if r["decode_tok_s"]),
                              peak_gpu_alloc_mib=max(r["peak_gpu_alloc_mib"] for r in g), peak_gpu_reserved_mib=max(r["peak_gpu_reserved_mib"] for r in g),
                              peak_rss_mib=max(r["peak_rss_mib"] for r in g), min_avail_ram_mib=min(r["min_avail_ram_mib"] for r in g),
                              swap_growth_mib_max=max(r["swap_growth_mib"] for r in g), sm_clock_avg_mhz=avg("gpu_sm_clock_avg"),
                              gpu_temp_max=max(r["gpu_temp_max"] for r in g)))
        else:
            st_ = [r["status"] for r in rows if r["length"] == L and r["method"] == m and r["status"] != "error"]
            table.append(dict(length=L, method=m, n=0, status=",".join(sorted(set(st_)))))
p("\n== speed / memory (mean over the 4 tasks; peaks = max) ==")
p(f"{'length':>6} {'method':12} {'n':>2} {'ctx':>6} {'kept':>6} {'prefill s':>9} {'TTFT s':>7} {'decode tok/s (min-max)':>23} {'peak GPU MiB':>12} {'peak RSS MiB':>12} {'swap+':>5} {'clk MHz':>7} {'Tmax':>4}")
for d in table:
    if d["n"] == 0: p(f"{d['length']:6d} {d['method']:12} {0:>2}  {d['status']}"); continue
    p(f"{d['length']:6d} {d['method']:12} {d['n']:2d} {d['ctx_tokens']:6.0f} {d['kv_kept']:6.0f} {d['prefill_s']:9.1f} {d['ttft_s']:7.1f} "
      f"{d['decode_tok_s']:7.2f} ({d['decode_tok_s_min']:.2f}-{d['decode_tok_s_max']:.2f}){'':>3} {d['peak_gpu_alloc_mib']:12d} {d['peak_rss_mib']:12d} {d['swap_growth_mib_max']:5d} {d['sm_clock_avg_mhz']:7.0f} {d['gpu_temp_max']:4d}")
base = os.path.splitext(src)[0]
open(base + "_summary.txt", "w").write("\n".join(out) + "\n")
keys = sorted({k for d in table for k in d}, key=lambda k: list(table[0].keys()).index(k) if k in table[0] else 99)
with open(base + "_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(table)
print(f"\nwritten {base}_summary.txt and {base}_summary.csv")
