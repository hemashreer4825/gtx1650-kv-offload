"""Check 5: download Qwen/Qwen3-1.7B and compute KV-cache size per token from its config."""
import json, os
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-1.7B", allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
print("snapshot:", p)
for f in sorted(os.listdir(p)):
    print(f"  {f:40s} {os.path.getsize(os.path.join(p, f))/1e6:9.1f} MB")
cfg = json.load(open(os.path.join(p, "config.json")))
keys = ["architectures", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size",
        "intermediate_size", "max_position_embeddings", "rope_scaling", "rope_theta", "torch_dtype", "tie_word_embeddings",
        "vocab_size", "sliding_window", "use_sliding_window"]
for k in keys:
    print(f"  {k:26s} {cfg.get(k)}")
L, kvh, hd = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
per_tok = 2 * L * kvh * hd * 2  # K and V, fp16 = 2 bytes
print(f"\nKV bytes per token (fp16) = 2(K,V) x {L} layers x {kvh} kv-heads x {hd} head_dim x 2 bytes = {per_tok} B = {per_tok/1024:.0f} KiB")
for n in (4096, 8192, 16384, 24576, 32768):
    print(f"  {n:6d} tokens -> {n*per_tok/2**30:5.2f} GiB")
