"""Correct, RAM-exact offloaded KV cache for experiment 2 (replaces transformers' DynamicCache(offloading=True)).

Why (checks 10c, 10g): transformers 5.2's built-in offloading (a) gives wrong tokens because a non-blocking
GPU->CPU copy can be read by the next prefetch before it finishes (forcing a sync makes outputs identical to
GPU-only), and (b) holds 2-3.5x the KV size in pinned RAM because it allocates new pinned buffers every step.

Design:
  * One pinned CPU buffer for the whole KV cache, allocated once at its exact size (normal RAM, then
    cudaHostRegister - avoids the power-of-two rounding of PyTorch's pinned-memory allocator).
  * Token-major layout [layer][token][K/V][head][dim]: a layer's first n tokens are ONE contiguous block, so the
    CPU->GPU copy is a single DMA transfer (check 11: the head-major layout made PyTorch repack on the CPU).
  * Each step a layer's NEW tokens are copied GPU->CPU (only the new tokens), and the layer's old tokens are
    copied CPU->GPU into one reusable GPU staging buffer for attention.
  * Every copy and every kernel is on the same CUDA stream, so a read can never overtake a write (correct by
    construction). No prefetch overlap: transfer time is fully visible.
"""
import torch
from transformers.cache_utils import Cache, DynamicLayer


class _Store:
    def __init__(self, n_layers, n_kv_heads, head_dim, max_len, dtype=torch.float16):
        self.cpu = torch.empty((n_layers, max_len, 2, n_kv_heads, head_dim), dtype=dtype)   # exact size
        self.nbytes = self.cpu.numel() * self.cpu.element_size()
        err = torch.cuda.cudart().cudaHostRegister(self.cpu.data_ptr(), self.nbytes, 0)       # pin in place
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister failed with code {err}")
        self.gpu = torch.empty((max_len, 2, n_kv_heads, head_dim), dtype=dtype, device="cuda")  # staging

    def close(self):
        if self.cpu is not None:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(self.cpu.data_ptr())
            self.cpu = None; self.gpu = None


def _as_bhsd(t):            # [n, H, D] (strided view) -> [1, H, n, D]
    return t.permute(1, 0, 2).unsqueeze(0)


class PinnedOffloadLayer(DynamicLayer):
    def __init__(self, store, idx):
        super().__init__()
        self.store, self.idx, self.n = store, idx, 0

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n, q = self.n, key_states.shape[-2]
        cpu = self.store.cpu[self.idx]                                    # [max_len, 2, H, D]
        if n + q > cpu.shape[0]:
            raise RuntimeError(f"offload cache full: {n + q} > max_len {cpu.shape[0]}")
        new = torch.stack((key_states[0], value_states[0]), 0).permute(2, 0, 1, 3).contiguous()  # [q, 2, H, D]
        cpu[n:n + q].copy_(new, non_blocking=True)                         # new tokens -> CPU (contiguous)
        self.n = n + q
        if n == 0:
            return key_states, value_states
        g = self.store.gpu
        g[:n].copy_(cpu[:n], non_blocking=True)                            # old tokens -> GPU, one block
        g[n:n + q].copy_(new)
        return _as_bhsd(g[:n + q, 0]), _as_bhsd(g[:n + q, 1])

    def get_seq_length(self):
        return self.n

    def get_max_cache_shape(self):
        return self.store.cpu.shape[1] if self.store.cpu is not None else -1


class PinnedOffloadCache(Cache):
    def __init__(self, config, max_len):
        c = config.get_text_config(decoder=True)
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.store = _Store(c.num_hidden_layers, c.num_key_value_heads, hd, max_len)
        super().__init__(layers=[PinnedOffloadLayer(self.store, i) for i in range(c.num_hidden_layers)])

    def close(self):
        self.store.close()
