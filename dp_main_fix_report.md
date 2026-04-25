# dp_main Branch Fix Report

## Background

The `fork/dp_main` branch was created by cherry-picking PR #213 (Data Parallelism) onto `fork/main`. During the merge/cherry-pick, conflict resolution introduced 11+ bugs that broke multi-host data parallelism. An orphan commit `cef4a18` ("remove 'if jax.process_count() == 1' (#245)") contained a known-working DP implementation.

**Goal**: Fix dp_main to work correctly with multi-host DP while preserving dp_main's legitimate upstream improvements (new models, kernels, sampler RNG, SWA abort handling, etc.).

## Strategy

We replaced 14 DP-critical files on dp_main with their versions from the known-working `cef4a18` commit. This was chosen over surgical edits because every dp_main change in these 14 files was a regression -- none contained improvements worth preserving.

## Commits

| Commit | Date | Files | Description |
|--------|------|-------|-------------|
| `2dc0b3af` | Apr 24 15:00 UTC | 11 files, 509+/530- | Fix multi-host DP: restore cef4a18 DP-critical code |
| `ccc88f79` | Apr 24 17:43 UTC | 3 files, 17+/13- | Fix cache compatibility: restore chunk_cache, common, swa_radix_cache |

## Bugs Fixed (14 files, 11 categories)

### 1. Sharding Guards (11 instances across 4 files)

**Pattern**: `NamedSharding(mesh, spec) if jax.process_count() == 1 else None`

This pattern disables sharding on multi-host setups (where `process_count() > 1`), causing JAX to place arrays on a single host instead of distributing them across the mesh. Every multi-host DP run would crash or produce incorrect results.

**Files affected**:
- `forward_batch_info.py` (5 instances) -- batch metadata sharding for seq_lens, position_ids, etc.
- `sampling_batch_info.py` (2 instances) -- sampling parameter sharding
- `logits_processor.py` (1 instance) -- logits output sharding
- `flashattention_backend.py` (3 instances) -- `device_array()` sharding kwargs

**Fix**: Always use `NamedSharding(mesh, spec)` regardless of process count. Multi-host JAX SPMD requires explicit sharding annotations.

### 2. `device_array()` Missing `make_array_from_callback` Path

**File**: `jax_utils.py`

dp_main used only `jax.device_put(data, device=sharding)` which doesn't work for multi-host sharded arrays. The working implementation needs `jax.make_array_from_callback()` when a sharding is provided, which correctly creates a distributed array by calling back for each shard's data slice.

**Fix**:
```python
def device_array(*data, sharding=None, **kwargs):
    if sharding is None:
        return jax.device_put(*data, device=sharding, **kwargs)
    def _to_device(arr):
        arr = np.asarray(arr)
        return jax.make_array_from_callback(arr.shape, sharding, lambda idx, a=arr: a[idx])
    return jax.tree.map(_to_device, *data)
```

### 3. Flash Attention Backend DP Logic (6 sub-issues)

**File**: `flashattention_backend.py`

**3a. Page indices / cu_q_lens / cu_kv_lens computation**: dp_main used per-rank Python loops instead of vectorized 2D operations. The working code reshapes these into `[dp_size, ...]` arrays and computes them in a vectorized manner.

**3b. Decode distribution**: dp_main used `[0, 0, N]` distribution (all decode tokens on last DP rank). The fix uses `np.repeat(local_num_seqs, 3)` to give `[N, N, N]` -- equal distribution across all 3 sections (prefill padding, decode, SWA decode).

**3c. SWA page indices mapping**: dp_main didn't correctly handle per-DP-rank SWA token-to-KV mappings. The fix applies the correct mapping for each DP rank's strided indices.

**3d. Pytree fields**: dp_main dropped `kv_partition_axis`, `attention_data_partition_axis`, and `mesh` from `tree_flatten`/`tree_unflatten`, breaking JAX's ability to trace through the attention data structure.

**3e. Kernel call arguments**: dp_main passed `decode_mode=decode_mode` (not a valid parameter) and omitted `vmem_limit_bytes`. The fix removes the invalid arg and adds the required VMEM limit.

**3f. SWA layer detection**: dp_main used `token_to_kv_pool.layers_mapping` (doesn't exist). The fix uses `layer.sliding_window_size` to detect SWA vs full attention layers.

### 4. Schedule Batch SWA Eviction

**File**: `schedule_batch.py`

dp_main's `maybe_evict_swa` / `_evict_swa` used `self.reqs` (a list) instead of `self.reqs_info` (a dict with per-request metadata needed for eviction). It also dropped the `dp_rank` parameter required for DP-aware SWA cache eviction.

**Fix**: Restore the iteration over `self.reqs_info` and pass `dp_rank` to eviction functions.

### 5. Scheduler Issues (4 sub-issues)

**File**: `scheduler.py`

**5a. JAX distributed init guard**: dp_main unconditionally called `jax.distributed.initialize()`, crashing if already initialized. Fix: check `jax.distributed.is_initialized()` first.

**5b. Tree cache priority**: dp_main checked `chunked_prefill_size` before `is_hybrid`. For hybrid SWA models, the `is_hybrid` check must come first to select `SWARadixCache`.

**5c. Unnecessary `process_allgather`**: dp_main added a `process_allgather` after sampling that cef4a18 doesn't need (single-controller DP handles this implicitly).

**5d. Empty batch guard**: dp_main was missing a guard before `prepare_for_decode()`, which could crash on empty decode batches.

### 6. TP Worker Issues

**File**: `tp_worker.py`

**6a. `max_req_len`**: dp_main used `per_rank_tokens - 1` (tokens per DP rank). The correct value is `self.max_total_num_tokens - 1` (global max across all ranks).

**6b. `attn_backend` reference**: dp_main used `self.attn_backend` (doesn't exist on worker). Fix: `self.worker.model_runner.attn_backend`.

**6c. `get_tokens_per_layer_info`**: dp_main used a `getattr` fallback pattern. Fix: direct attribute access matching the actual API.

### 7. Model Runner Issues

**File**: `model_runner.py`

**7a. `swa_head_num`**: dp_main used `self.tp_size` to compute SWA head count. The correct divisor is `self.attention_tp_size` (which accounts for DP splitting: `tp_size // dp_size`).

**7b. Memory alignment**: dp_main aligned KV cache pages to `self.page_size`. The fix aligns to `self.page_size * dp_size` so pages distribute evenly across DP ranks.

**7c. Allocator selection**: dp_main checked `page_size == 1` before `is_hybrid`. For hybrid models, `is_hybrid` must take priority to select `SWATokenToKVPoolAllocator`.

### 8. Memory Pool Pytree

**File**: `memory_pool.py`

dp_main dropped `dp_size` and `attention_data_partition_axis` from the pytree `aux_data` in `tree_flatten`/`tree_unflatten`. These are needed for JAX tracing to correctly reconstruct the memory pool with DP-aware sharding.

Also fixed page count rounding to be divisible by `dp_size`.

### 9. Allocator dp_rank

**File**: `allocator.py`

dp_main's `free_swa()` dropped the `dp_rank` parameter, making SWA cache deallocation DP-unaware. The fix restores the parameter so each DP rank manages its own SWA allocation independently.

### 10. Cache Compatibility (follow-up commit)

**Files**: `chunk_cache.py`, `common.py`, `swa_radix_cache.py`

After fixing schedule_batch.py (which calls `full_lru_list_evictable_size` and other methods during memory pressure), dp_main's cache classes lacked these methods, causing `AttributeError` at runtime.

**Fix**: Restore `evict()` signature in chunk_cache (with `.copy()` on `prefix_indices`), `full_lru_list_evictable_size` references in common.py, and `evictable_size` / `adjust_swa_protected_size` methods in swa_radix_cache.py.

## Verification

After applying both commits:
- dp=4 long context on v6e-16 achieved **587 tok/s** (14.4% better than cef4a18's 513 tok/s)
- dp=1/2/4 all operational on v6e-16 (16 chips)
- dp=1/2/4/8 all operational on v6e-32 (32 chips)
- Server starts, serves requests, and benchmarks complete successfully
