# dp_main Branch vs cef4a18 Orphan Commit: Detailed Code Change Report

**Date**: 2026-04-24
**Cluster**: a6e-wangez-16, us-east5-a (v6e-16, 4 workers x 4 chips)
**Model**: MiMo-V2-Flash (256 routed experts, top-8, FP8, 48 layers)
**Benchmark Config**: dp=4, tp=4, input=16384, output=1024, 256 prompts, rr=100

## What Are dp_main and cef4a18?

These are **two separate implementations of Data Parallelism (DP)** in sglang-jax. They are NOT the same branch or lineage:

- **`cef4a18`** (full hash: `cef4a181508ef22b452a38fe4210478e2e3672b1`) is an **orphan commit** — it does not sit on any branch. Its commit message is "remove 'if jax.process_count() == 1' (#245)". It was independently developed and tested as a standalone working DP implementation. This is the version that produces correct multi-host results.

- **`dp_main`** (`fork/dp_main` branch) was created by **cherry-picking PR #213** (Data Parallelism) onto `fork/main` at commit `6bed15ff`. The cherry-pick produced merge conflicts across 43+ files, and the conflict resolution introduced numerous bugs (documented below).

```
fork/main (6bed15ff)  ──  latest upstream main
  │
  └── fork/dp_main branch:  cherry-pick PR #213 → broken merge conflict resolutions
                             + 3 fix commits from previous debug sessions

cef4a18 (orphan commit):  standalone DP implementation → works on multi-host
```

They share the same DP *design* (single-controller JAX SPMD with mesh (dp, tp)), but dp_main's code is broken for multi-host due to the cherry-pick conflicts.

## Executive Summary

The `dp_main` branch contains **pervasive multi-host incompatibilities** that prevent it from running on multi-node TPU clusters. The orphan commit `cef4a18` is the known-working DP implementation.

**43 files** differ between cef4a18 and dp_main, with **4,200 insertions and 1,017 deletions**. The changes fall into several categories, detailed below.

### Benchmark Result (cef4a18 — the working version)

| Metric | Value |
|--------|-------|
| Output throughput | 513.15 tok/s |
| Peak decode throughput | ~2,315 tok/s |
| Total token throughput | 8,723.60 tok/s |
| DP distribution | Balanced [17,17,17,17] |
| Full KV usage at peak | 95-100% |
| Requests completed | 256/256 |
| Duration | 510.85s |
| Mean TPOT | 74.68ms |

---

## Category 1: Sharding Guard Bug (CRITICAL)

**Pattern**: `NamedSharding(mesh, P(...)) if jax.process_count() == 1 else None`

This is the single most pervasive and damaging pattern in dp_main. When `jax.process_count() > 1` (any multi-host setup), sharding is set to `None`, which causes:
1. `jax.device_put` to use `assert_equal` checks across hosts
2. TPU "unexpected peer with different launch id" fatal errors
3. Process termination cascading across all workers

**cef4a18** always uses `NamedSharding(...)` unconditionally, which is correct for both single-host and multi-host.

### Affected Files

#### 1.1 `flashattention_backend.py` — 3 instances

```python
# dp_main (BROKEN):
sharding=(NamedSharding(self.mesh, P("data")) if jax.process_count() == 1 else None),

# cef4a18 (CORRECT):
sharding=(NamedSharding(self.mesh, P("data"))),
```

**Locations**: Lines 235, 348, 458 (in `get_forward_metadata`, `get_forward_metadata_non_hybrid`, and mixed-chunk metadata construction).

#### 1.2 `forward_batch_info.py` — 5 instances

```python
# dp_main (BROKEN):
sharding=(
    NamedSharding(model_runner.mesh, PartitionSpec("data"))
    if jax.process_count() == 1
    else None
),

# cef4a18 (CORRECT):
sharding=(NamedSharding(model_runner.mesh, PartitionSpec("data"))),
```

**Locations**: Lines 334, 345, 352, 374, 391. Affects: `input_ids`/`positions`/`extend_prefix_lens`/`extend_seq_lens` array, `mrope_positions`, `input_embedding`, `lora_scalings`/`lora_token_indices`/`lora_ranks`, `deepstack_visual_embedding`.

#### 1.3 `sampling_batch_info.py` — 2 instances

```python
# dp_main (BROKEN):
sharding = NamedSharding(mesh, PartitionSpec("data")) if jax.process_count() == 1 else None
linear_penalty_sharding = (
    NamedSharding(mesh, PartitionSpec("data", "tensor"))
    if jax.process_count() == 1
    else None
)

# cef4a18 (CORRECT):
sharding = NamedSharding(mesh, PartitionSpec("data"))
linear_penalty_sharding = NamedSharding(mesh, PartitionSpec("data", "tensor"))
```

#### 1.4 `logits_processor.py` — 1 instance

```python
# dp_main (BROKEN):
sharding = NamedSharding(mesh, P("data")) if jax.process_count() == 1 else None

# cef4a18 (CORRECT):
sharding = NamedSharding(mesh, P("data"))
```

**Total**: 11 instances of the sharding guard bug across 4 files.

---

## Category 2: `device_array()` Implementation Change

### `jax_utils.py`

dp_main reverted `device_array()` to always use `jax.device_put`, removing the `make_array_from_callback` path that cef4a18 uses for multi-host array construction.

```python
# cef4a18 (CORRECT):
def device_array(*data, sharding=None, **kwargs) -> jax.Array:
    if sharding is None:
        return jax.device_put(*data, device=sharding, **kwargs)

    def _to_device(arr):
        arr = np.asarray(arr)
        return jax.make_array_from_callback(arr.shape, sharding, lambda idx, a=arr: a[idx])

    return jax.tree.map(_to_device, *data)

# dp_main (BROKEN):
def device_array(*data, sharding=None, **kwargs) -> jax.Array:
    return jax.device_put(*data, device=sharding, **kwargs)
```

**Impact**: With sharding=None (due to the guard bug), `jax.device_put` triggers `assert_equal` multi-host checks that cause the TPU crash. Even if sharding were provided, the `make_array_from_callback` path in cef4a18 is the correct approach for multi-host because it constructs the array from a local callback without requiring cross-host synchronization.

---

## Category 3: Attention Backend (`flashattention_backend.py`)

Beyond the sharding guards, dp_main has extensive logic differences in the attention forward metadata construction.

### 3.1 SWA Page Index Computation

```python
# cef4a18 (CORRECT) — vectorized 2D operations:
cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
strided_2d = cache_loc_2d[:, :: self.page_size]
page_indices = (strided_2d // self.page_size).ravel()

# SWA: per-rank mapping
for i in range(batch.dp_size):
    mapping = swa_mapping[i] if isinstance(swa_mapping, list) else swa_mapping
    swa_strided[i] = mapping[strided_2d[i]]
swa_page_indices = (swa_strided // self.page_size).ravel()

# dp_main (BROKEN) — scalar loop + flat mapping:
for i in range(batch.dp_size):
    rank_cache_loc = batch.cache_loc[start:end]
    # ... pad, stride, convert per rank
page_indices = np.concatenate(page_indices_list)

# SWA: flat mapping (ignores DP rank)
indices = np.arange(0, len(batch.cache_loc), self.page_size)
swa_slots = swa_mapping[batch.cache_loc[indices]]  # BUG: no per-rank mapping
swa_page_indices = (swa_slots // self.page_size).astype(np.int32)
```

**Bug**: dp_main's SWA mapping uses `swa_mapping` as a flat array and indexes into it without respecting DP rank boundaries. cef4a18 correctly handles per-DP-rank mappings via `swa_mapping[i]`.

### 3.2 Decode Distribution

```python
# cef4a18 (CORRECT):
if batch.forward_mode == ForwardMode.DECODE:
    distribution = np.repeat(local_num_seqs, 3)

# dp_main (BROKEN):
if batch.forward_mode == ForwardMode.DECODE:
    dist = np.array([0, 0, local_num_seqs], dtype=np.int32)
```

dp_main sets decode distribution to `[0, 0, N]` (mixed/generic path) while cef4a18 uses `[N, N, N]` (pure decode path). This forces all decode batches through the slower mixed kernel path.

### 3.3 cu_q_lens / cu_kv_lens

cef4a18 uses vectorized 2D `np.cumsum` with reshape; dp_main uses scalar Python loops with `np.concatenate`. Functionally equivalent but dp_main is slower.

### 3.4 Removed Mesh/Partition Axis from Pytree

dp_main removes `kv_partition_axis`, `attention_data_partition_axis`, and `mesh` from the FlashAttention pytree `tree_flatten`/`tree_unflatten`. This breaks JAX's ability to reconstruct the attention backend after JIT boundary crossings.

### 3.5 Unsupported Kernel Arguments

```python
# dp_main (BROKEN) — adds unsupported kwarg, removes required one:
    causal=causal,
    decode_mode=decode_mode,  # NOT supported by installed kernel
    sm_scale=scale,
    ...
    # MISSING: vmem_limit_bytes=self.vmem_limit_bytes

# cef4a18 (CORRECT):
    causal=causal,
    sm_scale=scale,
    ...
    vmem_limit_bytes=self.vmem_limit_bytes,
```

`decode_mode` is not accepted by the installed `ragged_paged_attention` kernel and raises `TypeError`. `vmem_limit_bytes` is required for correct memory management.

---

## Category 4: Schedule Batch (`schedule_batch.py`)

### 4.1 `maybe_evict_swa()` — Uses `self.reqs` Instead of `self.reqs_info`

```python
# cef4a18 (CORRECT) — iterates per DP rank:
def maybe_evict_swa(self):
    ...
    for dp_rank, info in enumerate(self.reqs_info):
        if not info.reqs:
            continue
        for req in info.reqs:
            ...
            self._evict_swa(req, ..., dp_rank)

# dp_main (BROKEN) — flat iteration:
def maybe_evict_swa(self):
    ...
    for req in self.reqs:  # AttributeError: 'ScheduleBatch' has no attribute 'reqs'
        ...
```

**Bug**: dp_main's `maybe_evict_swa` references `self.reqs` which doesn't exist in the DP-aware `ScheduleBatch`. cef4a18 correctly uses `self.reqs_info` (list of per-DP-rank request info).

### 4.2 `_evict_swa()` — Missing `dp_rank` Parameter

```python
# cef4a18 (CORRECT):
def _evict_swa(self, req, pre_len, sliding_window_size, page_size, dp_rank=0):
    ...
    self.token_to_kv_pool_allocator.free_swa(free_slots, dp_rank=dp_rank)

# dp_main (BROKEN):
def _evict_swa(self, req, pre_len, sliding_window_size, page_size):
    ...
    self.token_to_kv_pool_allocator.free_swa(free_slots)  # no dp_rank
```

### 4.3 `retract_decode()` Return Value Change

```python
# cef4a18: returns (retracted_reqs, new_estimate_ratio)
# dp_main: returns (retracted_reqs, new_estimate_ratio, reqs_to_abort)
```

dp_main adds abort handling (graceful OOM abort from PR #944), but `scheduler.py` in cef4a18 expects 2-tuple, not 3-tuple.

### 4.4 SWA Eviction Ordering

cef4a18 performs SWA eviction inside the per-DP-rank loop in `prepare_for_extend`, with dp_rank passed to `_evict_swa`. dp_main moves eviction outside the loop to `maybe_evict_swa()` but calls it on `self.reqs` (broken) and without dp_rank (broken).

---

## Category 5: Scheduler (`scheduler.py`)

### 5.1 JAX Distributed Init Guard

```python
# cef4a18 (CORRECT):
if not jax.distributed.is_initialized():
    jax.distributed.initialize(server_args.dist_init_addr, self.nnodes, self.node_rank)
else:
    logger.info("JAX distributed already initialized, skipping re-initialization")

# dp_main (BROKEN):
jax.distributed.initialize(server_args.dist_init_addr, self.nnodes, self.node_rank)
```

dp_main unconditionally calls `jax.distributed.initialize()`, which will crash if called twice (e.g., after a scheduler restart).

### 5.2 Tree Cache Priority

```python
# cef4a18 — hybrid check first:
if self.is_hybrid:
    self.tree_cache = SWARadixCache(...)
elif server_args.chunked_prefill_size is not None and server_args.disable_radix_cache:
    self.tree_cache = ChunkCache(...)

# dp_main — chunked prefill check first:
if server_args.chunked_prefill_size is not None and server_args.disable_radix_cache:
    self.tree_cache = ChunkCache(...)
elif self.is_hybrid:
    self.tree_cache = SWARadixCache(...)
```

dp_main's ordering means hybrid SWA models with `--disable-radix-cache` get `ChunkCache` instead of `SWARadixCache`, losing SWA eviction tracking.

### 5.3 `process_allgather` After Sampling

```python
# dp_main (BROKEN) — adds process_allgather:
if self.dp_size > 1:
    from jax.experimental.multihost_utils import process_allgather
    next_token_ids_device = process_allgather(next_token_ids_device, tiled=True)

# cef4a18 — no allgather needed (single-controller design):
next_token_ids = np.array(jax.device_get(next_token_ids_device))
```

In cef4a18's single-controller architecture, the scheduler on worker 0 already has all token IDs via SPMD. The `process_allgather` is unnecessary and adds latency.

### 5.4 Empty Batch Guard

```python
# cef4a18 (CORRECT):
if batch.is_empty():
    return batch

# dp_main — missing this check
```

---

## Category 6: TP Worker (`tp_worker.py`)

### 6.1 `max_req_len` Computation

```python
# cef4a18 (CORRECT):
self.max_req_len = min(
    self.model_config.context_len - 1,
    self.max_total_num_tokens - 1,
)

# dp_main (BROKEN):
per_rank_tokens = (
    self.max_total_num_tokens // self.dp_size
    if self.dp_size > 1
    else self.max_total_num_tokens
)
self.max_req_len = min(
    self.model_config.context_len - 1,
    per_rank_tokens - 1,
)
```

dp_main divides by dp_size, which unnecessarily limits max request length. In cef4a18's architecture, the full pool is shared across DP ranks.

### 6.2 Attention Backend Reference

```python
# cef4a18 (CORRECT):
self.worker.model_runner.attn_backend.get_forward_metadata(model_worker_batch)

# dp_main (BROKEN):
self.model_runner.attn_backend.get_forward_metadata(model_worker_batch)
```

dp_main uses `self.model_runner` directly instead of `self.worker.model_runner`, which may reference a stale or incorrect model runner instance.

### 6.3 `get_tokens_per_layer_info`

```python
# cef4a18 (CORRECT):
return (
    self.model_runner.full_max_total_num_tokens,
    self.model_runner.swa_max_total_num_tokens,
)

# dp_main (BROKEN):
return (
    getattr(self.model_runner, "full_max_total_num_tokens", self.model_runner.max_total_num_tokens),
    getattr(self.model_runner, "swa_max_total_num_tokens", self.model_runner.max_total_num_tokens),
)
```

dp_main uses `getattr` fallbacks, which mask missing attributes and return incorrect values.

---

## Category 7: Model Runner (`model_runner.py`)

### 7.1 SWA Head Count — `tp_size` vs `attention_tp_size`

```python
# dp_main (BROKEN):
swa_head_num = max(swa_num_kv_heads, self.tp_size)    # max(8, 16) = 16

# cef4a18 (via our fix):
swa_head_num = max(swa_num_kv_heads, self.attention_tp_size)  # max(8, 4) = 8
```

Using `self.tp_size` (=16, global) instead of `self.attention_tp_size` (=4, per DP rank) doubled the SWA KV cache size, causing OOM on full_kv_pool allocation.

### 7.2 Pool Size Alignment

```python
# dp_main (BROKEN):
alignment = self.page_size  # Only page_size

# cef4a18 (via our fix):
alignment = self.page_size * dp_size  # page_size * dp_size
```

Missing dp_size in alignment causes pool sizes that aren't evenly shardable across DP ranks.

### 7.3 Sampler RNG Refactor

dp_main adds `fold_in(base_key, step)` RNG generation inside JIT for deterministic sampling across DP ranks:

```python
# dp_main (NEW):
self._sampler_base_rng = jax.random.PRNGKey(server_args.random_seed)
self._sampler_step = 0
...
rng_key = jax.random.fold_in(base_rng_key, rng_step)
return sampler(*args, rng_override=rng_key)
```

This is a legitimate improvement from PR #940 (`fix: advance sampler RNG each decode step`), not present in cef4a18.

### 7.4 `adjust_layer_num()` Placement

cef4a18 has `adjust_layer_num()` as a local function inside `profile_max_num_token()`. dp_main extracts it as a method of `ModelRunner`. The dp_main version also references `self.model_config.hf_config` while cef4a18 uses `self.model_config.hf_text_config` — either could be correct depending on the model config structure.

### 7.5 KV Pool Allocator Logic

cef4a18 checks `is_hybrid` first (before `page_size == 1`), ensuring hybrid models always get `SWATokenToKVPoolAllocator`. dp_main checks `page_size == 1` first, which could route hybrid models to the wrong allocator.

---

## Category 8: Memory Pool (`memory_pool.py`)

### 8.1 Fused KV Cache Shape (3D vs 5D)

cef4a18 uses a 5D fused KV cache buffer:
```
[total_pages, page_size, num_kv_heads*2//packing, packing, head_dim]
```

dp_main uses a 3D buffer in some paths:
```
[total_tokens, num_kv_heads*2, head_dim]
```

The `merge_kv()` function, `get_kv_buffer()`, and `set_kv_buffer_legacy()` all differ to accommodate the 5D format in cef4a18. The 5D format is required by the rpav3 kernel.

### 8.2 `SWAKVPool` — Per-DP-Rank Mapping

```python
# cef4a18 (CORRECT) — mapping can be a list (one per DP rank):
mapping = self.full_to_swa_index_mapping
mapping_children = tuple(mapping) if isinstance(mapping, list) else (mapping,)

# dp_main (BROKEN) — always scalar:
children = (self.swa_kv_pool, self.full_kv_pool, self.full_to_swa_index_mapping)
```

cef4a18 supports per-DP-rank SWA index mappings (list of arrays), while dp_main assumes a single mapping.

### 8.3 Page Count Rounding

```python
# cef4a18 (via our fix):
total_pages = ((total_pages + self.dp_size - 1) // self.dp_size) * self.dp_size

# dp_main — no dp_size rounding
```

---

## Category 9: Other Notable Differences

### 9.1 `sampler.py` — `rng_override` Parameter

dp_main adds `rng_override: jax.Array | None = None` to `Sampler.__call__`, used by the `fold_in` RNG approach. This is a legitimate enhancement.

### 9.2 `mimo_v2_flash.py` — 204 lines changed

Significant model code differences (likely from ongoing development on main), but not directly related to DP bugs.

### 9.3 `fused_moe.py` / `moe.py` — 197 lines changed

Expert-parallel MoE changes, likely from the EPMoE backend evolution.

### 9.4 `allocator.py` — 200 lines changed

Memory allocator refactoring with different free/alloc patterns between the two versions.

---

## Summary of Breaking Changes

| # | File | Bug Type | Severity | Instances |
|---|------|----------|----------|-----------|
| 1 | flashattention_backend.py | Sharding guard + SWA mapping + decode dist + missing vmem_limit + unsupported decode_mode | CRITICAL | 5+ |
| 2 | forward_batch_info.py | Sharding guard | CRITICAL | 5 |
| 3 | sampling_batch_info.py | Sharding guard | CRITICAL | 2 |
| 4 | logits_processor.py | Sharding guard | CRITICAL | 1 |
| 5 | jax_utils.py | device_array() implementation | CRITICAL | 1 |
| 6 | schedule_batch.py | self.reqs vs self.reqs_info + missing dp_rank | CRITICAL | 2 |
| 7 | scheduler.py | Missing init guard + wrong cache priority + extra allgather + missing empty check | HIGH | 4 |
| 8 | tp_worker.py | Wrong max_req_len + wrong attn_backend ref | HIGH | 3 |
| 9 | model_runner.py | tp_size vs attention_tp_size + missing alignment | HIGH | 2 |
| 10 | memory_pool.py | 3D vs 5D KV cache + scalar vs list mapping | HIGH | 2 |

## Root Cause

`dp_main` and `cef4a18` are two different implementations of DP. They are NOT on the same branch:

- `dp_main` was created by cherry-picking PR #213 onto `fork/main` at `6bed15ff`. `fork/main` had evolved significantly since the DP code was originally developed, producing merge conflicts across 43+ files. The conflict resolution introduced the `if jax.process_count() == 1 else None` pattern as a workaround, which fundamentally breaks multi-host operation.

- `cef4a18` is an orphan commit (not on any branch) that represents a correct, independently-developed DP implementation. It was tested and validated on multi-host TPU clusters.

The two share the same DP architecture (single-controller JAX SPMD), but dp_main's cherry-pick conflict resolutions corrupted the implementation at 11+ call sites.

## Recommendation

1. **Do not use dp_main for multi-host benchmarks.** Use cef4a18 directly.
2. **To create a proper DP branch**, rebase cef4a18's changes onto the current main, resolving conflicts correctly (always keeping `NamedSharding(...)` without guards).
3. **Cherry-pick legitimate improvements** from main that are missing in cef4a18:
   - Sampler RNG fix (PR #940)
   - Graceful OOM abort (PR #944)
   - Tuned MoE block configs
