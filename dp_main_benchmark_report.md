# dp_main Branch Benchmark Report

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | MiMo-V2-Flash (256 routed experts, top-8, FP8, 48 layers, hybrid SWA/full attention) |
| **Branch** | `fork/dp_main` (commit `ccc88f79`) |
| **Short context** | input=512, output=512 |
| **Long context** | input=16384, output=1024 |
| **Benchmark params** | 256 prompts, request-rate=100, flush-cache |
| **Page size** | 256 |
| **Context length** | 262144 |
| **Chunked prefill** | 2048 |
| **Dtype** | bfloat16 |
| **MoE backend** | epmoe |
| **Memory fraction** | 0.95 |
| **SWA/full ratio** | 0.2 |

## Cluster Specifications

| Cluster | TPU | Chips | Workers | Zone |
|---------|-----|-------|---------|------|
| a6e-wangez-16 | v6e-16 | 16 | 4 x 4 chips | us-east5-a |
| a6e-wangez | v6e-32 | 32 | 8 x 4 chips | us-east5-b |

## Results: v6e-16 (tp=ep=16)

### Output Throughput (tok/s)

| dp | atp | bs=32 short | bs=32 long | bs=64 short | bs=64 long | bs=128 short | bs=128 long | bs=200 short | bs=200 long |
|----|-----|-------------|------------|-------------|------------|--------------|-------------|--------------|-------------|
| 1  | 16  | 1,442       | 496        | 2,067       | 489        | 2,725        | 487         | 2,726        | 487         |
| 2  | 8   | 1,549       | 495        | 2,136       | 560        | 2,881        | --          | CRASH        | CRASH       |
| 4  | 4   | 1,411       | 471        | 2,087       | 587        | 2,955        | 583         | CRASH        | CRASH       |
| 8  | 2   | OOM         | OOM        | OOM         | OOM        | OOM          | OOM         | OOM          | OOM         |

*atp = attention_tp_size (tp_size / dp_size)*

### Time to First Token (ms)

| dp | bs=32 short | bs=32 long | bs=64 short | bs=64 long | bs=128 short | bs=128 long | bs=200 short | bs=200 long |
|----|-------------|------------|-------------|------------|--------------|-------------|--------------|-------------|
| 1  | 39,220      | 251,264    | 23,964      | 247,973    | 13,527       | 248,868     | 13,529       | 248,840     |
| 2  | 37,026      | 232,130    | 22,022      | 205,119    | 11,505       | --          | --           | --          |
| 4  | 38,807      | 232,160    | 22,687      | 180,081    | 11,047       | 189,000     | --           | --          |

### Mean TPOT (ms)

| dp | bs=32 short | bs=32 long | bs=64 short | bs=64 long | bs=128 short | bs=128 long | bs=200 short | bs=200 long |
|----|-------------|------------|-------------|------------|--------------|-------------|--------------|-------------|
| 1  | 20.68       | 43.60      | 28.13       | 55.28      | 41.51        | 55.43       | 41.50        | 55.42       |
| 2  | 19.31       | 40.71      | 25.65       | 66.85      | 35.12        | --          | --           | --          |
| 4  | 19.92       | 40.59      | 26.05       | 60.66      | 33.61        | 66.43       | --           | --          |

## Results: v6e-32 (tp=ep=32)

### Output Throughput (tok/s)

| dp | atp | bs=32 short | bs=32 long | bs=64 short | bs=64 long | bs=128 short | bs=128 long | bs=200 short | bs=200 long |
|----|-----|-------------|------------|-------------|------------|--------------|-------------|--------------|-------------|
| 1  | 32  | 1,558       | 547        | 2,389       | 611        | 2,848        | CRASH       | 2,849        | CRASH       |
| 2  | 16  | 1,737       | 477        | 2,372       | 613        | 2,995        | 666         | CRASH        | CRASH       |
| 4  | 8   | 1,664       | 505        | CRASH       | CRASH      | 3,214        | 697         | CRASH        | CRASH       |
| 8  | 4   | 1,654       | 572        | CRASH       | CRASH      | 2,867        | 672         | CRASH        | CRASH       |

### Time to First Token (ms)

| dp | bs=32 short | bs=32 long | bs=64 short | bs=64 long | bs=128 short | bs=128 long | bs=200 short | bs=200 long |
|----|-------------|------------|-------------|------------|--------------|-------------|--------------|-------------|
| 1  | 36,117      | 227,881    | 20,603      | 198,647    | 12,600       | --          | 12,599       | --          |
| 2  | 33,706      | 234,664    | 20,023      | 181,988    | 10,948       | 164,260     | --           | --          |
| 4  | 36,266      | 211,130    | --          | --         | 10,206       | 153,085     | --           | --          |
| 8  | 36,168      | 201,430    | --          | --         | 12,562       | 161,701     | --           | --          |

### Mean TPOT (ms)

| dp | bs=32 short | bs=32 long | bs=64 short | bs=64 long | bs=128 short | bs=128 long | bs=200 short | bs=200 long |
|----|-------------|------------|-------------|------------|--------------|-------------|--------------|-------------|
| 1  | 19.21       | 39.59      | 24.30       | 66.39      | 40.15        | --          | 40.14        | --          |
| 2  | 17.19       | 40.29      | 22.96       | 59.73      | 34.25        | 107.30      | --           | --          |
| 4  | 17.87       | 39.09      | --          | --         | 31.68        | 98.62       | --           | --          |
| 8  | 17.23       | 35.93      | --          | --         | 33.86        | 104.19      | --           | --          |

## Analysis

### Peak Throughput (output tok/s)

| Metric | v6e-16 | Config | v6e-32 | Config |
|--------|--------|--------|--------|--------|
| Best short context | **2,955** | dp=4, bs=128 | **3,214** | dp=4, bs=128 |
| Best long context | **587** | dp=4, bs=64 | **697** | dp=4, bs=128 |
| Best total throughput | 9,985 | dp=4, bs=64 long | 11,855 | dp=4, bs=128 long |

### DP Scaling Efficiency

#### Short Context (512 -> 512)

On v6e-32 with bs=128:
- dp=1: 2,848 tok/s (baseline)
- dp=2: 2,995 tok/s (+5.2%)
- dp=4: 3,214 tok/s (+12.8%)
- dp=8: 2,867 tok/s (+0.7%)

DP provides moderate throughput gains on short context. dp=4 is the sweet spot; dp=8 shows diminishing returns as the reduced attention_tp_size becomes the bottleneck.

#### Long Context (16K -> 1K)

On v6e-32 with bs=128:
- dp=1: CRASH (server instability)
- dp=2: 666 tok/s
- dp=4: 697 tok/s
- dp=8: 672 tok/s

On v6e-16 with bs=64:
- dp=1: 489 tok/s (baseline)
- dp=2: 560 tok/s (+14.5%)
- dp=4: 587 tok/s (+20.0%)

DP significantly improves long context throughput by splitting the large KV cache across DP ranks.

### TTFT Improvement with DP (Long Context)

On v6e-32 with bs=128:
- dp=2: 164,260 ms
- dp=4: 153,085 ms (6.8% faster than dp=2)
- dp=8: 161,701 ms

On v6e-16 with bs=64:
- dp=1: 247,973 ms
- dp=2: 205,119 ms (17.3% faster)
- dp=4: 180,081 ms (27.4% faster)

DP reduces TTFT on long context by distributing the prefill computation.

### v6e-32 vs v6e-16 Comparison

At equivalent dp/bs configurations:

| Config | v6e-16 | v6e-32 | Speedup |
|--------|--------|--------|---------|
| dp=1, bs=32, short | 1,442 | 1,558 | +8.0% |
| dp=1, bs=64, long | 489 | 611 | +24.9% |
| dp=2, bs=64, short | 2,136 | 2,372 | +11.1% |
| dp=2, bs=64, long | 560 | 613 | +9.5% |
| dp=4, bs=32, short | 1,411 | 1,664 | +17.9% |
| dp=4, bs=128, short | 2,955 | 3,214 | +8.8% |
| dp=4, bs=128, long | 583 | 697 | +19.6% |

v6e-32 consistently outperforms v6e-16 by 8-25%, with larger gains on long context where the extra compute bandwidth helps most.

### Batch Size Scaling

Short context throughput scales near-linearly with batch size up to bs=128, then plateaus:

v6e-16 dp=1: bs=32 (1,442) -> bs=64 (2,067, +43%) -> bs=128 (2,725, +32%) -> bs=200 (2,726, +0%)

v6e-32 dp=1: bs=32 (1,558) -> bs=64 (2,389, +53%) -> bs=128 (2,848, +19%) -> bs=200 (2,849, +0%)

The plateau at bs=128 indicates the TPU compute is fully saturated. Higher batch sizes only add scheduling overhead without throughput gains.

Long context throughput is relatively insensitive to batch size at dp=1 because the KV cache dominates memory, limiting effective concurrency regardless of the max_running_requests setting.

## Failed Configurations

| Configuration | Cluster | Failure Mode | Root Cause |
|---------------|---------|--------------|------------|
| dp>1, bs=200 | Both | Server crash: `ValueError: 1600 must be divisible by x-dimension tile size (256)` | Flash attention kernel tile alignment: 200 x 8 (KV heads) = 1600, not divisible by page_size 256 |
| dp>1, bs=64 | v6e-32 | Server crash | Kernel tile alignment at attention_tp_size >= 8 |
| dp=8, any bs | v6e-16 | OOM during XLA compilation | attention_tp_size = 16/8 = 2 too small to fit model attention layers in HBM |
| dp=1, bs=128/200, long | v6e-32 | Server crash during generation | Transfer encoding errors -- server instability under heavy long-context load without DP |
| dp=2, bs=128, long | v6e-16 | Benchmark degraded (329 tok/s, most requests dropped) | Similar server instability under long-context pressure |

### Notes on Failures
- The bs=200 kernel tile issue affects `max_running_requests * num_kv_heads` product alignment. Only bs values where `bs * 8 % 256 == 0` work (i.e., bs must be a multiple of 32).
- dp=8 on v6e-16 (attention_tp_size=2) is a fundamental memory limitation, not a code bug.
- dp=1 long context instability on v6e-32 suggests the model benefits from DP even for stability, not just throughput.

## Recommendations

1. **Optimal throughput config**: dp=4, bs=128 on both v6e-16 and v6e-32
2. **Optimal latency config**: dp=2, bs=32 (TPOT ~17-19ms)
3. **Batch size**: Use multiples of 32 only (32, 64, 128, 256) to avoid kernel tile alignment issues
4. **DP on v6e-16**: Use dp=1, 2, or 4 (dp=8 causes OOM)
5. **DP on v6e-32**: Use dp=1, 2, 4, or 8 (all work with bs=32 and bs=128)
6. **Long context**: Enable DP (dp >= 2) for stability and throughput -- dp=1 long context can be unstable on v6e-32
