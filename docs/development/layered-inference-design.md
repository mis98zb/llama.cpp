# GLM5 layered inference: target design

## Status and scope

This document records the requested architecture, not a claim that it is fully implemented. It is the design contract for the private-fork experiment. The default [layered-prefill implementation](layered-prefill.md) stages routed experts only. The opt-in prototype below stages complete layers and supports joint full-layer capacity planning.

The target is GLM-5.3-Flash Q4_K_XL on a CPU with sufficient system RAM and a 20 GiB GPU. The end-to-end experiment uses exactly 200000 input tokens and a generation budget of 2048 output tokens. Actual output count, early EOS and any forced-length policy must be reported separately.

## Optimization ledger

The private-fork series starts after `620d2225c`. Earlier commits establish the execution model:

| Commit | Optimization or capability |
| --- | --- |
| `f5a794b22` | Layer-major expert residency, F32/BF16 activation staging and complete hybrid state preservation |
| `abf46c871` | Double-buffered activation transfers with pinned staging, separate streams and events |
| `d825b0f08` | Budgeted, chunked next-group expert prefetch; this remains disabled in full-layer mode |
| `a2005cf9c` | Expert batch/group/prefetch planning and complete resident-prefix reuse |
| `2f4553e9c` | Complete-layer weight/state staging, capacity planning and GPU MTP with CPU verification |

This follow-up revision adds the following, with validation and limitations recorded in the linked sections:

| Optimization | Record |
| --- | --- |
| Metadata-first greedy capacity planning without candidate device-bank allocations or calibration uploads | [Planner policy](#opt-in-prototype) |
| Actual-length activation prefetch, including recurrent rollback tails | [Variable-microbatch transfers](#200k-phase-profile-and-variable-microbatch-transfers) |
| Omission of the final group's unused residual download | [Variable-microbatch transfers](#200k-phase-profile-and-variable-microbatch-transfers) |
| Aggregate and per-group phase timing, graph sub-timers and actual-path counters | [Timing definitions](layered-prefill.md#execution-timing) |
| F16 pooled-key restoration and existing WMMA dispatch, scoped to layered inference | [Four follow-up optimizations](#four-follow-up-optimizations) |
| Reuse of admitted F32 full-layer workspace bounds and existing compatible-graph guards | [Four follow-up optimizations](#four-follow-up-optimizations) |
| Automatic transfer-ceiling sizing, admitting actual microbatch 2048 within memory limits | [Four follow-up optimizations](#four-follow-up-optimizations) |
| Compact selected positions passed to existing sparse Flash Attention, with CPU reference and fallback capability checks | [Four follow-up optimizations](#four-follow-up-optimizations) |

The completed 200000-input/2048-output experiment reached 589.230 seconds of prefill including planning (339.43 tokens/second), versus 815.852 seconds (245.14 tokens/second) for the prior streaming baseline. These are combined single-run results, not isolated component gains. WMMA changes some top-k selections despite byte-identical output on this repetitive workload. Remaining measured costs include 62.679 seconds of graph allocation and 55.111 seconds of input preparation; the final run did not reuse graph objects, only admission results.

## Terminology

- Main model: the full target transformer used to establish the prompt state and verify generated candidates.
- Draft: the model's trained MTP block, executed on the GPU.
- Output head: the projection from hidden states to vocabulary logits, conventionally called `lm_head`. It is not another numbered transformer block. "Message head" is treated as this output head in this document; its exact tensor boundary must be confirmed before implementation.
- Layer group: N consecutive main-model transformer blocks whose complete weights are temporarily resident on the GPU.
- Microbatch: one bounded chunk of context data transferred into and out of the GPU for computation by the current group. Input and output DMA operations belong to that microbatch; they are not separate microbatches.
- Prefix cache: the complete compatible model state for an already computed prefix, not merely matching tokens or a saved hidden vector.

## Prefill contract

### 1. Layer-major execution and once-only weight loading

Partition the main-model blocks into consecutive, non-overlapping groups. Load the complete weights of one group, process all context positions that require prefill through that group, and only then release it and move to the next group.

The complete weights include attention, dense FFN, routed experts, shared experts, gates, normalization and other block tensors. Staging only MoE experts does not satisfy this requirement.

Each required main-model weight tensor is uploaded at most once during a successful prefill invocation. Each nonresident tensor used by prefill must be uploaded exactly once. Already resident shared tensors need not be uploaded again. Upload count and bytes must not grow with the number of microbatches.

N and microbatch size are separate choices. A larger N reduces group-boundary activation traffic; it does not remove the need to upload each required weight once. Group sizes may vary with capacity, but their ranges must cover each main-model block exactly once.

Planning must not upload a weight bank for calibration, discard it, and upload it again for execution while claiming once-only loading. Use metadata, compatible prior measurements, or retained first-use banks for planning. Any separate calibration run must be reported as a separate operation with its transfer cost.

### 2. Prefix reuse and streamed context

For a valid cached prefix P and a new suffix S, process S at positions starting at the endpoint of P. Do not recompute P's transformer layers. Attention and recurrent computation must still consume the appropriate prefix state.

For every group, stream its input activations in microbatches, compute them in causal position order, and stream the output activations back to system memory. Finish all suffix microbatches before changing the group's weights. The next group consumes the previous group's outputs.

Preserve the complete multi-stream residual at group boundaries. Do not replace it with the collapsed output-head hidden state.

Microbatching limits activations and transfer buffers. It does not mean attention may discard historical keys or that KDA may restart at every microbatch. Historical state needed by the current computation must either remain in the current group's device state or be accessed through a separately validated state-streaming algorithm. Arbitrary KV paging is not implied by activation streaming.

Prefix compatibility includes model, token positions, relevant settings and complete MLA, indexer, pooled-indexer, KDA and convolution state. Reuse must preserve partial pools and recurrent endpoints. A failed admission must preserve the prefix; a partially executed failure must explicitly invalidate it if rollback is unavailable.

### Prefill placement

| Content | System RAM | GPU VRAM | Lifetime and computation |
| --- | --- | --- | --- |
| Main-model weights | Complete authoritative weights | Current group's complete weights only | GPU executes every required suffix microbatch before group replacement |
| Prefix and resulting main-model state | Authoritative state for CPU verification | State needed by the current group | Read the group's prefix state, append the suffix, then return updated state to RAM |
| Group-boundary activations | Streamed input/output storage, optionally two alternating buffers | Bounded microbatch slots and internal group activations | Transfer at group boundaries, not after every internal layer |
| MTP and output head | CPU-accessible originals or copies where needed | Resident draft/head tensors | Not part of main-model group replacement |
| Workspace and staging | Bounded pinned buffers and metadata | Operator workspace, device transfer slots and safety reserve | Include every allocation in admission; prefetch is optional |

The state placement above is the initial prototype choice, not a requirement to keep the entire main-model state in VRAM. A capacity estimate must include weights, group state, resident draft/head, workspace, staging, backend overhead and other GPU users.

Optional next-group prefetch must preserve once-only loading and stay inside the same capacity limits. Failure to fit must reduce N or microbatch size where effective, or report an explicit error. It must not silently retain unrelated main-model layers, reduce precision or fall back to CPU prefill.

### Execution order

```text
validate cached prefix and resources
for each consecutive group:
    upload complete group weights once
    make the group's required prefix state available on GPU
    for each suffix microbatch in position order:
        transfer group input into GPU
        compute all N blocks and update their state
        transfer group output back to RAM
    return updated group state to RAM
    drain transfers and release group weights/state on GPU
produce final target logits and the hidden states needed by MTP
initialize/catch up MTP state without recomputing the main-model prefix
```

## Decode contract

### 1. Weight and state placement

All main-model weights remain CPU-accessible in system RAM. Ordinary main-model transformer blocks are not permanently resident on the GPU and are not uploaded again in each verification round.

The MTP draft block and output head reside on the GPU. The main-model verifier needs CPU-accessible embedding/head tensors too; a GPU copy does not require deleting the CPU copy. Embedding placement and the exact shared-head tensor set must be specified and measured before implementation.

Main-model verification state resides in RAM. MTP's own state resides on the GPU. These are distinct states, even when some model tensors are shared.

### 2. GPU drafting and CPU verification

The GPU generates candidate tokens using MTP. The CPU evaluates the main model on a candidate batch and determines the accepted prefix under the selected sampling rule. Only verified tokens are committed as output.

After rejection, discard unaccepted speculative state and align both contexts to the accepted sequence endpoint. KV deletion alone is not sufficient to roll back KDA or convolution history. Use validated recurrent rollback/checkpoint facilities.

Transfer the tokens, hidden states and probability information actually required by this protocol. Do not transfer the whole main-model weight set or historical state between devices each round.

### Decode placement

| Content | System RAM | GPU VRAM | Computation |
| --- | --- | --- | --- |
| Main-model weights, including CPU-accessible embedding/head | Complete weights | No ordinary main-model block bank | CPU batch verification, including target logits |
| Main-model KV/indexer/recurrent state | Resident and updated | No full target-state copy required | CPU verification and acceptance/rejection recovery |
| MTP weights, embedding/head dependencies and MTP state | Original tensor sources as needed | Resident | GPU candidate generation and draft-state catch-up |
| Candidate tokens, required probabilities and hidden rows | Verification inputs/results | Draft inputs/results | Bounded communication each round |
| Temporary buffers | CPU verification workspace | Draft workspace | Release prefill group banks and full-context activation buffers before steady-state decode |

MTP must receive the correct target hidden states and initialize its prompt history before drafting. Loading its weights alone is not sufficient. Prefill-to-decode handoff is part of the experiment, not an omitted setup step.

## Acceptance criteria and measurements

1. Start with N=1 and a short synthetic model containing both recurrent and attention layers. Compare logits, complete resulting state and subsequent CPU decode against a reference at matched precision.
2. Verify prefix plus suffix against full-prompt execution, including incomplete indexer pools and partial final microbatches.
3. Count complete layer uploads and tensor bytes. Increasing microbatch count must not increase main-model weight uploads.
4. Verify that ordinary main-model blocks leave VRAM after their group, that updated state is CPU-owned, and that decode does not reload them onto the GPU.
5. Validate MTP prompt initialization, batch verification, full/partial/zero draft acceptance, rejection recovery and the prefill/decode boundary. Check output correctness before throughput.
6. Run Q4_K_XL with exactly 200000 input tokens and a 2048-token generation budget only after short correctness checks pass. Record actual output count and stop reason.
7. Report loading, planning, prefix setup, prefill, MTP initialization and decode times separately, plus cold end-to-end time. Report draft acceptance, committed tokens/second, transfer bytes, RAM and sampled/peak VRAM with measurement method.
8. Keep model, input, precision, sampling and relevant CPU settings matched in comparisons. Declare BF16 boundary rounding separately from scheduling correctness. Do not claim a speedup from an estimated plan, matching text alone, or an unfinished run.

## Opt-in prototype

`llama_layered_prefill_params::full_layers` temporarily binds complete CPU-owned layer tensors and their cache storage to device banks. It requires exclusive access to the model while prefill runs. Weights remain CPU-owned outside that call. State is uploaded and returned once per group; the current prototype transfers the group's entire allocated state, including unused cache capacity. This is bounded by context capacity, not by the microbatch size.

Full-layer mode requires CPU cache placement and disabled expert-only prefetch. It reports separate complete-weight and state transfer counters. With `auto_plan`, tensor metadata determines complete weights, current-group state and transfer-buffer capacity. A deterministic greedy pass packs consecutive layers using a workspace reserve, then checks only that plan's real full-context graphs. Dry checks use zero-sized device-placement markers, restore all tensor bindings, and never allocate weight/state device banks, upload weights or run calibration graphs. Plans include the maximum scheduler workspace retained across groups, activation pipeline buffers when enabled and the device safety margin. The per-layer weight limit applies to each complete layer; the optional group limit applies to the entire weight bank. Allocation failures are explicit; execution after state mutation is not transactional.

This is capacity planning, not measured latency optimization or a minimum-group guarantee. It starts at `n_ubatch` (bounded by suffix length and the rollback minimum) and reserves one quarter of the current device budget for workspace, capped by `max_compute_bytes` and the space remaining after the largest single layer/state bank. This is a packing heuristic, not a hardcoded model-specific workspace bound. The selected plan's dry graphs validate the real scheduler workspace before any weight upload or execution. A larger measured workspace raises the reserve and repacks the groups; other admission failures halve the microbatch. The planner stops at the first validated plan instead of enumerating smaller batches and group caps. Graph checks are cached across corrections. Groups contain at most `max_group_layers` layers (default 4).

Smaller microbatches stop at 1, or above the recurrent rollback tail when enabled. Short suffixes retain a valid nominal microbatch even when fewer tokens remain. Statistics report `capacity_only`, zero calibration probes, attempted plans, dry graph checks and selected layer ranges. Planning never changes staging or state precision, and execution uploads each required weight only once. The graph check covers scheduler buffers, not backend-private kernel workspace; headroom remains necessary and concurrent GPU use can invalidate capacity after planning.

The `llama-layered-prefill --full-layers` example tests this prefill mode followed by ordinary CPU decode. It does not exercise speculative decoding.

The `llama-speculative-simple --layered-prefill` prototype instead forces a CPU target, loads a separate MTP-enabled model with only its MTP block and output head offloaded (`n_gpu_layers = 2`), and uses the existing speculative driver for GPU drafting and CPU verification. The separate model maps the same GGUF files; it also has CPU-side trunk metadata/mappings and is not an MTP-only GGUF. Its token embedding remains CPU-accessible and is not yet a dedicated persistent GPU embedding copy.

This prototype enables full-layer capacity planning, F32 group-boundary activations, a 32 GiB host-activation budget and explicit MTP prompt initialization from all target hidden rows. `-ub` bounds the automatic microbatch search. F32 uses about 12.2 GiB for one pinned 200k-token activation buffer, updated in place with the same pipeline used by Qwen. Target hidden rows require additional host storage. The activation pipeline uses the actual prepared microbatch lengths, including recurrent rollback tails. It does not advance state for lookahead. The final group downloads only logits and MTP hidden rows, not the unused complete residual. The `llama-layered-prefill --no-pipeline` example retains a synchronous reference path.

```sh
cmake --build build-cuda --target test-llama-archs llama-speculative-simple
build-cuda/bin/test-llama-archs --full-layers --seed 42 --backend CUDA0
build-cuda/bin/llama-speculative-simple --layered-prefill -m /path/to/model-00001-of-00006.gguf --device CUDA0 -c 202240 -b 200000 -ub 256 -fa on -ctk f16 -ctv f16 -t 16 -tb 16 --temp 0 --spec-draft-n-max 3 --spec-draft-n-min 0 -f /path/to/prompt-200k.txt -n 2048 --perf
```

The command is an experiment recipe, not a completed 200k result. By default EOS can stop it early. Use `--ignore-eos` explicitly only for a fixed-length throughput experiment and report that policy.

Synthetic full-layer tests compare N=1/2 and automatic groups with microbatches 7/32, require invariant complete-weight bytes including planning, compare all MTP hidden rows against a single-group same-GPU reference, and check CPU decode, cached-prefix suffix execution, binding restoration and budget failure cleanup. Automatic cases require complete two-layer groups and zero calibration uploads/probes. Cross-CPU/CUDA final-logit NMSE is checked separately at `1e-4`; this is not a real-model quality bound. Same-GPU hidden-row comparison isolates scheduling from sparse-routing differences between CPU and CUDA.

### Initial verification

The measurements below predate integration with the single pinned activation buffer and Qwen weight-upload pipeline. They describe the recorded executables, not new performance measurements of the integrated branch.

The SM86 CUDA build passed the full-layer tests with seeds 42 and 17, the four existing registered CPU allocator/layered tests, and the existing CUDA layered-prefill and resident-prefix tests with seed 42. Deliberate full-layer budget rejection logs are expected in the new tests.

The real Q4_K_XL short run with 126 input tokens and a forced 32-token greedy output completed N=1 full-layer prefill in 47.549 seconds, MTP prompt initialization in 0.086 seconds and decode in 3.249 seconds. Its output text matched the CPU greedy reference for the same prompt. Complete uploaded weights were 195118640376 bytes; state upload and download were each 620625920 bytes. A separate 5-input/8-output run accepted 3 of 10 proposed tokens and completed, exercising partial acceptance. These are short functional checks, not evidence of 200k performance or general model quality.

### Completed 200k experiment

The N=1, F32-staged Q4_K_XL run on the RTX 3080 20 GiB completed with exit code 0. Input length was independently tokenized as exactly 200000 tokens. The prompt repeats `The sky is blue. ` 39999 times and ends with `The capital of France is`; it is a synthetic throughput workload, not a representative document or quality benchmark.

| Measurement | Result |
| --- | --- |
| Full-layer target prefill | 1499.236 seconds, about 133.40 tokens/second |
| MTP prompt initialization | 65.956 seconds |
| Encoded phase including MTP setup and bookkeeping | 1565.227 seconds |
| Actual generated output | 2048 tokens, greedy, EOS explicitly ignored |
| Decode | 360.963 seconds, 5.674 tokens/second |
| Proposed / accepted draft tokens | 1536 / 1535 (99.935 percent) |
| End-to-end process wall time | 1963.183 seconds |
| Complete prefill weight uploads | 195118640376 bytes |
| State upload / download | 4597088256 / 4597088256 bytes |
| Largest measured weight/state bank | 5831554048 bytes |
| Highest sampled total GPU use | 13822 MiB, sampled every 5 seconds, including desktop and other users |

The speculative loop reported the generation budget as its stop reason. The repeated prompt and highly repetitive continuation make the acceptance rate unsuitable as a general MTP acceptance estimate. No matched 200k CPU baseline or full-length output-quality comparison has completed, so this run establishes execution and capacity, not a speedup or a quality guarantee. Use the explicit phase timers above: the generic context performance summary mixes custom prefill, batched verification and initialization accounting.

### Automatic complete-layer capacity experiment

The same model, exact 200000-token prompt and 2048-token forced greedy continuation completed with exit code 0 after enabling the original candidate-search full-layer planner in the MTP example. This measurement predates the deterministic metadata planner above. The command above used `-ub 1024` instead of `-ub 256`; all other arguments were unchanged. This comparison changes both grouping and microbatch size and does not isolate their individual effects. The input SHA-256 was `21d5ba1e6233d7dd649de3e44705691721f1f2d1d8cf624d4b87aa615209a04e`.

The planner selected microbatch 1024 and 22 groups covering all 45 blocks: `[0,4)`, two-layer groups from `[4,6)` through `[42,44)`, then `[44,45)`. It evaluated 32 candidate plans with zero calibration probes. There were no extra weight uploads for planning.

| Measurement | N=1, microbatch 256 | Automatic, microbatch 1024 |
| --- | --- | --- |
| Target prefill, including planning when enabled | 1499.236 seconds | 874.084 seconds |
| Target prefill throughput, including planning | 133.40 tokens/second | 228.81 tokens/second |
| Capacity planning | Not enabled | 139.617 seconds |
| Target prefill excluding planning (subtracted phase timers) | 1499.236 seconds | 734.467 seconds, 272.31 tokens/second |
| MTP prompt initialization | 65.956 seconds | 40.428 seconds |
| Encoded phase including MTP and bookkeeping | 1565.227 seconds | 914.546 seconds |
| Decode, 2048 actual output tokens | 360.963 seconds | 323.915 seconds, 6.323 tokens/second |
| End-to-end process wall time | 1963.183 seconds | 1334.945 seconds |
| Complete prefill weight uploads | 195118640376 bytes | 195118640376 bytes |
| State upload / download | 4597088256 / 4597088256 bytes | 4597088256 / 4597088256 bytes |
| Largest weight/state bank | 5831554048 bytes | 10407346560 bytes |
| Highest sampled total GPU use | 13822 MiB | 18901 MiB |

Prefill throughput including planning improved by 1.72x and prefill time fell by 41.70 percent relative to the prior N=1 run. These are separate single runs, not repeated controlled measurements. GPU sampling includes planning, desktop and other users; it is not an allocator peak measurement. The automatic run proposed 1536 draft tokens and accepted 1535, stopped at the generation budget, and produced byte-identical stdout to the prior run (SHA-256 `dd3cebd8f0853506f5a579c92bdd3d9f28fe9d2b45f1f87f7b675c67c5453434`). This synthetic repeated-text result does not establish general quality or representative MTP acceptance.

Validation passed: the Release SM86 build of `test-llama-archs`, `llama-speculative-simple` and `llama-layered-prefill`; full-layer CUDA tests with seeds 42 and 17; existing CUDA layered-prefill, resident-prefix and joint-planner tests with seed 42; and `ctest --test-dir build-cuda --output-on-failure -R '^(test-layered-prefill|test-layered-planner|test-layered-prefix|test-alloc)$'` (4/4 passed). Automatic full-layer tests also cover workspace-driven microbatch reduction, weight-budget-driven group reduction, short suffix reuse, invalid rollback batch rejection and preservation of the prefix on admission failure.

### Deterministic planner smoke test

The metadata-first planner completed the same real-model short-input test used to validate phase timings: context capacity 202240, 45 input tokens, microbatch ceiling 16 and 32 forced greedy output tokens. It selected 23 groups from one metadata plan, built 23 admission graphs and performed zero calibration probes. Planning took 1.339 seconds, compared with 34.174 seconds for the earlier candidate-search run with the same arguments. Generated stdout was byte-identical (SHA-256 `ac1e29e320b7737aff283709b93823fd187452fe94880ab669bedd6dc857a0b1`), and complete weight/state transfer counts were unchanged.

The new prefill phase took 29.088 seconds including planning and 27.749 seconds executing. These are single smoke runs; CUDA regression tests overlapped the new run, so execution timing is not an isolated performance comparison. Context capacity is 200k-class, but actual input is only 45 tokens. This does not replace the full 200000-token benchmark above or establish the new planner's long-context throughput.

### 200k phase profile and variable-microbatch transfers

Two sequential runs used the same exact 200000-token synthetic input, `-ub 1024`, F32 staging, CPU-owned main-model state, GPU MTP and 2048 forced greedy output tokens. The first profiled the deterministic planner with synchronous activation transfers. The second enabled the existing activation pipeline for rollback-aware variable microbatches and removed the final group's unused complete-residual export. Neither run overlapped the regression tests.

| Measurement | Synchronous profile | Variable-microbatch pipeline |
| --- | --- | --- |
| Prefill including planning | 868.610 seconds | 815.852 seconds |
| Prefill throughput including planning | 230.25 tokens/second | 245.14 tokens/second |
| Planning | 2.367 seconds | 2.899 seconds |
| Selected execution | 866.233 seconds | 812.940 seconds |
| Selected groups / dry admission graphs | 32 / 32 | 42 / 42 |
| Microbatch ceiling and selection | 1024 | 1024 |
| Initial setup | 16.624 seconds | 17.073 seconds |
| Memory metadata and batch preparation | 21.981 seconds | 28.929 seconds |
| Weight-bank loading | 19.928 seconds | 20.423 seconds |
| State upload / download | 0.455 / 0.568 seconds | 0.512 / 0.566 seconds |
| Graph build, measure and allocation | 75.697 seconds | 69.140 seconds |
| Mask/index and other graph inputs | 52.566 seconds | 53.534 seconds |
| Explicit activation upload calls/waits | 26.816 seconds | 0.218 seconds |
| Compute submission through synchronization | 592.305 seconds | 584.251 seconds |
| Explicit activation download calls/waits | 55.926 seconds | 34.691 seconds |
| Logits and MTP hidden-row download | 0.280 seconds | 0.425 seconds |
| Cleanup | 3.075 seconds | 3.154 seconds |
| MTP prompt initialization | 41.368 seconds | 46.320 seconds |
| 2048-token decode | 340.742 seconds | 337.058 seconds |
| Process wall time | 1288.413 seconds | 1236.647 seconds |
| Highest sampled total GPU use | 17684 MiB | 15292 MiB |

Both runs exited with code 0 and generated byte-identical stdout with SHA-256 `dd3cebd8f0853506f5a579c92bdd3d9f28fe9d2b45f1f87f7b675c67c5453434`. Complete weight uploads stayed at 195118640376 bytes; state upload/download stayed at 4597088256 bytes each. The pipeline run allocated 536870912 bytes of combined device/pinned transfer buffers, issued 7995 next-input prefetches and 8036 asynchronous residual downloads, and transferred 537395200000 activation bytes in each direction. These counts equal 41 group boundaries times 196 microbatches (195 prefetched after the first) and 200000 complete four-stream residual rows. The final group's omitted download saves 13107200000 bytes independently of the number of groups.

Observed prefill throughput increased by 6.47 percent and prefill time fell by 6.07 percent. This is not a fixed-plan isolated pipeline A/B: admission selected 32 versus 42 groups, with different memory footprints, and the runs are single samples in a shared GPU environment. The larger boundary count in the pipeline run means total residual traffic increased despite removal of the final download. Host timing categories do not measure pure GPU kernel or DMA time; in particular, next-input preparation and prefetch submission inside the compute/wait interval are charged there. Graph handling and input preparation remain measurable optimization targets, but this profile does not identify which CUDA kernels dominate.

The updated Release build passed the four registered CPU layered/allocator tests, complete-layer CUDA tests with seeds 42 and 17 in synchronous and pipelined modes, and existing CUDA layered-prefill, resident-prefix and joint-planner tests with seed 42. Complete-layer tests cover groups 1/2/4 and automatic groups, microbatches 7/32, a rollback-split penultimate batch (52 tokens at microbatch 7), a suffix shorter than the rollback tail, once-only weights, hidden rows, subsequent CPU decode and admission-failure prefix preservation. A single-group case requires zero residual uploads/downloads. The cost-model test also verifies that the last group is not charged residual-download bandwidth.

### CUDA compute trace

A real 200000-token run with the same F32 staging, context capacity 202240, microbatch ceiling 1024 and GPU MTP was sampled with Nsight Systems 2026.1.3. The generation budget was one token to isolate prefill rather than repeat the long decode experiment. CUDA collection used `--trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --delay=180 --duration=120 --kill=none`. CPU sampling was disabled because the local perf permissions do not allow it. The target continued after collection and completed successfully. Profiled elapsed time is not used as the unprofiled throughput baseline.

The approximately 120-second window includes KDA and DSA work, not the whole prefill. Summed kernel durations were 82.906 seconds:

| Kernel family | Device duration | Share of summed kernel time |
| --- | --- | --- |
| DSA lightning indexer | 33.379 seconds, 392 calls | 40.26 percent |
| Quantized matrix multiplication, including dense and MoE | 19.913 seconds | 24.02 percent |
| Flash attention | 13.159 seconds | 15.87 percent |
| KDA gated delta net | 3.155 seconds | 3.81 percent |
| Top-k and argsort | 0.576 seconds | 0.69 percent |
| Other kernels | 12.723 seconds | 15.35 percent |

The dominant indexer was `lightning_indexer_kernel_vec<8,64,128,32,GGML_TYPE_F32>`, not the WMMA implementation. The current dispatch excludes F32/BF16 keys from WMMA. Its grid has one query per block in the batch dimension; raising the microbatch does not turn this vector implementation into a cross-query matrix tile. The WMMA path narrows query/key arithmetic, so forcing it is not a precision-preserving optimization. FP32-preserving query tiling/key reuse is a candidate, not a demonstrated speedup.

CUDA copies also matter. Host-to-device copies summed to 21.640 seconds, device-to-host to 9.269 seconds, and device-to-device to 1.220 seconds. Merging device intervals instead of adding overlapping operations gives 31.474 seconds of copy-active time, 9.174 seconds overlapping kernels, 22.300 seconds with copies but no traced kernel, and 14.764 seconds with neither traced kernels nor copies in the 119.970-second GPU-event span. Copy-only time includes required bank uploads and must not all be called avoidable stalls.

The activation-upload stream transferred 92.107 GB in 8.634 seconds, with 8.345 seconds overlapping kernels. The other H2D stream transferred 137.988 GB in 13.006 seconds without kernel overlap, including scheduler inputs and bank uploads. Activation downloads transferred 92.086 GB in 9.165 seconds, with only 1.214 seconds overlapping kernels. This shows that upload overlap is effective but input/bank transfers and output downloads still expose latency. CUDA API `cudaStreamSynchronize` summed to 86.938 seconds; that host wait overlaps device work and is not another 86.938 seconds of GPU idleness.

These measurements identify the indexer as a larger kernel-time target than top-k sorting in this window. They do not separate MoE from all other quantized matrix multiplication, establish achieved DRAM bandwidth or SM occupancy, or prove a memory-bound versus compute-bound roofline classification. Those require additional operation attribution or hardware counters. The model has 288 routed experts and selects eight per token: under uniform routing, microbatch 1024 gives only about 28.4 token assignments per expert. Small expert matrices remain plausible, but they are not the largest measured kernel family here.

### Unprofiled microbatch-ceiling experiments

Three additional complete 200000-token runs used the same executable, input SHA-256, model, F32 staging, context 202240, target/draft placement, 16 threads, greedy sampling and one forced output token. No profiler or other test was launched concurrently with these runs. Automatic grouping was left enabled, so these are capacity-policy experiments rather than fixed-group GEMM benchmarks.

| Requested microbatch ceiling | Selected microbatch | Groups | Prefill including planning | Throughput | Compute/wait | Sampled total GPU peak |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 23 | 851.569 seconds | 234.86 tokens/second | 639.660 seconds | 17406 MiB |
| 1024 | 1024 | 32 | 755.421 seconds | 264.75 tokens/second | 544.102 seconds | 17891 MiB |
| 2048 | 1024 | 42 | 835.661 seconds | 239.33 tokens/second | 593.777 seconds | 16640 MiB |

The default combined device/pinned transfer limit is 512 MiB. Four F32 device slots and four pinned slots require `8 * 4096 * 4 * 4 * microbatch` bytes: exactly 512 MiB at 1024 and 1 GiB at 2048. Therefore the requested-2048 experiment reduced the actual selection to 1024; it does not measure the performance of 2048-token microbatches. An actual-2048 experiment needs a larger admitted transfer budget or a synchronous reference, with the changed placement/capacity explicitly accounted for.

The requested-512 run was slower than the first unprofiled 1024 run, consistent with a benefit from larger microbatches, but it does not isolate small expert-matrix utilization. In particular, the two runs that selected 1024 differed by about 10.6 percent in prefill time and selected 32 versus 42 groups. The same ceiling is not a fixed plan, and the shared GPU environment plus capacity policy prevent attributing this difference solely to matrix shape.

All three plain runs and the profiled run exited with code 0, covered all 45 main-model layers exactly once, uploaded 195118640376 weight bytes, uploaded/downloaded 4597088256 state bytes each, and generated an identical first output token. The profiled run selected 42 groups, took 756.173 seconds for prefill and is excluded from the plain-run throughput comparison. Phase-sum and layer-coverage checks passed for every run. No production code or numerical precision was changed for these experiments.

### Four follow-up optimizations

The following changes are confined to the explicitly enabled layered path and optional compact-index operation input. Ordinary GLM5-Next inference retains its prior graph and indexer precision policy.

1. A fused GLM5-Next indexer restores gathered pooled keys to F16 only with F32 activation staging and underlying F16 cache storage. BF16 activation staging retains reference F32 indexer dispatch because boundary rounding can amplify dispatch differences. Pooling and gate arithmetic remain F32; F32/BF16/quantized cache types are not forced to F16. This reuses the existing CUDA WMMA dispatch. F16 -> F32 -> F16 preserves finite cached values, but the WMMA kernel also narrows queries, so the complete computation is not bit-exact to the F32 vector reference.
2. Admitted F32 full-layer execution reuses the dry-admitted workspace bound instead of measuring and rebuilding each execution graph. Actual scheduler buffer sizes are checked before compute, including the separate retained host/device maxima. Compatible graphs use existing reuse guards; alternating pipeline slots, growing attention shapes and rollback heads can still prevent reuse. Manual/BF16/expert paths retain measurement where needed. Build, measurement and allocation sub-timers make first-use backend allocation distinct from repeated graph construction.
3. The MTP example automatically sizes the combined device/pinned transfer ceiling from the requested microbatch. Host/device budgets and headroom still constrain admission, and explicit limits in the standalone example are preserved. Actual microbatch 2048 can now be admitted with 1 GiB transfer staging.
4. Supported single-stream, independent-indexer CUDA layers pass compact I32 selected positions directly to the existing sparse Flash Attention machinery. The backend compacts the supplied positions rather than scanning a reconstructed dense selection mask. The ordinary causal mask remains and still controls per-query visibility and additive bias; this is not removal of all CPU mask preparation. Unsupported shapes/backends and shared indexers keep the prior mask graph. CPU provides a compact-index reference; other backends explicitly reject this optional input. Empty selections return zero, including the no-sink case.

A same-GPU synthetic indexer comparison used 50000 keys, 1024 queries, 32 heads of dimension 128, a growing causal prefix, F16 cached keys and the same F32 queries/weights. Relative to the F32 vector path, restored-F16 WMMA scores had NMSE `4.27753971e-8`, maximum absolute difference `7.45132565e-5`, and top-512 set overlap 99.969482 percent; 159 of 1024 rows changed at least one selection. This is numerical characterization, not proof of unchanged model quality. The test passed the existing indexer score NMSE bound of `1e-6`.

The real 45-token prompt with 32 forced greedy outputs remained byte-identical to the earlier reference. A CUDA trace of that short run observed 33 launches of `lightning_indexer_kernel_wmma<8,32,128,32,GGML_TYPE_F16>`; runtime graph flags confirmed both the F16 indexer and compact attention paths. Compact-index operator tests cover CPU/CUDA parity with 256/4096 keys, 1/7/32 queries, one/two streams, K/V views, sinks, finite additive mask bias, negative/out-of-range indices and completely empty selections. Ordinary mask-based attention is tested separately.

The final combined run used the exact same 200000-token input, F32 boundary staging and 2048 forced greedy output tokens as the earlier streaming baseline. It exited with code 0. The requested and selected microbatch were both 2048, with 42 groups, 1 GiB combined transfer staging, one metadata plan and 42 dry admission graphs. Runtime graph flags confirmed `f16-indexer=1` and `compact-attention=1`.

| Measurement | Earlier streaming baseline | Four optimizations |
| --- | --- | --- |
| Actual microbatch | 1024 | 2048 |
| Groups | 42 | 42 |
| Prefill including planning | 815.852 seconds | 589.230 seconds |
| Prefill throughput including planning | 245.14 tokens/second | 339.43 tokens/second |
| Planning | 2.899 seconds | 5.282 seconds |
| Selected execution | 812.940 seconds | 583.936 seconds |
| Compute/wait | 584.251 seconds | 368.232 seconds |
| Graph handling | 69.140 seconds | 66.119 seconds |
| Graph build / measurement / allocation | Not separated | 0.350 / 0.000 / 62.679 seconds |
| Graph builds / compatible reuses / execution measurements | Not counted | 4116 / 0 / 0 |
| Input preparation | 53.534 seconds | 55.111 seconds |
| MTP prompt initialization | 46.320 seconds | 42.149 seconds |
| 2048-token decode | 337.058 seconds | 334.044 seconds |
| Process wall time | 1236.647 seconds | 1004.967 seconds |
| Sampled total GPU peak | 15292 MiB | 17510 MiB |

Combined observed prefill throughput improved by 1.385x and prefill time fell by 27.78 percent. This changes several mechanisms and the actual microbatch together; it does not isolate each optimization's gain, and these remain single runs in a shared GPU environment. The original allocation-related cold cost remains: actual graph construction now totals only 0.350 seconds, while allocation accounts for 62.679 seconds. No compatible graph-object reuse occurred with these alternating pipeline slots; the measured reduction is repeated admission/build work, not a claim that all graphs were cached.

The complete-weight upload count remained 195118640376 bytes and state upload/download remained 4597088256 bytes each. Activation transfers were 537395200000 bytes in each direction, with 3977 next-input prefetches and 4018 asynchronous downloads. All 45 main-model layers were covered once. The 2048-token stdout was byte-identical to the earlier baseline (SHA-256 `dd3cebd8f0853506f5a579c92bdd3d9f28fe9d2b45f1f87f7b675c67c5453434`), and MTP accepted 1535 of 1536 proposed tokens. Repetitive-text equality does not establish general quality equivalence given the measured top-k changes.

Validation passed on the Release SM86 build: `ctest --test-dir build-cuda --output-on-failure -R '^(test-layered-prefill|test-layered-planner|test-layered-prefix|test-alloc)$'` (4/4); full-layer CUDA tests with seeds 42 and 17; existing CUDA layered-prefill, prefix and joint-planner tests with seed 42; `test-backend-ops -b CUDA0 -o LIGHTNING_INDEXER -p 'hsk=128,nh=32.*type_K=f16'` (6/6); compact-index CPU/CUDA comparison including empty single-query rows (28/28); and combined compact/ordinary-mask operator tests (33/33). The real-run verifier checked actual batch size, both active kernel paths, zero execution dry measurements, phase sums, layer coverage, transfer counts and exact output bytes. Backends other than CPU/CUDA were not built locally; their changes only reject the new optional input rather than adding implementations.

### Workspace, query windows, visibility and segmented graphs: validation pending

The previous implementation and measurements are committed in `b5350fd7a`. The next prototype follows the order below. These changes reuse existing ggml storage, arithmetic and CUDA graph mechanisms; no source from the sibling engines is copied.

1. Admitted F32 complete-layer groups reserve scheduler workspace with a full-span graph after binding that group's weights/state and before its first compute. The graph is not executed and weights are not uploaded again. This uses the same real tensor layout as execution rather than synthetic byte tensors, preserves the existing admission bound, and moves buffer growth out of the incremental microbatch loop. `reserved_workspace_bytes` and `workspace_reserve_ms` report the reservation; the latter is part of `graph_ms`, not an extra additive phase. This is not evidence that driver cold-loading or the previously measured allocation cost has been eliminated.
2. Layered fused indexer scoring and top-k ordering run in windows of at most 64 queries. Projection, pooling, model microbatch size and layer-major weight residency are unchanged. Window scores are consumed before the next window's score buffers are needed; only the selected indices are concatenated for the whole microbatch. Non-layered and non-fused paths retain their full-query calculation. `max_indexer_queries` reports the actual operator width.
3. Eligible single-sequence causal attention uploads F32 key/query positions instead of a full host-filled causal mask. Empty or unrelated cache cells are represented by -1. Existing device arithmetic constructs the same zero/-infinity visibility mask. The compact path excludes ALiBi, sliding windows and contexts outside exact F32 integer range. Layered fused pool inputs similarly upload one half-open pool interval per query; each scoring window constructs its own F16 mask. Existing pool layout/rank logic determines the intervals, so partial pools, padded pools, replay and restored prefixes retain their original semantics. This still materializes ordinary attention masks on the device; it is not a new mask-free attention kernel.
4. `GGML_TENSOR_FLAG_GRAPH_BOUNDARY` requests a scheduler segment even when the backend does not change. Layered GLM5 marks FFN entrances and exits, separating the stable FFN body from growing attention/indexer work. Same-device segments share the existing allocator and compute stream; no cross-device copy or extra host synchronization is implied. Existing CUDA graph compatibility checks, warmup and fallback remain authoritative. Segment creation is not a claim that CUDA capture/replay actually occurred, especially when allocated addresses or batch shapes change.

CPU-only validation passed:

- `cmake -S . -B build-prefill-cpu -G Ninja -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON` and the `test-llama-archs`, `test-backend-ops`, `test-alloc` build targets.
- `ctest --test-dir build-prefill-cpu --output-on-failure -R '^(test-layered-prefill|test-layered-planner|test-layered-prefix|test-alloc)$'`: 4/4 passed, including actual compact causal input and backend segmentation assertions, 97-token microbatches and existing F32/BF16 state/decode/prefix checks.
- `build-prefill-cpu/bin/test-llama-archs --full-layers --backend CPU --seed 42`: passed for groups 1/2/4/automatic and nominal microbatches 7/32/97, with short-prefix suffixes and workspace/weight rejection checks. The pipeline-requested cases retain the synchronous CPU path; they do not validate GPU DMA. Cross-reference CPU logits had NMSE zero.
- `build-prefill-cpu/bin/test-backend-ops -b CPU -o SUB -p 'query_window=64'`: 6/6 passed for complete-versus-windowed scores with 65/97/137 queries and one/two streams.
- The allocator test checks that explicit boundaries create three same-device segments instead of one while maintaining valid allocations and tensor lifetimes.

The corresponding SM86 CUDA targets compile, but final CUDA execution, actual capture/replay and the new 200k benchmark are blocked by the local driver. A GPU test blocked before producing its first CUDA initialization message, and an independent `nvidia-smi` also blocked in `os_acquire_rwlock_write`. No new model graph had executed when the initialization stalled. Only task-owned clients were stopped; the shared GPU was not reset and other processes were not terminated. The earlier 589.230-second result belongs to `b5350fd7a`, not to this unmeasured prototype. CUDA validation and long-context performance/quality comparisons must be repeated after driver recovery before treating this round as a verified optimization.
