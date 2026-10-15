# Experimental layer-major prefill for GLM5-Next

## Goal and scope

Process a text prompt or appended suffix one contiguous layer group at a time. Keep the group's complete routed experts on the device while processing all input microbatches in position order. Only then replace them with the next group's weights. Group size one is the original layer-major path. This is not ordinary chunked prefill and is not a larger decode expert cache.

The initial target is GLM-5.3-Flash, quantized GGUF, EPYC 7532 (AVX2), 256 GB RAM and a 20 GB SM86 GPU. Development on Windows has no target model. No real-model accuracy, 200K-token performance or speedup has been established.

The path is explicit opt-in. Default decode remains unchanged. It accepts sequence 0, consecutive text positions starting at zero, and the final token's logits. With `reuse_prefix = true`, the supplied tokens are only the suffix of the context's complete resident prefix state. Multimodal input batches, speculative contexts, adapters, shared-indexer variants, pending sequence edits and concurrent sequences remain unsupported.

## Execution

1. Validate the input, cache endpoints and configured resource limits. Automatic planning preflights every selected group before updating model state.
2. Allocate two bounded host activation buffers.
3. For each group, make its complete routed expert tensors device-resident, build a multi-layer graph and process every microbatch in position order.
4. Keep internal group activations on the device. Read and write the complete mHC residual only at group boundaries: `[hidden_size, 4, suffix_tokens]`. Never collapse the streams before the output head.
5. Preserve every layer's final MLA, indexer, convolution and recurrent state for subsequent ordinary decode.
6. Run the output head for the final token and expose the usual logits.

The implementation snapshots and replays only host metadata: attention/indexer cell maps, allocation heads, recurrent cell/source maps, rollback indices, k-pool layout and stale markers. Tensor data is not copied out for replay. Each group starts from the same prefix metadata and appends the same suffix positions; previously completed groups' tensor contents remain intact. Graph synchronization precedes metadata replay and weight replacement.

This preserves each layer's own prefix MLA, indexer key/gate/pooled rows, KDA matrix and Q/K/V convolution window. It does not substitute a token match or one hidden-state vector for the complete hybrid state. Metadata snapshots use additional ordinary host memory outside the activation-buffer budget.

The official GLM-5.3-Flash configuration has full indexers at every attention layer. This implementation rejects shared-indexer variants explicitly. Supporting them would additionally need compact selected indices and visibility information across layers. Saving a dense whole-prompt selection mask is prohibited because it grows quadratically.

## Precision and resource budgets

Use BF16 for host activation staging by default, with F32 as a correctness reference. Conversion must happen on the device to reduce PCIe traffic. This does not change the precision of KDA recurrent state, mHC computation, attention masks or arithmetic inside existing operators.

For BF16 staging, internal group boundaries still round to BF16 and convert back to F32 on the device. Increasing group size therefore removes PCIe round trips, not the specified rounding points. The planner never changes staging or KV precision to make a plan fit.

BF16 staging is not lossless. In particular, rounding can change the DSA top-k selection. An exploratory sparse miniature model with seed 17 produced final-logit NMSE about `8.9e-3` relative to unrounded F32. This is not a real-model quality measurement, but it rules out claiming that BF16 staging always preserves the original logits closely. Evaluate BF16 on representative real prompts before using it for quality-sensitive work; select F32 staging to isolate scheduling correctness.

For hidden size 4096, four streams and microbatch 256, one BF16 boundary block is 8 MiB in each direction. At 200000 tokens, two complete BF16 activation buffers require about 12.2 GiB of host RAM, excluding all other allocations.

The routed experts of one layer contain `3 * 4096 * 2048 * 288` parameters. If all three tensors use Q4_K, their storage is about 3.80 GiB. Mixed GGUF quantization can require more. Check actual tensor sizes and backend operator support, not the file's Q4 label.

The complete working set includes current-layer weights, persistent device caches, activation buffers, masks, operator workspace, backend overhead and safety headroom. Do not double-count allocations already reflected in device free memory.

The experimental API separately limits the two host activation buffers, the current expert buffer including backend padding, and the sum of scheduler compute buffers. The last limit is not a total-VRAM limit: persistent model/context allocations and backend-private kernel workspaces remain additional costs. Backend allocation failures are errors, not fallbacks.

Start evaluation at microbatch 256, compare 512, and consider 1024 only after measuring real graphs. These are candidates, not verified capacity limits. Manual execution uses `n_ubatch` and `group_layers`; `auto_plan` searches smaller batches and variable-length groups. The selected batch size remains fixed for the entire request, so metadata replay never silently changes chunk boundaries. Actual graphs are measured again before allocation and must stay within their admitted workspace bound.

Admission/planning failures preserve an existing prefix. Once model execution has updated state, a failure can invalidate the whole context because KDA cannot be rolled back by deleting token positions. `prefix_invalidated` explicitly reports this case; the implementation logs it, returns an error and does not pretend that the old prefix can still be reused.

Flash Attention and a suitable fused indexer are important at long context. With 200000 keys and 1024 queries, one dense F32 attention score tensor across 64 heads is about 48.8 GiB. A BF16 staging buffer does not reduce that workspace. Unsupported fused operations must not be disguised as successful GPU acceleration.

If a complete layer's experts and required persistent state cannot fit, shrinking the microbatch cannot solve the fixed-memory problem. Report it instead of silently streaming experts repeatedly.

## Validation

Use a small synthetic GLM5-Next model with both KDA and DSA layers and multiple microbatches. Compare normal and layer-major F32-staged logits, final cache/state and subsequent ordinary decode. Test BF16 staging separately with a documented tolerance. Cover partial final microbatches, pool boundaries, invalid requests, budgets and failure cleanup.

Count uploaded expert bytes and layer loads. For the layer-major path, complete expert uploads must not grow with the number of microbatches. CPU execution can validate scheduling and state handling, but cannot validate CUDA kernels, GPU residency or PCIe performance.

The synthetic test uses four alternating KDA/DSA layers, three MoE layers, F32 or Q4_K experts, 137 tokens, top-k 32 and microbatches of 7, 32 and 64 tokens. F32-staged CPU execution requires identical serialized sequence state and prompt/decode logits NMSE at most `1e-10`.

BF16 execution is compared with a microbatch-major reference that rounds intermediate layer boundaries to BF16 using the existing evaluation callback. That comparison tests execution order at the same staging precision, not BF16 quality relative to unrounded F32. Its NMSE bound is `1e-5`. CUDA execution uses a same-GPU reference with the same precision and an NMSE bound of `1e-5`, rather than conflating CPU/CUDA kernel differences with scheduling changes. Subsequent decode uses the same model and weight placement for both prefixes by restoring the reference's sequence state. GPU tests do not require byte-identical cross-backend state.

```powershell
cmake -S . -B build-prefill -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-prefill --config Release --target test-llama-archs llama-layered-prefill
.\build-prefill\bin\test-llama-archs.exe --layered-prefill --seed 42
```

For a CUDA build, add `-DGGML_CUDA=ON` and select an available device for the same synthetic test:

```powershell
.\build-prefill-cuda\bin\test-llama-archs.exe --layered-prefill --seed 42 --backend CUDA0
```

On the actual machine, compare complete prompt prefill time with identical GGUF, tokens, cache types and relevant llama.cpp settings. Report weight transfer, activation transfer and computation separately where measurable. Do not infer end-to-end speedup from reduced transfer counts.

### Local verification

CPU and CUDA builds completed with MSVC. The registered CPU test passed:

```powershell
ctest --test-dir build-prefill -R "^test-layered-prefill$" --output-on-failure
```

The precision/quantization combinations passed on CPU and on a local NVIDIA T1000 (SM75); additional seed 17 runs exercised different sparse selections. The activation-pipeline tests run all 12 combinations with pipelining disabled and enabled (24 cases). CPU execution retains its synchronous path. The CUDA runs used `CUDA_VISIBLE_DEVICES=2` to isolate that device and `--backend CUDA0` inside the process. CPU F32 sequence-state comparisons were byte-identical. Existing `glm5-next`, `qwen4exp` and `kimi-linear` architecture checks also passed.

Across microbatches 7, 32 and 64, the miniature model copied exactly 1769472 expert bytes with F32 experts, or 663552 bytes with Q4_K experts, in three layer loads. The tests also cover invalid tokens, nonempty-context rejection, host/compute/expert budgets, cleanup and successful single-token retry after failure.

The standalone CUDA example loaded a locally generated tiny GGUF and completed both prefill and ordinary generation with BF16 and F32 staging. These tests did not use a downloaded GLM model. They do not validate SM86 fused long-context attention, 20 GB capacity, 200K latency, or real-model BF16 quality.

## Experimental API and example

`llama_prefill_layered` in `src/llama-ext.h` accepts a complete initial tokenized prompt, or only new tokens when `reuse_prefix = true`, and returns the final token's logits through the ordinary logits getters. A suffix starts at the live sequence-0 endpoint plus one. A subsequent ordinary decode starts at prefix length plus suffix length. Unsupported configurations return `-1`; resource and execution failures return `-2`, with explicit diagnostics.

The defaults are BF16 staging, microbatch 256, 16 GiB of host activation storage including pinned staging, 6 GiB of current-layer expert storage, and 8 GiB of scheduler compute buffers. The 512 MiB transfer budget covers activation device buffers and all pinned staging, including optional weight-prefetch bounce buffers. C++ callers can set these limits through `llama_layered_prefill_params`. CPU execution requires `require_gpu = false` and is intended for correctness testing. Only one GPU is supported.

The separate `llama-layered-prefill` example uses the shared llama.cpp argument parser, keeps routed experts in plain CPU buffers, invokes the new path, reports copy statistics and then performs greedy ordinary decode. `--staging-type f32` selects the unrounded reference; the default is `bf16`. It raises the logical batch capacity to hold the complete prompt; `-ub` remains the physical microbatch bound. It does not enable the feature in the normal CLI or server.

```powershell
.\build-prefill-cuda\bin\llama-layered-prefill.exe -m D:\models\glm-q4.gguf -f D:\prompts\input.txt --device CUDA0 -ngl 99 -c 200064 -ub 256 -fa on -ctk f16 -ctv f16 -n 32
```

The command requires a suitable real GGUF and is not a verified 200K/20GB configuration. Expert tensors must be contiguous, host-resident and in their ordinary GGUF layout. Repacked CPU-only expert buffers are rejected. The scheduler must place expert operations and staging conversions on the selected backend; otherwise the call fails instead of presenting a CPU fallback as GPU acceleration.

For a fixed two-layer group, add `--group-layers 2`. For joint automatic planning, add `--auto-plan --max-group-layers 4`; `-ub` becomes the search ceiling. `--group-budget-mib` optionally caps all current-group experts, independently of the per-layer expert limit. `--max-prefetch-layers` is a separate ceiling and need not equal the compute group size. No change is enabled in the normal CLI or server.

## Microbatch activation pipeline

On an asynchronous GPU backend with events and pinned host buffers, `pipeline = true` (the default) uses two input slots and two output slots. The complete layer activations stay in pageable system memory; only the bounded bounce buffers are pinned. Failure to obtain real pinned storage is an error, not a silent pageable substitute. `--no-pipeline` selects the synchronous reference in the example.

Each slot owns persistent device storage outside the scheduler's graph buffers. A graph reads its input slot and writes its output slot directly, so resetting the graph cannot invalidate an in-flight activation copy.

- Upload and download use separate backend streams from compute.
- Before compute consumes an input slot, its stream waits for that slot's upload event.
- Once a microbatch has been submitted, the host queues the next microbatch's input upload before waiting for current compute.
- After compute completes, the output slot is copied asynchronously to pinned host memory. It can remain in flight during the next microbatch's compute.
- A slot is not reused until its previous transfer completes. At layer boundaries, all copies are drained before swapping host activation buffers.

KDA computation remains sequential in token order. This pipelines transfers, not independent execution of causally dependent microbatches. Compute is still synchronized before graph reset; graph reuse and concurrent graph execution are not implemented.

For hidden size 4096, four streams, BF16 staging and microbatch 256, the device slots occupy 32 MiB and the pinned slots another 32 MiB, before backend padding. Both are charged to `max_transfer_bytes`; pinned bytes also count against `max_host_bytes`. Scheduler compute buffers, experts and persistent state are additional allocations.

Statistics report active pipelining, allocated transfer bytes, next-input prefetches and asynchronous output batches. Tests require these counts to cover every eligible microbatch, not merely numerical parity. They also cover insufficient transfer budgets and retry after failure. These counters establish that the asynchronous path executed, not a hardware-overlap percentage or an end-to-end speedup.

## Conditional next-layer expert prefetch

`weight_prefetch = true` enables optional routed-expert prefetch for a consecutive prefix of the next compute group. `max_prefetch_layers` is independent of the compute group size. This does not move KV/recurrent state between layers, and does not prefetch non-expert weights that the normal model configuration already manages.

After allocating the current layer's first microbatch graph, the implementation checks the next expert bank's actual tensor formats, dimensions, backend padding and alignment. Before attempting an additional device allocation, it checks:

- The bank fits both the normal per-layer expert limit and the separate `max_prefetch_bytes` limit (default 6 GiB).
- Reported free device memory can hold the bank, additional scheduler-buffer growth and `prefetch_headroom_bytes` (default 512 MiB).
- Host and transfer budgets can hold two additional pinned bounce buffers.
- The backend supports asynchronous copies, events and pinned storage.

Free VRAM already excludes current weights, persistent state, activation slots and allocated compute buffers. Those allocations are not subtracted again. Only potential additional compute-buffer growth is reserved. This is deliberately conservative: a large configured compute budget can disable prefetch even when a smaller measured workload would fit.

Manual execution reserves possible growth up to `max_compute_bytes`. Automatic plans use their preflighted device-workspace ceiling instead. If the checks fail, no additional device bank is allocated. Normal mandatory loading is retained. If allocation or pinned staging becomes unavailable after the check, a warning records the optional optimization's failure and normal loading is retained. Other model/compute errors are not suppressed.

An accepted prefetch owns a second device expert bank and a separate upload stream. It uploads weight slices through two pinned buffers, each at most 8 MiB. The host queues a bounded share of the upload after submitting each current-layer microbatch, before waiting for compute. Slice events prevent reuse of pinned memory that DMA still reads. At the next layer boundary, the upload is completed before the bank is used, and that bank replaces the normal upload rather than duplicating it.

Prefetch works with or without the activation pipeline. A partial next-group prefix is consumed first; missing group layers are then loaded normally. It never requires extra banks when space is insufficient. Pinned weight buffers count against both `max_host_bytes` and remaining `max_transfer_bytes`; all additional expert banks together count against `max_prefetch_bytes`. Backend-private workspace and other processes can still change memory availability, so headroom is not an allocation guarantee.

The example accepts:

- `--no-weight-prefetch` to disable the optimization.
- `--compute-budget-mib N` to set the scheduler-buffer limit and corresponding growth reservation.
- `--prefetch-budget-mib N` to limit the additional expert bank; zero forces skipping it.
- `--prefetch-headroom-mib N` to set reserved free VRAM beyond the compute-buffer growth.

Do not lower compute budgets only to force prefetch: an actual graph exceeding the compute limit still fails explicitly. In manual mode the default 8 GiB growth reservation normally skips prefetch on the local 4 GiB T1000. Automatic mode can admit a small synthetic model using its measured bound, not evidence that the real GLM model fits.

Statistics distinguish accepted prefetches, skips, device-allocation attempts, prefetched bytes, bytes queued between compute submission and synchronization, peak active extra-bank bytes and pinned weight-staging bytes. Submission counters do not measure actual copy/compute overlap or speedup.

GPU tests additionally cover a permitted prefetch, insufficient extra-bank budget, and an impossible free-memory headroom requirement. The skip cases require zero device-allocation attempts. They compare logits and subsequent decode, preserve once-per-layer expert byte counts, and retry after resource errors. Allowed prefetch is tested both with and without activation pipelining.

Local validation passed 48 CUDA cases for each of seeds 42 and 17 on the T1000, along with the registered CPU test and the three existing hybrid architecture checks. The standalone synthetic-GGUF example reported one accepted prefetch and 589824 prefetched bytes when given an appropriate small-model compute budget. With the extra-bank budget set to zero it reported one skip and zero allocation attempts. With pinned memory disabled and the activation pipeline explicitly off, it also skipped weight prefetch and completed normal layered execution and decode. These remain synthetic checks, not real GLM/200K capacity or speed measurements.

## Joint batch/group/prefetch planning

`auto_plan = true` jointly chooses one batch size for the request, a sequence of consecutive compute groups, and an independent prefetch prefix for each group. It does not change quantization, staging precision, KV types, model offload placement, or sequence semantics.

The implementation first collects bounded timing probes while the context is empty. It measures actual compute/setup time and activation/weight transfer throughput on the configured backend. Dense/MoE, KDA/DSA and default CPU/GPU layer placement form separate timing families. Probes cover the first occurrence of each family; later layers use that family's estimate. Context-local profiles are keyed by staging precision and batch-thread count, with model/backend/KV configuration fixed by the context. Incompatible or undersized profiles are not reused.

`max_calibration_ms` is a soft admission budget, not an interruptible deadline. A running probe may exceed it, and resource-rejected probes may need smaller batches before one succeeds. At most two promising schedules receive additional short validation when budget remains. Finalist probes spread weight uploads over the real request's batch count, so a short probe does not incorrectly assume that a long request must upload its whole next group in four steps.

Candidate batch sizes halve from the request's `n_ubatch` ceiling down to one. Group size is bounded by `max_group_layers`, actual expert layouts and optional `max_group_expert_bytes`. Every candidate group uses real backend expert allocations and dry graph measurement. Measurement reserves cache-span upper bounds while bounding pool counts and new/stale pools for the active prefix plus suffix, rather than pretending that every allocated cache cell must be re-pooled.

The cost model simulates compute, a shared H2D queue for activations and prefetched weights, D2H output, double-buffer reuse, dispatch and measured metadata replay. Prefix length affects the attention/setup estimate. Long traces use at most 256 representative batch slices to bound search work. This is a calibrated heuristic, not a guarantee of the fastest schedule. It cannot perfectly model routing skew, arbitrary per-tensor placements, kernel transitions, contention, page faults or changing clocks.

Prefetch is admitted only within current-group, future workspace, staging and safety budgets, and only when it improves estimated group latency beyond a one-percent threshold. Near-tied full schedules prefer smaller device footprints. Compute groups can differ in size; a prefetched prefix need not cover the whole next group.

The selected schedule is executed, not just returned as advice. Statistics include `selected_ubatch`, each group's range/resources/prefetch count, scored alternatives, probe count, profile reuse and planning/calibration time. Transfer counters describe the selected execution; calibration performs additional transfers. Include planning and calibration in cold-context end-to-end measurements. Actual graph allocation repeats the workspace check before execution. Batch size is fixed throughout replay; there is no mid-group retile or transparent recovery after partially updating KDA state.

```powershell
.\build-prefill-cuda\bin\llama-layered-prefill.exe --auto-plan --max-group-layers 4 --max-prefetch-layers 2 --no-weight-prefetch -m D:\models\glm-q4.gguf -f D:\prompts\input.txt --device CUDA0 -ngl 99 -c 200064 -ub 512 -fa on -ctk f16 -ctv f16 -n 32
```

This is an example of configuration, not a verified real-model capacity or speed claim. `--no-weight-prefetch` prioritizes grouped execution without allocating a second group.

## Resident prefix reuse

The caller must provide a context containing the complete state at the desired prefix endpoint. A successful native decode, successful layered prefill, or compatible full state restore can establish it. Set `reuse_prefix = true` and pass only the new tokens. The API appends positions; it does not find matching historical prompts, truncate KDA history, or reconstruct state from a token string.

Validation requires contiguous sequence-0 positions, matching MLA/indexer/recurrent endpoints, matching attention/indexer cell maps, ready recurrent sources, no pending stream/shift updates, and no shared or other active sequence cells. The caller remains responsible for model/settings/token provenance. Shape/position validation cannot prove that arbitrary externally modified tensors contain the correct state.

All persistent tensor buffers remain at their existing configured locations. Prefix replay copies host metadata only; it does not save prefix KV/state tensors to CPU or disk. It preserves incomplete k-pool tails, stale markers, cached keys/gates/pooled rows, KDA matrices and convolution windows. Full-state restoration can legitimately require re-pooling stale cached indexer entries, just as the native path does; prefix transformer layers are not recomputed.

A live prefix is never used for destructive timing calibration. Compatible existing timing profiles can guide joint planning. Without them, the planner explicitly reports `capacity_only`: it searches resident candidates, preserves the largest feasible batch first, then reduces group-boundary count, and disables unmeasured prefetch. If calibrated candidates do not fit, it also tries the smaller resident-only batch candidates before reporting failure. A capacity plan has no latency estimate and is not claimed optimal.

The example's `--prefix-tokens N` tokenizes the complete input once, seeds the context with its first N tokens, then appends the remainder with prefix reuse in the same process. This avoids token-boundary mismatches and demonstrates state reuse; it is not cross-process caching and its total runtime includes seeding. Statistics distinguish reused prefix length from evaluated suffix length.

Admission or planning failures preserve existing prefix data. A failure after kernels update state sets `prefix_invalidated`, clears the invalid context and returns an error. No atomic KDA rollback is implemented. The caller must restore/recompute a valid prefix rather than retrying the suffix against invalid state.

## Resident capacity and explicit shortfall

On the selected execution device, persistent model weights and allocated KV/indexer/recurrent/convolution buffers are reported separately. They are already reflected in free VRAM and are not subtracted again. A candidate additionally needs current-group experts, device workspaces, activation transfer slots, optional prefetched banks and reserved headroom. Transfer-slot sizing uses the backend's allocation requirements, including padding. Host activation and pinned-staging budgets are checked independently.

`--device-headroom-mib` is an alias of `--prefetch-headroom-mib` and controls the common resident-planning/prefetch safety reserve. It applies even when prefetch is disabled.

If no resident candidate fits, diagnostics report resident model/state bytes, free VRAM at planning time, the minimum planned extra requirement, safety margin, capacity shortfall and configured-budget shortfall. If a bank cannot even be allocated for measurement, its missing workspace makes the reported gap a lower bound; the flag and log say so. Measured graph requirements are conservative reservations, not a proof that every backend-private allocation is known.

No plan automatically evicts, swaps or changes KV/state placement. A reported state-buffer size versus gap can inform a later design discussion, but does not establish that eviction alone would solve it. KV eviction and swap-aware planning are not implemented.

This execution planner runs after model/context creation. Base resident allocations must first succeed under the normal llama.cpp configuration. If the model or KV/state cannot be allocated at initialization, the existing backend reports that failure and the example stops explicitly; this API is not a pre-load capacity estimator and does not silently move state elsewhere.

## Additional validation

```powershell
ctest --test-dir build-prefill -R "^(test-alloc|test-layered-prefill|test-layered-planner|test-layered-prefix)$" --output-on-failure
.\build-prefill-cuda\bin\test-llama-archs.exe --layered-planner --seed 42 --backend CUDA0
.\build-prefill-cuda\bin\test-llama-archs.exe --layered-prefix --seed 42 --backend CUDA0
```

Grouped tests compare one-, two-, three- and four-layer execution, including tail batches and F32/BF16 staging. Joint-plan tests verify actual execution at the selected batch size, CPU F32 serialized state, subsequent decode, context-local profile reuse, group budgets and non-destructive timing-profile reuse for a live prefix.

Resident-prefix tests cover native and restored prefixes of 3 and 33 tokens, 19-token suffixes, incomplete pool boundaries, partial next-group prefetch, activation pipeline on/off, F16 and matched Q8_0 K/V caches, and both staging types. They compare final logits, full CPU F32 state and subsequent native decode. They reject inconsistent recurrent state, shared cells and orphaned recurrent state behind an empty attention cache, preserve prefixes on admission/capacity failures, and explicitly test invalidation after partial execution.

Planning exposed a size-only allocator reservation bug: an allocation plan could exist without backing buffers on a fresh scheduler. The allocator now requests reallocation in that case. `test-alloc` includes fresh and growing size-only reservation regressions, and CPU/CUDA planner tests exercise real scheduler allocation after measurement.

The final local runs passed the four registered CPU tests above, 48 single-layer/prefetch CUDA cases, 36 grouped CUDA cases, calibrated joint-planner tests with seeds 42 and 17, and all 32 resident-prefix combinations for each seed on the T1000. The existing CUDA `glm5-next`, `qwen4exp` and `kimi-linear` architecture checks also passed. Error logs from deliberate budget/state rejection cases are expected; all test processes exited successfully.

An MSVC AddressSanitizer build using `/fsanitize=address` and `/EHsc` passed these targeted CPU commands:

```powershell
.\build-prefill-asan\bin\test-alloc.exe
.\build-prefill-asan\bin\test-llama-archs.exe --joint-only --seed 42
.\build-prefill-asan\bin\test-llama-archs.exe --layered-prefix --seed 42
```

The standalone generated two-layer GGUF also passed automatic BF16 and manual two-layer F32 prefix reuse followed by ordinary decode. This tested invocation reused three tokens and evaluated the remaining six:

```powershell
$env:CUDA_VISIBLE_DEVICES = '2'
.\build-prefill-cuda\bin\llama-layered-prefill.exe --auto-plan --prefix-tokens 3 --compute-budget-mib 64 --device-headroom-mib 64 -m .\build-prefill\tests\test-models\glm5-next-moe.gguf --device CUDA0 -ngl 99 -c 256 -ub 7 -p "A small synthetic prompt for layered prefill." -n 2
```

These are synthetic correctness/resource-policy checks on the local CPU and SM75 GPU. They do not establish real GLM-5.3-Flash accuracy, SM86/20GB capacity, 200K speed or tuning optimality.

## Implementation files

| Surface | Files |
| --- | --- |
| Experimental API and execution | `src/llama-ext.h`, `src/llama-context-prefill.cpp`, `src/llama-context.h`, `src/llama-context.cpp` |
| Cost model and build registration | `src/llama-prefill-plan.h`, `src/llama-prefill-plan.cpp`, `src/CMakeLists.txt` |
| Grouped model graphs | `src/models/glm5-next.cpp`, `src/llama-graph.h` |
| Complete-prefix metadata replay | `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-memory-recurrent.h`, `src/llama-memory-recurrent.cpp`, `src/llama-memory-hybrid-idx.h`, `src/llama-memory-hybrid-idx.cpp` |
| Allocation fix and tests | `ggml/src/ggml-alloc.c`, `tests/test-alloc.cpp`, `tests/test-llama-archs.cpp`, `tests/CMakeLists.txt` |
| Runnable example and documentation | `examples/layered-prefill/layered-prefill.cpp`, `docs/development/layered-prefill.md` |

## Reference

- [Official GLM-5.3-Flash configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)
- [ik_llama.cpp expert cache](https://github.com/ikawrakow/ik_llama.cpp/blob/main/src/llama-moe-cache.cpp)
- Local implementation surfaces: `src/models/glm5-next.cpp`, `src/llama-context.cpp`, `src/llama-memory-hybrid-idx.cpp`, `src/llama-memory-recurrent.cpp`, `src/llama-moe-cache.cpp`.

The ik fork's cache uses layout-compatible device banks, expert slot maps and upload callbacks. Its LRU policy alone does not implement layer-major whole-input prefill. Reuse upstream GGUF and backend operations rather than importing fork-specific quantization formats or kernels without compatibility checks.
