# Experimental layer-major prefill for GLM5-Next and Qwen3.5 MoE

This document describes the default expert-staging implementation. The [target layered-inference design and opt-in prototype](layered-inference-design.md) instead cover complete layer staging, upload-free complete-layer capacity planning, CPU-owned target state, and GPU MTP drafting with CPU verification. The expert-only latency planner and complete-layer capacity planner have different admission and calibration rules.

## Goal and scope

Process a text prompt or appended suffix one contiguous layer group at a time. Keep the group's complete routed experts on the device while processing all input microbatches in position order. Only then replace them with the next group's weights. Group size one is the original layer-major path. This is not ordinary chunked prefill and is not a larger decode expert cache.

The initial target is GLM-5.3-Flash, quantized GGUF, EPYC 7532 (AVX2), 256 GB RAM and a 20 GB SM86 GPU. Development on Windows has no target model. Real-model GLM accuracy and a default-path speedup have not been established. Separate [full-layer GLM experiments](layered-inference-design.md#completed-200k-experiment) completed a synthetic 200K-token workload; they are not a representative quality benchmark or a controlled grouping comparison.

The path is explicit opt-in. Default decode remains unchanged. It accepts sequence 0, consecutive text positions starting at zero, and the final token's logits. With `reuse_prefix = true`, the supplied tokens are only the suffix of the context's complete resident prefix state. Multimodal input batches, speculative contexts, adapters, shared-indexer variants, pending sequence edits and concurrent sequences remain unsupported.

## Execution

1. Validate the input, cache endpoints and configured resource limits. Automatic planning preflights every selected group before updating model state.
2. Allocate one bounded host activation buffer for this call's tokens only. GPU execution requires the complete buffer to be pinned; CPU correctness tests use ordinary host memory.
3. For each group, make its complete routed expert tensors device-resident, build a multi-layer graph and process every microbatch in position order.
4. Keep internal group activations on the device. At group boundaries, read each microbatch's input slice and overwrite the same host slice with its output after compute finishes. GLM preserves `[hidden_size, 4, suffix_tokens]`; Qwen preserves `[hidden_size, suffix_tokens]`. Never collapse GLM's streams before the output head.
5. Preserve every layer's final MLA, indexer, convolution and recurrent state for subsequent ordinary decode.
6. Run the output head for the final token and expose the usual logits.

The implementation snapshots and replays only host metadata: attention/indexer cell maps, allocation heads, recurrent cell/source maps, rollback indices, k-pool layout and stale markers. Tensor data is not copied out for replay. Each group starts from the same prefix metadata and appends the same suffix positions; previously completed groups' tensor contents remain intact. Graph synchronization precedes metadata replay and weight replacement.

This preserves each layer's own prefix MLA, indexer key/gate/pooled rows, KDA matrix and Q/K/V convolution window. It does not substitute a token match or one hidden-state vector for the complete hybrid state. Metadata snapshots use additional ordinary host memory outside the activation-buffer budget.

The official GLM-5.3-Flash configuration has full indexers at every attention layer. This implementation rejects shared-indexer variants explicitly. Supporting them would additionally need compact selected indices and visibility information across layers. Saving a dense whole-prompt selection mask is prohibited because it grows quadratically.

## Precision and resource budgets

Use BF16 for host activation staging by default, with F32 as a correctness reference. Conversion must happen on the device to reduce PCIe traffic. This does not change the precision of KDA recurrent state, mHC computation, attention masks or arithmetic inside existing operators.

For BF16 staging, internal group boundaries still round to BF16 and convert back to F32 on the device. Increasing group size therefore removes PCIe round trips, not the specified rounding points. The planner never changes staging or KV precision to make a plan fit.

BF16 staging is not lossless. In particular, rounding can change the DSA top-k selection. An exploratory sparse miniature model with seed 17 produced final-logit NMSE about `8.9e-3` relative to unrounded F32. This is not a real-model quality measurement, but it rules out claiming that BF16 staging always preserves the original logits closely. Evaluate BF16 on representative real prompts before using it for quality-sensitive work; select F32 staging to isolate scheduling correctness.

GLM restores pooled indexer keys to F16 only with F32 activation staging. BF16 staging keeps the reference F32 indexer dispatch: the F16 dispatch combined with BF16 boundaries exceeded the existing same-backend logits tolerance at microbatch 97. Windowed indexer execution and compact visibility inputs remain enabled in both staging modes.

For hidden size 4096, four streams and microbatch 256, one BF16 boundary block is 8 MiB in each direction. At 200000 newly evaluated tokens, the single complete BF16 activation buffer requires about 6.1 GiB of host RAM. One million newly evaluated tokens require about 30.5 GiB. These sizes exclude all other allocations and do not depend on microbatch size. An already cached prefix does not contribute activation rows.

The routed experts of one layer contain `3 * 4096 * 2048 * 288` parameters. If all three tensors use Q4_K, their storage is about 3.80 GiB. Mixed GGUF quantization can require more. Check actual tensor sizes and backend operator support, not the file's Q4 label.

The complete working set includes current-layer weights, persistent device caches, activation buffers, masks, operator workspace, backend overhead and safety headroom. Do not double-count allocations already reflected in device free memory.

The experimental API separately limits the single host activation buffer plus pinned weight staging, the current expert buffer including backend padding, and the sum of scheduler compute buffers. The last limit is not a total-VRAM limit: persistent model/context allocations and backend-private kernel workspaces remain additional costs. Backend allocation failures are errors, not fallbacks. Locking the full activation buffer removes staging copies but makes that physical RAM unavailable for paging; keep the host limit below what the machine can safely pin.

Start evaluation at microbatch 256, compare 512, and consider larger batches only after checking real graphs. These are candidates, not universal capacity limits. Manual execution uses `n_ubatch` and `group_layers`; `auto_plan` searches smaller batches and variable-length groups. The selected batch size remains fixed for the entire request, so metadata replay never silently changes chunk boundaries. Admitted F32 complete-layer execution reuses the planner's workspace bound and checks actual allocated scheduler buffers before compute; other paths retain per-graph measurement.

Admission/planning failures preserve an existing prefix. Once model execution has updated state, a failure can invalidate the whole context because KDA cannot be rolled back by deleting token positions. `prefix_invalidated` explicitly reports this case; the implementation logs it, returns an error and does not pretend that the old prefix can still be reused.

Flash Attention and a suitable fused indexer are important at long context. With 200000 keys and 1024 queries, one dense F32 attention score tensor across 64 heads is about 48.8 GiB. A BF16 staging buffer does not reduce that workspace. Unsupported fused operations must not be disguised as successful GPU acceleration.

If a complete layer's experts and required persistent state cannot fit, shrinking the microbatch cannot solve the fixed-memory problem. Report it instead of silently streaming experts repeatedly.

## Execution timing

Both examples report aggregate and per-group `prefill wall ms` from `stats.timing` and `group.timing`. `total_ms` measures selected execution, including host activation allocation and cleanup, but excludes planning/calibration; add `planning_ms` for the complete automatic-prefill phase. Failed requests do not expose partial execution timing as a successful result.

The phases are host-wall intervals: `setup_ms` covers initial synchronization, metadata snapshot, host/pipeline buffers, batch construction and initial scheduler/output reservation; `memory_ms` covers group metadata replay, batch preparation, memory apply and microbatch advance; `weight_ms` covers weight-bank allocation/loading and next-bank prefetch setup; `state_upload_ms` and `state_download_ms` cover complete-layer state-bank allocation/upload and state download; `graph_ms` covers graph construction, workspace measurement, rebuilding, placement, checks and allocation; `inputs_ms` covers `set_inputs`, including CPU mask/index generation; `activation_upload_ms` and `activation_download_ms` cover explicit activation transfer calls and pipeline slot waits/drains; `output_ms` covers logits and MTP hidden-row downloads; `cleanup_ms` covers graph reset and resource release. `other` is total wall time minus the accounted phases, including bookkeeping between intervals.

`compute_ms` is compute submission through the existing synchronization, including any overlapping activation/weight prefetch submissions in that interval. It is not CUDA-event kernel time. With pipelining, transfer fields measure host call/wait time, not independent device DMA duration; some transfer completion is charged to compute/wait. No extra device synchronization or events are added for timing. Phase wall intervals do not overlap, but they cannot establish actual GPU copy/compute overlap. `dispatch_ms` remains a planner dry-graph measurement, not execution `graph_ms`.

`graph_build_ms`, `graph_measure_ms` and `graph_alloc_ms` subdivide `graph_ms`; do not add them to the phase total again. Counters distinguish graph builds, compatible graph reuse and execution-time dry measurements. Compatible graphs use the ordinary reuse guards, including recurrent positions, layer weights, outputs and pipeline slots. Admitted F32 full-layer plans avoid the measure/rebuild cycle even when shapes cannot be reused. Their allocated bound includes separately retained device and host scheduler maxima. BF16 and expert-only paths keep the existing measurement policy.

The next prototype additionally reports `workspace_reserve_ms` within `graph_ms`, peak reserved workspace, maximum backend segment count, actual indexer query-window width, and compact causal/pool input flags. Boundaries identify potential stable CUDA graph segments; they do not count successful CUDA graph replays. See the [pending-validation record](layered-inference-design.md#workspace-query-windows-visibility-and-segmented-graphs-validation-pending) before interpreting these changes as measured GPU improvements.

Accounting checks cover CPU/CUDA execution, synchronous and pipelined transfers, expert prefetch, complete-layer plans and prefix reuse. The real-model timing smoke test and deterministic-planning comparison are recorded in the [complete-layer design](layered-inference-design.md#deterministic-planner-smoke-test). These short-input tests are not a 200k phase profile.

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

The defaults are BF16 staging, microbatch 256, 16 GiB of host activation storage including pinned weight staging, 6 GiB of current-layer expert storage, and 8 GiB of scheduler compute buffers. `--host-budget-mib` sets the host limit. The 512 MiB transfer budget covers activation device slots and pinned weight staging, including optional weight-prefetch bounce buffers. The complete pinned activation buffer counts against the host limit, not the microbatch transfer limit. C++ callers can set these limits through `llama_layered_prefill_params`. CPU execution requires `require_gpu = false` and is intended for correctness testing. Only one GPU is supported.

The separate `llama-layered-prefill` example uses the shared llama.cpp argument parser, keeps routed experts in plain CPU buffers, invokes the new path, reports copy statistics and then performs greedy ordinary decode. `--staging-type f32` selects the unrounded reference; the default is `bf16`. It raises the logical batch capacity to hold the complete prompt; `-ub` remains the physical microbatch bound. It does not enable the feature in the normal CLI or server.

```powershell
.\build-prefill-cuda\bin\llama-layered-prefill.exe -m D:\models\glm-q4.gguf -f D:\prompts\input.txt --device CUDA0 -ngl 99 -c 200064 -ub 256 -fa on -ctk f16 -ctv f16 -n 32
```

The command requires a suitable real GGUF and is not a verified 200K/20GB configuration. Expert tensors must be contiguous, host-resident and in their ordinary GGUF layout. Repacked CPU-only expert buffers are rejected. The scheduler must place expert operations and staging conversions on the selected backend; otherwise the call fails instead of presenting a CPU fallback as GPU acceleration.

For a fixed two-layer group, add `--group-layers 2`. For joint automatic planning, add `--auto-plan --max-group-layers 4`; `-ub` becomes the search ceiling. `--group-budget-mib` optionally caps all current-group experts, independently of the per-layer expert limit. `--max-prefetch-layers` is a separate ceiling and need not equal the compute group size. No change is enabled in the normal CLI or server.

## Microbatch activation pipeline

Single-layer groups remain the default. Multiple-layer groups reduce activation transfers, but do not reduce once-per-layer weight uploads. If the pipeline already hides those transfers behind compute, larger groups need not improve throughput and can reduce the microbatch that fits. Compare groups at a fixed microbatch before attributing a speedup to grouping. Full-layer GLM offload and its capacity planner are explicit opt-in experiments described in [the layered-inference design](layered-inference-design.md); they do not replace Qwen streaming or enable Qwen automatic planning.

The GLM full-layer path defers full-model compute-buffer reservation. The full-layer weight/state allocation-reuse experiment was removed because the local 200k comparison did not show a throughput benefit; each group now allocates and releases its own banks. Qwen's expert-bank reuse, pinned-staging and worker paths are unchanged. Context/model initialization and prefill wall times are reported separately. The [GLM comparison](layered-inference-design.md#deferred-initialization-and-full-layer-bank-reuse-2026-10-30) preserves both negative timings and the unchanged baseline's logits repeatability limitation; the [keep/remove ledger](layered-inference-design.md#current-glm-keepremove-decisions-2026-10-31) distinguishes measured benefits from unproven speedups.

On an asynchronous GPU backend with events and pinned host buffers, `pipeline = true` (the default) uses two device input slots and two device output slots. One complete host activation buffer is pinned and updated in place. DMA reads and writes its microbatch slices directly, with no activation bounce buffers or host `memcpy`. Failure to obtain real pinned storage is an error, not a silent pageable substitute. `--no-pipeline` selects synchronous transfers but still uses one complete pinned buffer on a GPU.

Each slot owns persistent device storage outside the scheduler's graph buffers. A graph reads its input slot and writes its output slot directly, so resetting the graph cannot invalidate an in-flight activation copy.

- Upload and download use separate backend streams from compute.
- Before compute consumes an input slot, its stream waits for that slot's upload event.
- Once a microbatch has been submitted, the host queues the next microbatch's input upload before waiting for current compute.
- After compute completes on all participating backends, the output slot is copied asynchronously over the same host slice that supplied its input. That slice's upload has already completed; the next input upload reads a disjoint slice.
- A device output slot is not reused until its previous download completes. At group boundaries, all transfers are drained before the next group reads the overwritten host buffer. Error cleanup also drains the streams before releasing host storage.
- Input lookahead reads the next prepared hybrid microbatch without advancing or applying memory state. It uses the actual token count, including a shortened penultimate microbatch that keeps the recurrent rollback tail intact.

KDA computation remains sequential in token order. This pipelines transfers, not independent execution of causally dependent microbatches. Compute is still synchronized before graph reset. Full-layer execution can reuse compatible graphs; concurrent graph execution is not implemented.

For hidden size 4096, four streams, BF16 staging and microbatch 256, the device slots occupy 32 MiB before backend padding, charged to `max_transfer_bytes`. There is no additional 32 MiB of pinned activation slots. The full pinned buffer is charged once to `max_host_bytes`. Scheduler compute buffers, experts and persistent state are additional allocations.

Statistics report `activation_host_bytes`, `activation_pinned_bytes`, active pipelining, allocated device-slot/weight-staging transfer bytes, next-input prefetches and asynchronous output batches. Tests require exactly one suffix-sized activation allocation, no activation rows for the prefix, and transfer counts covering every eligible microbatch. They also cover a host budget that fits only one full buffer, insufficient transfer budgets and retry after failure. These counters establish that the asynchronous path executed, not a hardware-overlap percentage or an end-to-end speedup.
In full-layer mode, `max_transfer_bytes = 0` (example option `--transfer-budget-mib 0`) sizes the transfer ceiling from the requested batch and actual backend padding. It does not bypass host/device admission or headroom. Explicit nonzero budgets retain their existing hard limits. Full host activations count only against the host budget, not the device-slot transfer budget. Short suffixes still reserve a legal nominal rollback batch.

The final group does not export its complete residual to the host or to an output pipeline slot. It still computes all required state, final logits and requested MTP hidden rows. For a complete invocation with G groups, activation uploads and downloads each cover G-1 complete residual buffers; a single-group invocation needs neither transfer. With rollback disabled, each noninitial group prefetches every microbatch after its first, and each nonfinal group downloads every microbatch. The same rules apply to the actual variable-size splits when rollback is enabled. F32 staging at 200000 tokens, hidden size 4096 and four streams saves about 12.2 GiB of final-group downloads.

## Conditional next-layer expert prefetch

`weight_prefetch = true` enables optional routed-expert prefetch for a consecutive prefix of the next compute group. `max_prefetch_layers` is independent of the compute group size. This does not move KV/recurrent state between layers, and does not prefetch non-expert weights that the normal model configuration already manages.

After allocating the current layer's first microbatch graph, the implementation checks the next expert bank's actual tensor formats, dimensions, backend padding and alignment. Before attempting an additional device allocation, it checks:

- The bank fits both the normal per-layer expert limit and the separate `max_prefetch_bytes` limit (default 6 GiB).
- Reported free device memory can hold the bank, additional scheduler-buffer growth and `prefetch_headroom_bytes` (default 512 MiB).
- Host and transfer budgets can hold two additional pinned bounce buffers.
- The backend supports asynchronous copies, events and pinned storage.

Free VRAM already excludes current weights, persistent state, activation slots and allocated compute buffers. Those allocations are not subtracted again. Only potential additional compute-buffer growth is reserved. This is deliberately conservative: a large configured compute budget can disable prefetch even when a smaller measured workload would fit.

Manual execution reserves possible growth up to `max_compute_bytes`. Automatic plans use their preflighted device-workspace ceiling instead. If the checks fail, no additional device bank is allocated. Normal mandatory loading is retained. If allocation or pinned staging becomes unavailable after the check, a warning records the optional optimization's failure and normal loading is retained. Other model/compute errors are not suppressed.

An accepted prefetch owns a second device expert bank and a separate upload stream. The baseline uploads weight slices through two pinned buffers, each at most 8 MiB, with the host queuing a bounded share after each current-layer microbatch submission. Slice events prevent reuse of pinned memory that DMA still reads. Streamed complete banks on CUDA instead automatically use the worker and eligible direct source uploads described below. At the next layer boundary, the upload is completed before the bank is used, and that bank replaces the normal upload rather than duplicating it.

Prefetch works with or without the activation pipeline. A partial next-group prefix is consumed first; missing group layers are then loaded normally. It never requires extra banks when space is insufficient. Pinned weight buffers count against both `max_host_bytes` and remaining `max_transfer_bytes`; all additional expert banks together count against `max_prefetch_bytes`. Backend-private workspace and other processes can still change memory availability, so headroom is not an allocation guarantee.

The example accepts:

- `--no-weight-prefetch` to disable the optimization.
- `--compute-budget-mib N` to set the scheduler-buffer limit and corresponding growth reservation.
- `--prefetch-budget-mib N` to limit the additional expert bank; zero forces skipping it.
- `--prefetch-headroom-mib N` to set reserved free VRAM beyond the compute-buffer growth.

Do not lower compute budgets only to force prefetch: an actual graph exceeding the compute limit still fails explicitly. In manual mode the default 8 GiB growth reservation normally skips prefetch on the local 4 GiB T1000. Automatic mode can admit a small synthetic model using its measured bound, not evidence that the real GLM model fits.

Statistics distinguish accepted prefetches, skips, device-allocation attempts, prefetched bytes, bytes queued between compute submission and synchronization, peak active extra-bank bytes and pinned weight-staging bytes. Submission counters do not measure actual copy/compute overlap or speedup.

GPU tests additionally cover a permitted prefetch, insufficient extra-bank budget, and an impossible free-memory headroom requirement. The skip cases require zero device-allocation attempts. They compare logits and subsequent decode, preserve once-per-layer expert byte counts, and retry after resource errors. Allowed prefetch is tested both with and without activation pipelining.

Before the full pinned activation change, local validation passed 48 CUDA cases for each of seeds 42 and 17 on the T1000, along with the registered CPU test and the three existing hybrid architecture checks. The standalone synthetic-GGUF example reported one accepted prefetch and 589824 prefetched bytes when given an appropriate small-model compute budget. With the extra-bank budget set to zero it reported one skip and zero allocation attempts. That version also allowed synchronous pageable activations when pinning was disabled. The current GPU path instead requires full pinned activation storage even with `--no-pipeline`; pinning failure is an explicit error. These remain synthetic checks, not real GLM/200K capacity or speed measurements.

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

Both GLM5-Next and Qwen3.5 MoE support resident prefix reuse. The caller must provide a context containing the complete state at the desired prefix endpoint. A successful native decode, successful layered prefill, or compatible full state restore can establish it. Set `reuse_prefix = true` and pass only the new tokens. The API appends positions; it does not find matching historical prompts, truncate recurrent history, or reconstruct state from a token string. Native llama.cpp already supports appending to a live context; these checks apply to this separate layer-major execution order and do not change native decode or the server's prompt cache.

Shared validation requires contiguous sequence-0 attention positions, matching attention/recurrent endpoints, ready recurrent sources, no pending stream/shift updates, and no shared or other active sequence cells. GLM additionally requires matching indexer endpoints and attention/indexer cell maps. The caller remains responsible for model/settings/token provenance. Shape/position validation cannot prove that arbitrary externally modified tensors contain the correct state.

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

Resident-prefix tests cover both architectures, native and restored prefixes of 3 and 33 tokens, 19-token suffixes followed by native decode and another 9-token suffix, incomplete GLM pool boundaries, partial next-group prefetch, activation pipeline on/off, F16 and matched Q8_0 K/V caches, and both staging types. Qwen covers complete banks, grouped execution and a route-driven bank with streamed non-expert weights on CUDA. Tests compare final logits, full CPU F32 state and subsequent native decode. They reject inconsistent recurrent state, shared cells and orphaned recurrent state behind an empty attention cache, preserve prefixes on admission/capacity failures, and explicitly test invalidation after partial execution.

Planning exposed a size-only allocator reservation bug: an allocation plan could exist without backing buffers on a fresh scheduler. The allocator now requests reallocation in that case. `test-alloc` includes fresh and growing size-only reservation regressions, and CPU/CUDA planner tests exercise real scheduler allocation after measurement.

The earlier GLM-only revision passed the four registered CPU tests above, 48 single-layer/prefetch CUDA cases, 36 grouped CUDA cases, calibrated joint-planner tests with seeds 42 and 17, and all 32 resident-prefix combinations for each seed on the T1000. The existing CUDA `glm5-next`, `qwen4exp` and `kimi-linear` architecture checks also passed. Error logs from deliberate budget/state rejection cases are expected; all test processes exited successfully.

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

## Qwen3.5 MoE route-driven prefill

Qwen3.5 MoE supports manual layer groups and a single-sequence text context, initially empty or extended with `reuse_prefix`. Its group boundaries contain `[hidden_size, suffix_tokens]` activations, not GLM's four mHC streams. Each group replays the attention and recurrent allocation metadata without copying or rolling back cache tensors. Layers consume microbatches in position order. Ordinary decode can continue from the resulting KV, convolution and DeltaNet state. Automatic planning remains GLM-only; Qwen requests for it are rejected explicitly. Prefix reuse works with complete banks, the route-driven bank and `--stream-weights`; it does not require future-route prediction.

With `--expert-slots 0` (the default), complete routed expert banks are uploaded once per layer. This reference supports multiple-layer groups and optional complete-bank next-group prefetch. Both merged gate/up and separate gate/up tensors are supported. Non-expert weight placement still follows ordinary model offload settings.

`--expert-slots N` enables a compact, single-layer cache. Only experts selected by the actual router are uploaded, and resident experts are reused across microbatches. A FIFO replacement policy protects every expert needed by the current microbatch. Router probabilities and per-expert scales retain their original expert indices; only weight lookup uses compact slot indices. The cache does not predict future routes.

The selected expert union of each microbatch must fit the configured slots. Exhaustion reports an error and clears partial context state; it does not silently omit experts, change the batch size or fall back to CPU. Choosing at least `top_k * n_ubatch` slots, capped at the model's expert count, guarantees capacity independently of routing. Fewer slots are an explicit workload-dependent choice.

Routing callbacks match the original router and shared-branch tensors exactly, not derived contiguous views with the same name prefix. Returning false from an evaluation callback aborts the entire scheduled graph, including later backend splits; resource errors must not reach expert kernels with invalid slots.

On CUDA with pipelining enabled, missing experts are staged in bounded pinned host memory and uploaded on a separate stream after routing. The graph schedules the shared expert branch between routing and routed-expert evaluation. A transfer event is synchronized after the shared branch, before slot lookup or routed-expert kernels can consume the new weights. Previous microbatch computation and transfers finish before slots or staging storage are reused. `--no-pipeline` selects synchronous uploads and activation transfers.

This overlaps eligible weight transfer with the current microbatch's shared expert work, not concurrent cross-layer or cross-microbatch graph execution. Shared expert work can execute on CPU when non-expert weights are CPU-resident. Asynchronous byte and overlap-window counters establish that this scheduling path ran; they do not measure physical overlap or prove a speedup.

Router probabilities, selected-weight normalization, shared FFN arithmetic and layer residual output use the layer's configured backend in both reference and route-driven execution. Changing branch order must not silently move these operations between CPU and GPU. Real-model checks exposed a one-ULP difference in selected-weight normalization that amplified across layers; expert matrices and selected IDs were identical. Keep routing arithmetic placement fixed instead of relaxing the final-logit tolerance.

Exact route-driven mode disables complete-bank next-layer prefetch: the next layer's routes are not known yet. The complete-bank reference retains that option. Neither mode claims to know later microbatches' routes.

The expert budget includes the compact bank and its device slot map. Pinned expert staging counts against the remaining host and transfer budgets. Statistics separate peak bank bytes, cache hits/misses, actual uploaded bytes and pinned staging. Manual execution checks available device memory and safety headroom before allocating a group; this is a snapshot, not a guarantee against concurrent allocations by other processes.

For a small-memory GPU, begin with non-expert weights on CPU and microbatch 8. Both Qwen3.5 models select eight routed experts per token, so 64 slots suffice for any routing at this batch size:

```powershell
$env:CUDA_VISIBLE_DEVICES = '2'
.\build-prefill-cuda\bin\llama-layered-prefill.exe -m D:\AI\Qwen3.5-35B-A3B\Qwen3.5-35B-A3B-Q4_K_M.gguf --device CUDA0 -ngl 0 -c 512 -ub 8 -b 512 -f D:\prompts\input.txt -n 0 -t 10 -tb 10 --staging-type f32 --expert-slots 64 --compute-budget-mib 256 --group-budget-mib 1536 --device-headroom-mib 128
```

For the 122B model, use the first GGUF shard with all shards in the same directory. Keep context length, prompt, microbatch, staging precision and non-expert placement identical when comparing complete groups and route-driven execution. The example reports prefill wall time excluding model/context creation, as well as expert upload and residency statistics. Board-memory samples include other processes and are not equivalent to process-exclusive VRAM measurements.

Use an explicit group budget on Windows: CUDA's free-memory report alone does not reliably exclude WDDM residency migration into shared memory. A successful allocation above physical VRAM availability is not a valid GPU-only performance result. `--logits-file PATH` writes final-token F32 logits for direct comparisons of successful real-model runs.

Size-only scheduler reservation preserves explicit backend assignments, like ordinary reservation. Otherwise measurement could place staging conversions on CPU while execution places them on GPU, invalidating both admission estimates and the synchronous reference. Mixed CPU non-expert/GPU expert tests cover this case.

Qwen correctness tests cover F32/Q4_K experts, F32/BF16 activation staging, partial microbatches, synchronous/asynchronous operation, cache hits and replacement, slot exhaustion, resource limits, final state and subsequent decode. The CUDA candidate keeps non-expert weights on CPU to cover the small-memory placement used above.

```powershell
.\build-prefill-cuda\bin\test-llama-archs.exe --qwen-prefill --seed 42
.\build-prefill-cuda\bin\test-llama-archs.exe --qwen-prefill --seed 42 --backend CUDA0
```

### All-layer streaming and large-microbatch evaluation

`--stream-weights` keeps every model tensor in ordinary CPU buffers. The loader's logical GPU layer assignment is used only to place KV/recurrent state and select GPU execution; it does not make model weights GPU-resident. Embedding lookup and the final single-token output head remain on the host-backed path. All transformer-block weights, including attention, DeltaNet, router, shared experts and scales, are temporarily supplied to the current layer graph.

Non-routed layer weights are packed with backend alignment into one pinned host buffer and uploaded with one bulk copy per layer. Known next-layer static weights can be prefetched without knowing routes. Routed experts use bounded double-buffered pinned uploads. Compatible device expert banks are recycled instead of allocated and freed at every layer; their contents are overwritten, and the banks are freed when the prefill request ends. Complete-bank mode may prefetch the next complete expert bank when admitted. Route-driven mode still waits for actual expert IDs.

`llama_init_from_model_layered` initializes persistent state, fused-operation capability checks and scheduler metadata without reserving full-model prompt compute buffers. Actual layer graphs are measured and allocated later. Ordinary initialization would otherwise reserve a full-model worst-case graph before the layer-major planner can use that memory for its own working set.

`--device-budget-mib` caps the accounted GPU working set: persistent state, current/prefetched/recycled weight banks, activation device slots and the larger of measured or already allocated device scheduler buffers. It excludes driver/context overhead and backend-private kernel allocations, so leave safety headroom. `--static-budget-mib` bounds current plus next static banks, while `--transfer-budget-mib` separately bounds activation slots and pinned staging. Optional prefetch yields to these budgets; an infeasible requested microbatch is rejected, not silently shrunk or run through a CPU fallback.

Keep the CPU model genuinely loaded for throughput evaluation (`--load-mode none`) and provide enough host RAM. Use a long input, not the earlier 131-token capacity check. A context sweep keeps the same host model loaded while creating a fresh state/context for every requested microbatch. It logs each success/failure and returns status 2 if any sweep point fails. `--logits-file` appends `.ubN` during a sweep so comparisons can use the same microbatch and cache precision.

```powershell
$env:CUDA_VISIBLE_DEVICES = '2'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
.\build-prefill-cuda\bin\llama-layered-prefill.exe --stream-weights --load-mode none -m D:\AI\Qwen3.5-35B-A3B\Qwen3.5-35B-A3B-Q4_K_M.gguf --device CUDA0 -c 4096 -b 4096 -n 0 -t 10 -tb 10 -fa on -f D:\prompts\long-input.txt --prompt-tokens 4096 --sweep-ubatch 128,256,512,1024,2048 --staging-type f32 --compute-budget-mib 768 --group-budget-mib 1536 --static-budget-mib 256 --transfer-budget-mib 2048 --device-budget-mib 1984 --device-headroom-mib 128
```

The input file must tokenize to at least 4096 tokens. Adjust the device budget to actual free VRAM; 1984 MiB is a local T1000 evaluation limit, not a portable default. CUDA Graph capture is disabled consistently for these tests because this implementation rebuilds layer graphs; this is not a claim about the isolated benefit of that switch or about an implementation with graph reuse.

At large microbatches the routed union can cover nearly all 256 experts. Compare complete-bank mode with `--expert-slots 256` at the same microbatch, rather than forcing microbatch 8 to accommodate a 64-slot cache. Transfer counts, staging allocation and capacity failures are part of the comparison.

Phase counters separate current-weight loading, graph/input setup, compute plus waits, and activation-copy submission. Prefetch staging work can appear in setup or compute/wait time; these are selected host wall-time intervals, not exhaustive partitions or isolated CUDA kernel/hardware-overlap measurements.

The example also reports metadata restore, batch initialization and memory preparation separately from the microbatch setup interval. Setup details split memory apply, both graph builds, workspace measurement/admission, graph allocation, input construction, activation-input waits and prefetch admission. These are host wall times: input construction includes attention masks and other graph inputs, while workspace measurement can include synchronization. The matching NVTX subranges allow correlation with actual CUDA activities; do not add nested setup details to the parent setup total.

Batch initialization, microbatch splitting and cache-slot preparation run once per layered call. Later groups restore the original cache metadata and rewind the prepared attention, recurrent and indexer context cursors. Each microbatch still applies its state, and each layer still writes its own cache tensors. Indexer replay also discards derived pool state and reads the restored stale-pool metadata. The plan is local to the call and is not reused across prefix changes, requests or different microbatch sizes. `plan-reuses` reports the number of later groups that reuse it.

Ordinary causal masks use a contiguous-cache fast path when one sequence occupies a consecutive prefix of the addressed cells with consecutive positions, followed only by empty padding. Each row is filled as visible and masked spans rather than checking sequence membership for every query/key pair. M-RoPE still checks the spatial ordering at an equal position. ALiBi, sliding-window attention, multiple streams, mixed query sequences, fragmented cells and nonconsecutive positions keep the generic mask implementation. The fast path changes neither mask values nor attention precision and does not allocate a whole-prompt mask cache.

Hybrid attention inputs use the configured batch thread count (`-tb`) to fill large contiguous masks in disjoint row ranges. Masks smaller than 8 MiB remain serial; larger masks use at most one thread per 4 MiB and never more threads than rows or the configured count. The calling thread fills one range. Extra workers are created lazily, sleep between fills, and are retained by the KV cache until its destruction or a different parallel thread limit is requested. Layout validation runs once before any worker writes, and filling completes before graph submission. Generic masks remain serial. Changing thread counts invalidates the affected hybrid graph input's reuse decision.

The `mask` line reports time within input construction, including layout validation, filling and worker dispatch, plus the number of mask calls, parallel calls and maximum threads actually used in this layered call. It is not an additional phase to add to `inputs`. `test-kv-mask` checks serial and parallel output against a scalar reference, rejected layouts without writes, and worker growth and uneven row partitions. Its optional `--bench` mode measures 64,000-token mask construction at microbatch 2048 for F16/F32 storage and thread limits 1, 2, 4, 8, 12 and 24; it excludes attention computation and DMA. Each point has one warmup traversal and reports the median and sorted samples of three timed traversals.

On an RTX 3070 8 GiB with a Xeon Platinum 8268, a 64,000-token Qwen3.5-122B-A10B Q4_K_M prompt measured the following progression. These are single unprofiled runs with microbatch 2048, 24 inference/batch threads, F16 K/V, F32 activation staging, complete streamed expert banks, a 4096 MiB workspace ceiling and a 6144 MiB device budget. CUDA Graphs were disabled; there was no generated output. Wall time excludes model/context creation but includes run-local cleanup.

| Implementation | Wall seconds | Tokens/s | Graph/input setup seconds | Batch initialization seconds | Memory preparation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 277.891 | 230.306 | 134.684 | Not separately measured | Not separately measured |
| Contiguous mask plus detailed timing | 156.496 | 408.957 | 16.652 | 4.609 | 59.474 |
| Plus request-local batch/cache plan reuse | 93.518 | 684.360 | 16.647 | 0.070 | 1.217 |

The final run reused its memory plan for 47 later groups; expert upload remained 70,061,654,016 bytes and peak measured workspace remained 922,877,952 bytes. Separate Nsight captures reduced time without traced GPU activity from 203.712 of 280.723 seconds (72.6%) to 20.857 of 94.584 seconds (22.1%). Kernel-active interval coverage increased from 22.4% to 63.0%; this is not SM occupancy. H2D and D2H payloads were unchanged. Profiled windows exclude some cleanup and are not interchangeable with the unprofiled wall times above. These results are not a statistical throughput estimate or evidence for an isolated CUDA Graph/prefetch improvement: both graph rebuilding and weight-prefetch admission remain unchanged.

`test-kv-mask` compares both mask storage types against a scalar reference, including padding, nonzero positions, M-RoPE ties and rejected fragmented/mixed-sequence layouts. The existing `test-llama-archs --direct-weight-upload --backend CUDA0 --seed 42` and `test-llama-archs --layered-prefix --seed 42` cover streamed Qwen execution and default CPU GLM/Qwen prefix continuation, respectively, including state, subsequent decode and resource-admission checks.

#### Parallel mask measurements

A follow-up on the same RTX 3070/Xeon Platinum system kept the 64,000-token prompt, microbatch, precision and budgets above. A fresh previous-code control with `-t 24 -tb 24` measured 94.369 seconds (678.191 tokens/s). The parallel implementation measured:

| Batch thread limit | Wall seconds | Tokens/s | All graph inputs seconds | Mask seconds | Parallel mask calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 | 89.598 | 714.306 | 2.205 | 2.082 | 384 |
| 1 | 94.214 | 679.307 | 6.772 | 6.655 | 0 |

Inference threads stayed at 24 in both follow-up runs; only the batch thread limit changed for the serial comparison. All 993,280 final-logits bytes matched between them. The parallel run used up to 24 threads and reduced total graph/input setup to 12.570 seconds. Expert upload and measured workspace remained 70,061,654,016 and 922,877,952 bytes, respectively. The 5.3% throughput gain over the fresh same-configuration control is a single-run observation, not a confidence bound. Neither graph reuse nor prefetch admission was changed.

The standalone `test-kv-mask --bench` measured these median milliseconds for one traversal of the 64,000-token input's masks, not all 12 attention layers:

| Thread limit | F16 mask ms | F32 mask ms |
| --- | ---: | ---: |
| 1 | 527.789 | 814.723 |
| 2 | 388.075 | 584.012 |
| 4 | 341.282 | 422.634 |
| 8 | 239.796 | 247.292 |
| 12 | 205.485 | 270.572 |
| 24 | 188.490 | 281.573 |

These measurements include contiguous-layout validation and worker dispatch, but exclude model execution and transfers. F16 benefited through 24 threads in this scan; F32 did not improve beyond eight. More threads are not a portable guarantee of better memory-bandwidth performance.

#### Automatic CUDA weight staging

CUDA execution with streamed complete expert banks (`--stream-weights` and `--expert-slots 0`) automatically uploads compatible pinned expert tensors directly, reuses any required weight staging and uploads admitted next-layer expert banks on a worker. No additional switches are needed. Other backends, CPU execution, non-streamed weights and route-driven expert slots keep their existing paths.

Direct expert uploads require contiguous source tensors allocated with the selected CUDA device's host buffer type and no new gate/up interleaving. This includes source tensors that already have the merged layout. Both mandatory loading and optional prefetch submit the model-owned source pointer without copying through the two upload slots. The upload stream is synchronized before consuming the bank or releasing its upload task. Pageable sources, including `--no-host` models, and newly merged banks retain event-protected staging. Packed static weights still require their existing staging.

The CPU tensor-placement override can select `CUDA_Host`, not just ordinary pageable CPU memory. On Windows, these pinned model allocations can appear as shared GPU memory: the 122B Q4_K_M model alone occupies about 71 GiB. Direct uploads remove the redundant CPU copy and expert bounce buffers, not the resident pinned model or the expert H2D payload. Model storage is excluded from the incremental prefill host/transfer budget and the weight-staging peak counter. `direct_expert_bytes` counts expert bytes uploaded without a host staging copy; `expert_copy_bytes` still counts all expert payload uploaded.

Pinned expert-upload and packed-static-weight buffers are retained within one prefill. The cache has four spare slots and reuses exact-size buffers. Active and cached allocations share the remaining host and transfer budgets after activation staging. The pool's host limit subtracts the single allocated suffix activation array once; the transfer limit subtracts only device activation slots, since full host activations use the host budget. Unused cached buffers are released when another allocation needs the space. DMA completion is checked before a buffer is returned to the cache; all cached storage is freed when prefill ends. Device-bank layout, graph construction and scheduler workspace reuse are unchanged.

One worker with at most one queued task submits admitted next-layer expert banks. It is created lazily after the current layer's first successful graph submission, when there is an admitted expert prefetch to submit. It uploads each eligible source tensor directly; otherwise it packs and streams the complete next bank through the existing two event-protected slots, rather than copying a quota on the compute-submission thread after each microbatch. The next layer waits for task completion and upload completion before consuming the bank. Worker exceptions propagate to the prefill caller; cleanup joins outstanding work before releasing its tensors, events or staging buffers. Static-weight packing remains on the calling thread.

Device prefetch admission is unchanged. Direct expert uploads require no additional expert host/transfer staging reservation. With weight prefetch disabled or rejected, no worker is created, and mandatory loading still selects direct upload or reusable staging according to the source. `weight_worker_bytes` counts bytes actually submitted by completed worker tasks; it is separate from `weight_prefetch_submit_bytes`, which counts the original caller-thread quota submissions on unchanged paths. Neither counter measures physical overlap. Allocation/reuse counters and peak retained pinned bytes expose cache behavior.

A/B selection is restricted to the test-only `test_weight_staging` parameter: -1 selects automatic behavior including eligible direct uploads, while 0..3 retain staged uploads with the baseline/reuse/worker/both bitmask on the eligible CUDA path. It cannot enable these optimizations on other paths. The example CLI does not expose staging switches or staging comparison mode. Run `test-llama-archs --weight-staging --seed 42 --backend CUDA0` for baseline, isolated optimizations, automatic behavior and disabled/rejected prefetch coverage; the CPU form checks that automatic selection leaves synchronous execution unchanged. Run `test-llama-archs --direct-weight-upload --seed 42 --backend CUDA0` for direct uploads, pageable and repacking fallbacks, forced staged uploads, and disabled/rejected prefetch, with F32/BF16 activations and final-logits/decode comparisons. Keep the prompt, microbatch and budgets fixed when comparing logits.

Before making staging automatic, a local RTX 3070 8 GiB with PCIe 3.0 x16 and Xeon Silver 4112 produced the following mean throughputs from two points per mode, in forward/reverse order, using the 4096-token reference prompt at microbatch 1024. Both Q4_K_M models used F32 staging, four CPU threads, Flash Attention, disabled CUDA Graphs and a 5632 MiB device budget selected from available VRAM. The 35B compute/group limits were 2048/1536 MiB; the 122B limits were 1024/3584 MiB. Both used static/transfer limits of 256/2048 MiB and 128 MiB device headroom.

| Model | Baseline tokens/s | Reuse only | Worker only | Both | Both / baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | 386.488 | 510.846 | 560.962 | 744.353 | 1.926 |
| Qwen3.5-122B-A10B | 150.575 | 184.743 | 180.931 | 233.098 | 1.548 |

All eight final-token logits files per model matched their same-microbatch baseline byte-for-byte. With both optimizations, weight-staging allocations fell from 80 to 3 for 35B and from 96 to 3 for 122B, with peak retained weight staging of 88.3 and 210.7 MiB respectively. These are local paired measurements, not replacements for earlier T1000/3070 sweeps or confidence bounds; system conditions and sample order affect throughput.

A separate baseline/both Nsight comparison on 35B reduced the captured prefill window from 10.887 to 6.797 seconds. Within that window, `cudaMallocHost` calls fell from 82 to 5, with summed allocation time falling from 1.286 to 0.081 seconds. H2D/kernel overlap coverage increased from 15.3% to 26.1%, while H2D payload remained 22,667,848,676 bytes. Time without traced process GPU activity fell from 7.799 to 3.689 seconds. The capture ends before run-scoped cache cleanup, whereas the unprofiled throughput above includes cleanup. Profiled timings are not interchangeable with unprofiled timings, and the remaining gaps still include host preparation and submission; the GPU is not continuously busy.

A subsequent paired run compared the preserved automatic staging executable with direct pinned expert uploads on the same RTX 3070/Xeon system. Both used the 4096-token prompt, microbatch 1024, F32 staging, four CPU threads, Flash Attention, disabled CUDA Graphs, a fixed 5632 MiB device budget selected from current free VRAM, and the same per-model compute/group and static/transfer/headroom limits above. Each model ran in staged/direct/direct/staged order, with two samples per mode:

| Model | Staged mean tokens/s | Direct mean tokens/s | Direct / staged |
| --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | 680.458 | 682.492 | 1.003 |
| Qwen3.5-122B-A10B | 192.272 | 267.956 | 1.394 |

The 35B result does not establish a speedup. The 122B staged samples varied from 168.049 to 216.494 tokens/s, while direct samples were 266.662 and 269.249 tokens/s; the mean improvement is local evidence, not a confidence bound or a comparison with older measurements. All eight logits files matched the prior same-microbatch reference byte-for-byte. Direct uploads eliminated 19,461,570,560 bytes of expert host staging copies for 35B and 70,061,654,016 bytes for 122B, without changing the expert upload payload. Weight-staging allocations fell from three to two and peak incremental weight staging fell by exactly 16 MiB for each model, to 72.3 and 194.7 MiB; packed static staging remains. CPU `test-weight-staging` and all 24 CUDA `--direct-weight-upload` cases passed, including pageable/repacking fallbacks, staged diagnostic selection, and disabled/rejected prefetch.

#### Opt-in streamed compute experiments

Two switches apply only to GPU execution with `--stream-weights` and complete expert banks (`--expert-slots 0`). Neither changes the default path, the host model, the requested microbatch, the total expert payload or the device-budget policy.

- `--merge-gate-up` packs each expert's gate and up rows next to each other while filling the existing bounded pinned upload buffers. It reuses the existing merged `MUL_MAT_ID` graph, with no full-model host repack and no extra permanent device weights. Compatible banks retain this layout when recycled. Separate scales/biases, different gate/up formats or shapes, and activation clamping are rejected rather than silently changed.
- `--defer-static-upload` allocates and packs the next static bank before compute submission, but queues its DMA after submitting the current graph and before the next expert upload loop. The pinned buffer remains alive until its ready event completes; a prepared bank cannot be consumed before submission. Without weight prefetch there is no next static bank to defer.

For a paired comparison, add `--compare-optimizations -ub 2048 -n 0` to the long-input command instead of `--sweep-ubatch`. It loads the CPU model once and creates eight fresh contexts: baseline, merge, defer, both, then the reverse order. Prompt length, microbatch, state types and budgets remain fixed. A logits path gains `.caseN.ub2048`, so repeated points do not replace each other's results. This mode rejects other sweeps, prefix reuse, automatic planning and GPU profiling. Compare repeated end-to-end measurements; clock drift can make a sequence of different processes misleading.

The existing backend performance runner includes the five actual dense shapes `(output,input) = (8192,2048), (4096,2048), (2048,4096), (512,2048), (2048,512)` at 2048 tokens with Q8_0, F16 and F32 weights, plus separate/merged Q4_K expert widths. On the local T1000, the first isolated Q8_0 products took 3.59-54.35 ms; F16 products took 12.90-180.24 ms, 3.3-3.9 times slower before charging dequantization. Therefore these experiments do not replace dense MMQ with F16 GEMM. That result does not predict Ampere Tensor Core performance. A later three-type scan measured F32 products about 8-17% faster than Q8_0 at these shapes, before dequantization and extra workspace; that is not an implemented end-to-end gain and would change activation arithmetic.

```powershell
$env:CUDA_VISIBLE_DEVICES = '2'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
.\build-prefill-cuda\bin\test-backend-ops.exe perf -b CUDA0 -o MUL_MAT -p 'type_a=(q8_0|f16|f32),type_b=f32,m=(8192|4096|2048|512),n=2048,k='
.\build-prefill-cuda\bin\test-backend-ops.exe perf -b CUDA0 -o MUL_MAT_ID -p 'type_a=q4_[Kk],type_b=f32,n_mats=256,n_used=8,b=0,m=(512|1024),n=2048,k=2048'
```

The expert microbenchmark measured 60.774 ms for a 512-row Q4_K product and 114.503 ms for the merged 1024-row product, about 5.8% less than two separate products. It excludes host packing and the rest of the layer.

The same-host-model 35B Q4_K_M comparison (4096 tokens, microbatch 2048, the same 1984 MiB accounted device limit) returned:

| Flags | Forward-order tokens/s | Reverse-order tokens/s |
| --- | ---: | ---: |
| None | 124.370 | 99.331 |
| Merge gate/up | 114.787 | 99.469 |
| Defer static upload | 103.577 | 100.238 |
| Both | 100.967 | 100.300 |

The baseline itself drifted substantially; the reverse-order points differ by about 1%, not a demonstrated material throughput improvement. Do not compare an earlier isolated 131-133 tokens/s baseline with an older 102-107 tokens/s sweep and credit the difference to these flags. They remain off by default. All eight final-logits arrays matched the unchanged baseline exactly (NMSE zero); expert/static transfer bytes and peak accounted device bytes were unchanged.

A separate matched Nsight pair did establish the intended scheduling change: H2D/kernel coverage 84.223% -> 91.689%, with H2D without a kernel 587.913 -> 305.599 ms and identical 22709791476 H2D bytes. Both profiled outputs also matched the baseline exactly. This establishes more concurrent activity, not an end-to-end speedup. Validation covered 96 CUDA Qwen logits/state/decode/budget cases, including a split Q4 fixture whose fused payload crosses the 8 MiB upload boundary and merged-bank reuse without prefetch, plus the four CPU regression tests and 13 timeline analyzer cases.

#### Large-batch local results (T1000)

The portable numeric record is [layered-prefill-results.json](layered-prefill-results.json). It includes settings, model revisions, the exact synthetic prompt recipe and checksum, run order, rejected cases, timing units and validation limitations. It deliberately excludes raw profiler metadata and machine-specific artifact paths. The top-level environment and sweeps describe the T1000; the separate `rtx3070_direct_upload_comparison` preserves the historical RTX 3070 results: 682.492 tokens/s for 35B and 267.956 tokens/s for 122B, averaged over two direct-upload runs per model at microbatch 1024. See [Automatic CUDA weight staging](#automatic-cuda-weight-staging) for the paired staged/direct comparison and its limitations. The latest RTX 3070 length sweep is recorded below and in `rtx3070_latest_length_comparison`.

With 4096 tokens, F32 staging, Flash Attention, host-resident Q4_K_M weights and the limits above, the complete-bank sweep with prefetch requested measured:

| Microbatch | 35B-A3B tokens/s | 122B-A10B tokens/s |
| ---: | ---: | ---: |
| 128 | 25.287 | 5.556 |
| 256 | 36.412 | 9.376 |
| 512 | 57.180 | 15.242 |
| 1024 | 81.989 | Device-budget rejection |
| 2048 | 102.201 | Device-budget rejection |

The 122B 1024/2048 graphs required accounted working sets of 2138576000/2453156992 bytes, above the 2080374784-byte limit. Neither point was silently reduced or run through shared GPU memory. The fastest admitted 122B point was 512. Its subsequent eager-load comparison process could not allocate the roughly 76.5 GB host model under the then-current Windows commit limit, despite enough free physical RAM; the scan is complete, but those additional comparisons are not established by these measurements.

The 122B logs report zero accepted and 47 skipped next-layer expert prefetches; static prefetch was also skipped 47 times. Its 15.242 tokens/s is therefore complete-bank loading without effective next-layer prefetch, not a successful double-bank prefetch result. The corrected `mmap+mlock` retry has not been run; no successful page locking or additional 122B comparison is claimed.

At the same 35B microbatch 2048, no-prefetch complete loading measured 99.059 tokens/s, synchronous demand loading 100.946, and asynchronous demand loading 83.588. A complete-bank/prefetch repeat measured 106.685. All four comparisons produced identical final logits relative to the sweep reference (NMSE zero). An earlier isolated complete-bank/prefetch pilot reached 124.344; do not substitute that peak for the sustained sweep/repeat results.

Complete loading uploaded 19461570560 expert bytes; demand loading uploaded 17346265088, about 10.9% fewer. Both uploaded 1585932800 static layer-weight bytes. At this large microbatch a 256-slot demand bank has essentially the same current-layer capacity as a complete bank, unlike the earlier 64-slot/microbatch-8 experiment.

#### Latest RTX 3070 length sweep and T1000 retest

The 2026-10-25 comparison uses the latest mask fast path, batch/cache plan replay and parallel mask filling on an RTX 3070 8 GB with a Xeon Platinum 8268, 128 GB host RAM and 24 inference/batch threads. Both models use the same Q4_K_M revisions recorded above, F16 KV, F32 activation staging, Flash Attention, CPU-resident streamed weights and microbatch capacity 2048. Each point is a separate process with no prefix reuse and zero generated tokens. Throughput uses unprofiled prefill wall time, including prefill weight transfers and cleanup but excluding initial model loading and context creation.

| Actual input tokens | RTX 3070: 35B-A3B tokens/s | RTX 3070: 122B-A10B tokens/s | T1000: 35B-A3B tokens/s | T1000: 122B-A10B tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 332.417 | 104.162 | Pending | Pending |
| 4096 | 1031.532 | 333.212 | 102.201 (historical) | 15.242 (historical, microbatch 512) |
| 32000 | 1678.657 | 754.624 | Pending | Pending |
| 128000 | 1125.669 | 575.407 | Pending | Pending |

The T1000 has not been retested with this code. Its two 4096-token entries are references from the earlier sweep, not matched measurements: CPU, software revision, prompt length before truncation, microbatch and memory budgets differ. Do not interpret these columns as isolated GPU speedup ratios. All four T1000 lengths remain pending a new run; a rejected case must be recorded as a rejection, not zero throughput or an unreported change of settings.

The latest prompt uses the top-level numeric record's two paragraphs and CRLF recipe repeated 3000 times instead of 48, producing 2346002 bytes with SHA-256 `90686fed49f25a9811599ac20a6bc98e3b6fd6a14f9974085db181d33feb50ad`. Each model tokenizes this file and truncates to the exact input count; 32000/128000 are not 32768/131072. Context capacity is `max(input_tokens + 64, 2048)` and logical batch capacity is `max(input_tokens, 2048)`. The 1024-token runs therefore use `-c 2048 -b 2048 -ub 2048`, but process only 1024 tokens in one microbatch; the capacities are not extra input or padding tokens.

The RTX 3070 uses 6144 MiB accounted device budget, 256 MiB static budget, 2048 MiB transfer budget and 128 MiB headroom. Compute/group budgets are 2048/1536 MiB for 35B and 4096/3584 MiB for 122B. CUDA Graph capture remains disabled. Do not copy the 6144 MiB budget to a 4 GB T1000; use its actual available VRAM and record any budget, microbatch or KV precision differences explicitly.

Each length has one measurement, not confidence bounds. All eight saved final-token logits arrays have 248320 finite F32 values; this does not establish equality against an older executable or quality across a corpus. The earlier RTX 3070 direct-upload and T1000 measurements remain preserved separately.

### GPU activity timeline and overlap

`--profile-gpu` opts into a CUDA profiler capture range and NVTX annotations. It does not itself collect a timeline. The CUDA backend must have NVTX3 support (enabled when CMake provides `CUDA::nvtx3`), and Nsight Systems supplies the actual CUDA activity collector. Other backends are not presented as CUDA profiling. Use one manual prefill; CLI sweeps, automatic calibration and prefix demonstrations are rejected for capture.

```powershell
$env:CUDA_VISIBLE_DEVICES = '2'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe'
py .\scripts\gpu_timeline.py capture --nsys $nsys --output D:\AI\traces\prefill35 -- .\build-prefill-cuda\bin\llama-layered-prefill.exe --stream-weights --load-mode none -m D:\AI\Qwen3.5-35B-A3B\Qwen3.5-35B-A3B-Q4_K_M.gguf --device CUDA0 -c 4096 -ub 2048 -b 4096 -n 0 -t 10 -tb 10 -fa on -f D:\prompts\long-input.txt --prompt-tokens 4096 --staging-type f32 --compute-budget-mib 768 --group-budget-mib 1536 --static-budget-mib 256 --transfer-budget-mib 2048 --device-budget-mib 1984 --device-headroom-mib 128
```

The wrapper adds `--profile-gpu`, disables CPU sampling/context-switch tracing, requests node-level CUDA activity and exports SQLite. Model loading and context creation occur before capture. Only a completed `llama.prefill` NVTX range with its success marker is accepted for analysis. Captures/output files are not overwritten. A missing profiler, unsupported schema, reported CUDA loss, unattributed activities, incomplete range or batched-copy envelope is an error, not a zero-overlap result.

Each output prefix produces:

| Suffix | Purpose |
| --- | --- |
| `.nsys-rep` | Native Nsight Systems timeline |
| `.sqlite` | Original exported activity timestamps |
| `.trace.json` | Chrome Trace format, with host phases, CUDA streams and aggregate activity/overlap/gap lanes |
| `.summary.json` | Per-CUDA-device durations, byte counts and overlap ratios |
| `.capture.log` | Collector and target output |

Analyze an existing capture without running the GPU again:

```powershell
py .\scripts\gpu_timeline.py analyze D:\AI\traces\prefill35.sqlite --output D:\AI\traces\prefill35-analysis
py -m pytest .\tests\test_gpu_timeline.py -q
```

CUDA kernel and memcpy start/end timestamps define GPU activities. Concurrent kernels are merged into a kernel-active union; H2D/D2H intervals are merged separately. Their intersections give the overlap durations. H2D coverage is the intersection duration divided by H2D-active union duration, not the sum of overlapping stream envelopes or CPU API durations. Empty denominators are reported as null. Different GPUs and other processes are not combined. CUDA logical IDs are mapped through process-specific CUDA metadata, not assumed to equal system GPU indices.

`no_traced_gpu_activity_ns` means no kernel/memcpy/memset from the traced process in that interval. It is not whole-machine GPU idle time, SM occupancy, ALU utilization or a measurement of other applications. Run no unrelated GPU work in the target process. Runtime/NVTX ranges describe host submission/setup/waits and are not used as kernel-active intervals.

In a local 35B, 4096-token, microbatch-2048 capture:

| Mode | H2D active seconds | H2D/kernel overlap seconds | H2D coverage | H2D without kernel seconds |
| --- | ---: | ---: | ---: | ---: |
| Complete-bank prefetch | 3.681 | 3.118 | 84.708% | 0.563 |
| Complete-bank, no weight prefetch | 3.674 | 0.105 | 2.863% | 3.569 |
| Demand, synchronous | 3.388 | 0 | 0% | 3.388 |
| Demand, asynchronous | 3.408 | 0.000149 | 0.00436% | 3.408 |

The asynchronous-demand scheduling counters therefore did not establish effective physical overlap in this configuration. The timeline exposes that limitation instead of calling the submitted bytes hidden. All four profiled runs retained the unprofiled reference's final logits (NMSE zero).

These captures recorded 22709791476 H2D and 1342185472 D2H bytes for complete loading, versus 20594608884 H2D and 1509878272 D2H for demand loading. The reduced expert traffic is partly offset by routing readback and slot-map updates; host/device traffic savings are not the same percentage as expert-weight savings.

Activity coverage is not a counterfactual critical-path hiding percentage: concurrent transfers can contend with kernels, and transfer intervals without a kernel may overlap CPU preparation. Compare unprofiled end-to-end runs separately, since collection has overhead. Keep raw reports local: Nsight can include command-line/environment metadata. The filtered Chrome Trace output omits that metadata and can be opened in a local viewer; no report is uploaded by this tool.

### Performance analysis and unresolved questions

The following conclusions use the latest default/merge-and-defer captures, not the earlier four-mode capture or unprofiled wall times. They are observations on the T1000, not predictions of another GPU's throughput.

| Activity | Default seconds | Merge and defer seconds |
| --- | ---: | ---: |
| Capture window | 34.363 | 36.978 |
| Kernel-active union | 31.086 | 33.600 |
| H2D-active union | 3.726 | 3.677 |
| D2H-active union | 0.205 | 0.207 |
| H2D/D2H combined union | 3.930 | 3.883 |
| H2D without a kernel | 0.588 | 0.306 |
| No traced kernel/copy/memset | 2.369 | 2.735 |

Kernel execution occupies about 90.5%/90.9% of these windows. Matrix multiplication is about 85% of summed kernel duration, and therefore the largest measured target. This does not establish pure arithmetic saturation: memory access, quantized-weight processing, occupancy, scheduling and interference can contribute to kernel duration. The main products remain quantized MMQ; F32 residual staging or an FP16 attention kernel does not mean every matrix product is FP16.

The intervals with no traced GPU activity are not a CPU profile or whole-device idle measurement. CPU packing, graph construction, submission, synchronization and other processes have not been separated. PCIe is not the largest exposed activity in these captures, but it can still contend with kernels. A separate `nvidia-smi` observation reported PCIe Gen3 x8 (x16 maximum); it was not a continuous link/clock trace during these runs.

#### Why did kernel time increase with the experiments?

The merged gate/up calls halve in count but double their output width: they do not halve the model's multiplication count. The isolated product benchmark's gain did not reproduce in the complete capture:

| Kernel category | Default calls / summed seconds | Merge and defer calls / summed seconds | Duration change |
| --- | ---: | ---: | ---: |
| Q8 main projections | 500 / 10.506 | 500 / 11.285 | +7.4% |
| Q4 gate/up | 160 / 10.220 | 80 / 11.280 | +10.4% |
| Q5 down | 80 / 5.518 | 80 / 5.915 | +7.2% |
| Flash Attention | 20 / 2.634 | 20 / 2.923 | +11.0% |
| DeltaNet recurrence | 60 / 0.817 | 60 / 0.819 | +0.3% |
| MMQ activation quantization | 860 / 0.221 | 780 / 0.178 | -19.5% |

These are sums by selected kernel name, not union-based overlap metrics. Unchanged projections/down/attention also slowed, whereas DeltaNet was almost unchanged. Do not explain everything as extra merged-kernel work, or assert that uniform clock reduction was measured. Moving DMA into compute can introduce shared-memory-system or power-budget competition; run-condition drift and different kernel scheduling are other possibilities. No clock or hardware-metric tables were collected, and the captures do not isolate these causes. Independent merge-only/defer-only activity captures with telemetry would be needed for causal attribution. Both switches stay off by default.

#### When would transfer become the bottleneck?

At unchanged bytes and effective transfer rate, dividing kernel-active time by H2D-active time gives about 8.34x for the default and 9.14x for both switches. Including the H2D/D2H union gives 7.91x/8.65x. Thus roughly 8-9x kernel acceleration would make aggregate communication activity comparable to computation. This is a balance estimate, not a precise threshold: a layer can run out of prefetched data earlier, and producer staging/submission gaps are not included in DMA-active duration.

Do not divide by only the currently exposed 0.306 seconds: most of the presently covered 3.372 seconds can become exposed when compute shrinks. About 22.7 GB of H2D activity at the observed rate alone corresponds to roughly 1100 tokens/s for this 4096-token input, before other costs. Changed PCIe links, cache/residency policies, prompts and copy/compute contention change that estimate. GPU peak FP32 or INT8 TOPS ratios do not establish a prefill speedup.

#### Strata comparison

The code inspection used Strata revision `477ac13cbe8e582c20484871ce459d20d82df09d`, read-only. Its long-chunk path streams all non-resident experts in a known order through an event-protected ring, rather than predicting exact future routes. Expert groups can compute before an entire layer is uploaded. Separate host staging/issuer threads keep CPU copies away from the compute-submission thread; already pinned expert data can DMA directly. It also uses GPU-resident dense weights/selected experts, scratch reuse, larger chunks, dense FP16 GEMMs and format-specific fused Tensor Core expert kernels.

These are design differences to evaluate, not transplanted code or a local performance validation. Strata's `docs/QWEN36.md` reports Qwen3.6-35B-A3B UD-Q4_K_XL on an RTX 3060 12 GB at 2022 tokens/s for pp2048, against 1328 tokens/s for its llama.cpp comparison. That is not the T1000 or RTX 3070 Qwen3.5 CPU-resident placement experiment. Its Flash-Next long-prompt and multi-GPU results are different conditions again. The local RTX 3070 measurements above apply only to their recorded workload; they do not establish a portable 300-450 tokens/s limit or an 800/2000 tokens/s guarantee. No local RTX 3060 or 3080 throughput was measured.

### Earlier small-microbatch measurements

Measured on Windows with an i9-10900X, 128 GB host RAM and an NVIDIA T1000 4 GB (SM75). Both models use the downloaded Q4_K_M GGUFs. Non-expert weights and hybrid state remain on CPU (`-ngl 0`); routed expert kernels run on CUDA. The input is the same 131-token text, with context 512, logical batch 512, microbatch 8, ten CPU threads and F32 activation staging. Scheduler, expert-group and device-headroom limits are 256 MiB, 1536 MiB and 128 MiB respectively. Complete-bank next-layer prefetch is disabled in these comparisons.

Each configuration ran twice. The table reports the second, warm-cache run, not a statistical average. Wall time excludes model/context creation; initial memory-mapped weight faults can still dominate a cold prefill. Expert residency includes the compact device slot map, but not the rest of the process's GPU allocations. Uploaded GB are decimal expert-weight bytes only.

| Model | Mode | Peak expert MiB | Expert upload GB | Prefill seconds | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 35B-A3B | Complete, two-layer groups | 928 | 19.462 | 18.490 | 7.085 |
| 35B-A3B | Complete, one-layer groups | 464 | 19.462 | 19.156 | 6.838 |
| 35B-A3B | 64 slots, synchronous | 116.001 | 13.617 | 13.790 | 9.500 |
| 35B-A3B | 64 slots, pipelined | 116.001 | 13.617 | 15.561 | 8.418 |
| 122B-A10B | Complete, two-layer groups | 2784 required | Not run | Budget rejection | Not run |
| 122B-A10B | Complete, one-layer groups | 1392 | 70.062 | 57.899 | 2.263 |
| 122B-A10B | 64 slots, synchronous | 348.001 | 69.423 | 52.738 | 2.484 |
| 122B-A10B | 64 slots, pipelined | 348.001 | 69.423 | 54.278 | 2.414 |

All 14 successful runs produced numerically identical final-token F32 logits (248320 values, NMSE zero) relative to their model's complete one-layer reference. Both attempts at 122B two-layer groups were rejected before expert allocation by the explicit group budget; these are not successful GPU runs or measured OOM speedups.

The compact cache reduces expert residency by about 75% relative to complete single-layer banks. On this prompt it reduces 35B expert upload by about 30%, but 122B upload by less than 1%: a small resident set can require substantial eviction and re-upload. Residency savings and transfer savings are separate results.

Pipelining is slower than synchronous demand loading in these local measurements. Pinned staging, stream/event handling and synchronization are not fully hidden by the available shared-expert work. Use `--no-pipeline` for the faster tested T1000 configuration; do not infer that asynchronous submission establishes a speedup. These short-prompt, microbatch-8 results do not establish long-context performance, optimal batch sizes or quality across a prompt corpus.

The earlier small-microbatch revision passed the four registered CPU prefill tests, 60 Qwen CUDA precision/pipeline/placement combinations, and targeted `CPY,GET_ROWS` checks (258 CPU and 158 CUDA cases). Deliberate token, slot, budget and nonempty-context failures are expected diagnostics.

## Single pinned activation buffer and Qwen prefix reuse (2026-10-22)

The current path keeps one full activation array for the evaluated suffix, not one array per layer and not an array for the configured context capacity. GPU calls allocate that array through the backend's pinned host buffer type and reject a pageable fallback. CPU correctness calls use the same single-array lifetime without page locking. The array is released after all transfers complete at the end of the call; KV, indexer, convolution and recurrent state remain in the context.

In-place overwrite does not depend on approximate routing or reduced precision. Each graph reads a microbatch into separate device storage before computing. Its output can overwrite the same host slice only after compute synchronization, which also completes the input upload dependency. The prefetched next microbatch occupies a disjoint slice. Two device output slots retain their completion events, and every download is drained before the next group starts. This removes both complete-array ping-pong and activation bounce-buffer copies without changing expert uploads or the requested microbatch.

Prefix validation is shared by the ordinary hybrid memory used by Qwen and the indexed hybrid memory used by GLM. GLM adds its indexer checks on top. Both preserve a valid prefix when admission fails, reject incomplete or shared state, and report invalidation if execution fails after updating recurrent state. This is live-state suffix reuse, not automatic token-prefix matching or arbitrary recurrent rollback. Qwen automatic planning is still unsupported.

CPU and CUDA builds passed. The five registered CPU tests and the second-seed prefix run passed; CUDA passed both-model prefix tests at seeds 42 and 17 (64 combinations per seed), 96 Qwen prefill combinations and the grouped/joint-planner tests. Prefix tests cover a host budget equal to one suffix buffer, restored state, native decode followed by a second suffix, F32/BF16 staging, F16/Q8_0 caches and failure recovery. The final CUDA binary repeated the seed-17 prefix checks after the shared-validation cleanup. The MSVC AddressSanitizer build also passed all 64 prefix combinations at seed 42. Disabling pinning explicitly rejected the GPU call even with `--no-pipeline`; no pageable fallback was accepted.

```powershell
ctest --test-dir build-prefill -R "^(test-alloc|test-layered-prefill|test-qwen-prefill|test-layered-planner|test-layered-prefix)$" --output-on-failure
.\build-prefill\bin\test-llama-archs.exe --layered-prefix --seed 17
$env:CUDA_VISIBLE_DEVICES = '2'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
.\build-prefill-cuda\bin\test-llama-archs.exe --layered-prefix --seed 42 --backend CUDA0
.\build-prefill-cuda\bin\test-llama-archs.exe --layered-prefix --seed 17 --backend CUDA0
.\build-prefill-cuda\bin\test-llama-archs.exe --qwen-prefill --seed 42 --backend CUDA0
.\build-prefill-cuda\bin\test-llama-archs.exe --layered-planner --seed 42 --backend CUDA0
.\build-prefill-asan\bin\test-llama-archs.exe --layered-prefix --seed 42
```

These commands passed. CUDA builds used the matching MSVC 14.44 environment and CUDA 13.4; CPU builds used MSVC 14.51. Real GLM-5.3-Flash, 200K/1M capacity and corpus-level BF16 quality remain unmeasured.

### Real Qwen 35B measurements

The numeric record's `single_pinned_activation_prefix` section contains the exact prompt recipe, checksum, flags, order and results. This used the same T1000, Qwen3.5-35B-A3B Q4_K_M, 4096 tokens, microbatch 2048, F32 staging and 1984 MiB accounted device limit as the local streamed-weight experiment, but a separate recorded synthetic prompt. The baseline executable was saved before changing branch revision `de8d7f908`. Compilation and tests from this session finished before the serial old/new/new/old repetitions; other users' activity and GPU clocks were not controlled.

| Run order | Version | Prefill ms | Tokens/s |
| --- | --- | ---: | ---: |
| 1 | Previous two host arrays and pinned bounce buffers | 32192.223 | 127.236 |
| 2 | Single pinned in-place array | 35380.443 | 115.770 |
| 3 | Single pinned in-place array | 37830.387 | 108.273 |
| 4 | Previous two host arrays and pinned bounce buffers | 38392.224 | 106.688 |

The unchanged baseline itself slowed by about 16% across the sequence, mainly in compute/wait time. An earlier candidate check measured 127.617 against 124.789 tokens/s, but it was not the interleaved comparison and the early baseline overlapped build/test activity. These runs do not establish a stable throughput gain or isolate a scheduling regression. No new clock telemetry or GPU activity capture was collected.

The capacity result is direct: host activation storage falls from 64 MiB of pageable arrays plus 64 MiB of pinned bounce buffers to one 32 MiB pinned array. The four device slots stay at 64 MiB, peak accounted device storage stays at 1708934784 bytes, and expert upload stays at 19461570560 bytes. Activation H2D/D2H payloads are unchanged. All 248320 final logits in the interleaved F32 runs matched the old reference byte for byte (NMSE zero).

The real-model prefix check reused 2048 tokens and evaluated only the next 2048. F32 allocated 16 MiB of pinned activations, not the full prompt's 32 MiB; the final append run took 19459.214 ms (105.246 tokens/s). BF16 allocated 8 MiB for the suffix versus 16 MiB for a full 4096-token prompt; its full/append measurements were 38187.924/20706.980 ms (107.259/98.904 tokens/s, append timing excludes seeding). Both precisions' append logits matched their same-precision full-prompt reference byte for byte. This checks state reuse and storage size on this prompt, not broad quality or a 1M-context performance claim.

Reproduce using the recorded prompt file and the existing large-prompt command with `--prompt-tokens 4096 -ub 2048 --logits-file PATH`; add `--prefix-tokens 2048` for the append case and change `--staging-type` explicitly for each precision. Each run returns success; prefix seeding and model/context creation are excluded from the printed suffix prefill timing. Full logs and logits remain local.

## Implementation files

| Surface | Files |
| --- | --- |
| Experimental API and execution | `src/llama-ext.h`, `src/llama-context-prefill.cpp`, `src/llama-context.h`, `src/llama-context.cpp` |
| Cost model and build registration | `src/llama-prefill-plan.h`, `src/llama-prefill-plan.cpp`, `src/CMakeLists.txt` |
| Grouped model graphs and compact slot lookup | `src/models/glm5-next.cpp`, `src/models/qwen35moe.cpp`, `src/llama-graph.h`, `src/llama-graph.cpp` |
| Complete-prefix metadata replay and validation | `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-memory-recurrent.h`, `src/llama-memory-recurrent.cpp`, `src/llama-memory-hybrid.h`, `src/llama-memory-hybrid.cpp`, `src/llama-memory-hybrid-idx.h`, `src/llama-memory-hybrid-idx.cpp` |
| Allocation fix and tests | `ggml/src/ggml-alloc.c`, `tests/test-alloc.cpp`, `tests/test-llama-archs.cpp`, `tests/CMakeLists.txt` |
| Measurement placement and cross-split cancellation | `ggml/src/ggml-backend.cpp`, `ggml/include/ggml-backend.h` |
| Runnable example and documentation | `examples/layered-prefill/layered-prefill.cpp`, `docs/development/layered-prefill.md` |

## Reference

- [Official GLM-5.3-Flash configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)
- [ik_llama.cpp expert cache](https://github.com/ikawrakow/ik_llama.cpp/blob/main/src/llama-moe-cache.cpp)
- Local implementation surfaces: `src/models/glm5-next.cpp`, `src/llama-context.cpp`, `src/llama-memory-hybrid-idx.cpp`, `src/llama-memory-recurrent.cpp`, `src/llama-moe-cache.cpp`.

The ik fork's cache uses layout-compatible device banks, expert slot maps and upload callbacks. Its LRU policy alone does not implement layer-major whole-input prefill. Reuse upstream GGUF and backend operations rather than importing fork-specific quantization formats or kernels without compatibility checks.
