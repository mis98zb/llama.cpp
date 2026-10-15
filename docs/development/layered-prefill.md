# Experimental layer-major prefill for GLM5-Next

## Goal and scope

Process a complete initial text prompt one layer at a time. Keep the current layer's complete routed experts on the device while processing all prompt microbatches in position order. Only then replace the experts with the next layer's weights. This is not ordinary chunked prefill and is not a larger decode expert cache.

The initial target is GLM-5.3-Flash, quantized GGUF, EPYC 7532 (AVX2), 256 GB RAM and a 20 GB SM86 GPU. Development on Windows has no target model. No real-model accuracy, 200K-token performance or speedup has been established.

The path must be explicit opt-in. Default decode remains unchanged. The first implementation accepts one text sequence in an empty context, consecutive positions starting at zero, and the final token's logits. Prefix reuse, multimodal input, speculative execution, adapters and concurrent sequences are outside the initial scope and must produce explicit errors.

## Execution

1. Validate the complete request and resource limits before changing the context.
2. Allocate two bounded host activation buffers.
3. For each layer, make its complete routed expert tensors device-resident, build a single-layer graph and process every microbatch in position order.
4. Read and write the complete mHC residual at the layer boundary: `[hidden_size, 4, tokens]`. Never collapse the streams before the output head.
5. Preserve every layer's final MLA, indexer, convolution and recurrent state for subsequent ordinary decode.
6. Run the output head for the final token and expose the usual logits.

The implementation uses existing `clear(false)` operations to reset memory metadata without clearing tensor data. An empty-context, single-sequence request replays the same microbatch slot layout at each layer without introducing per-layer cache cursors. Graph synchronization precedes metadata resets and weight replacement. Synthetic CPU tests compare the final serialized sequence state and subsequent decode with the normal path.

The official GLM-5.3-Flash configuration has full indexers at every attention layer. This implementation rejects shared-indexer variants explicitly. Supporting them would additionally need compact selected indices and visibility information across layers. Saving a dense whole-prompt selection mask is prohibited because it grows quadratically.

## Precision and resource budgets

Use BF16 for host activation staging by default, with F32 as a correctness reference. Conversion must happen on the device to reduce PCIe traffic. This does not change the precision of KDA recurrent state, mHC computation, attention masks or arithmetic inside existing operators.

BF16 staging is not lossless. In particular, rounding can change the DSA top-k selection. An exploratory sparse miniature model with seed 17 produced final-logit NMSE about `8.9e-3` relative to unrounded F32. This is not a real-model quality measurement, but it rules out claiming that BF16 staging always preserves the original logits closely. Evaluate BF16 on representative real prompts before using it for quality-sensitive work; select F32 staging to isolate scheduling correctness.

For hidden size 4096, four streams and microbatch 256, one BF16 boundary block is 8 MiB in each direction. At 200000 tokens, two complete BF16 activation buffers require about 12.2 GiB of host RAM, excluding all other allocations.

The routed experts of one layer contain `3 * 4096 * 2048 * 288` parameters. If all three tensors use Q4_K, their storage is about 3.80 GiB. Mixed GGUF quantization can require more. Check actual tensor sizes and backend operator support, not the file's Q4 label.

The complete working set includes current-layer weights, persistent device caches, activation buffers, masks, operator workspace, backend overhead and safety headroom. Do not double-count allocations already reflected in device free memory.

The experimental API separately limits the two host activation buffers, the current expert buffer including backend padding, and the sum of scheduler compute buffers. The last limit is not a total-VRAM limit: persistent model/context allocations and backend-private kernel workspaces remain additional costs. Backend allocation failures are errors, not fallbacks.

Start evaluation at microbatch 256, compare 512, and consider 1024 only after measuring the real graph at the maximum prefix length. These are candidates, not verified capacity limits. The current API uses an explicit, fixed microbatch size, measures each graph before allocation, and rejects requests exceeding its scheduler-buffer budget. It does not yet select a microbatch automatically from free VRAM or preflight the complete prompt's peak. A late budget failure clears the initial prompt rather than preserving partially computed state. Every layer uses identical microbatch boundaries.

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

The 12 precision/quantization combinations passed on CPU and on a local NVIDIA T1000 (SM75); additional seed 17 runs exercised different sparse selections. The CUDA runs used `CUDA_VISIBLE_DEVICES=2` to isolate that device and `--backend CUDA0` inside the process. CPU F32 sequence-state comparisons were byte-identical. Existing `glm5-next`, `qwen4exp` and `kimi-linear` architecture checks also passed.

Across microbatches 7, 32 and 64, the miniature model copied exactly 1769472 expert bytes with F32 experts, or 663552 bytes with Q4_K experts, in three layer loads. The tests also cover invalid tokens, nonempty-context rejection, host/compute/expert budgets, cleanup and successful single-token retry after failure.

The standalone CUDA example loaded a locally generated tiny GGUF and completed both prefill and ordinary generation with BF16 and F32 staging. These tests did not use a downloaded GLM model. They do not validate SM86 fused long-context attention, 20 GB capacity, 200K latency, or real-model BF16 quality.

## Experimental API and example

`llama_prefill_layered` in `src/llama-ext.h` accepts the complete tokenized prompt and returns the final token's logits through the ordinary logits getters. Sequence 0 starts at position 0. A subsequent ordinary decode starts at the prompt length. Unsupported configurations return `-1`; resource and execution failures return `-2`, with an explicit log message.

The defaults are BF16 staging, microbatch 256, 16 GiB of host activation storage, 6 GiB of expert storage, and 8 GiB of scheduler compute buffers. C++ callers can set these limits through `llama_layered_prefill_params`. CPU execution requires `require_gpu = false` and is intended for correctness testing. Only one GPU is supported.

The separate `llama-layered-prefill` example uses the shared llama.cpp argument parser, keeps routed experts in plain CPU buffers, invokes the new path, reports copy statistics and then performs greedy ordinary decode. `--staging-type f32` selects the unrounded reference; the default is `bf16`. It raises the logical batch capacity to hold the complete prompt; `-ub` remains the physical microbatch bound. It does not enable the feature in the normal CLI or server.

```powershell
.\build-prefill-cuda\bin\llama-layered-prefill.exe -m D:\models\glm-q4.gguf -f D:\prompts\input.txt --device CUDA0 -ngl 99 -c 200064 -ub 256 -fa on -ctk f16 -ctv f16 -n 32
```

The command requires a suitable real GGUF and is not a verified 200K/20GB configuration. Expert tensors must be contiguous, host-resident and in their ordinary GGUF layout. Repacked CPU-only expert buffers are rejected. The scheduler must place expert operations and staging conversions on the selected backend; otherwise the call fails instead of presenting a CPU fallback as GPU acceleration.

The current staging transfers are synchronous and use pageable host storage. Pinned bounce buffers, overlap, graph reuse and automatic microbatch selection are future optimizations, not implemented capabilities.

## Reference

- [Official GLM-5.3-Flash configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)
- [ik_llama.cpp expert cache](https://github.com/ikawrakow/ik_llama.cpp/blob/main/src/llama-moe-cache.cpp)
- Local implementation surfaces: `src/models/glm5-next.cpp`, `src/llama-context.cpp`, `src/llama-memory-hybrid-idx.cpp`, `src/llama-memory-recurrent.cpp`, `src/llama-moe-cache.cpp`.

The ik fork's cache uses layout-compatible device banks, expert slot maps and upload callbacks. Its LRU policy alone does not implement layer-major whole-input prefill. Reuse upstream GGUF and backend operations rather than importing fork-specific quantization formats or kernels without compatibility checks.
