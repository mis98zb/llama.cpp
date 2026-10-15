#include "llama-context.h"

#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

size_t prefill_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error("layered prefill size overflow");
    }
    return a + b;
}

// Device weights outlive every microbatch graph of the current layer.
struct prefill_experts {
    llama_layer layer;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;

    prefill_experts(const llama_layer & source, ggml_backend_t backend, size_t budget,
                    llama_layered_prefill_stats & stats) : layer(source) {
        if (!source.ffn_gate_exps) {
            return;
        }
        const ggml_tensor * weights[] = { source.ffn_gate_exps, source.ffn_up_exps, source.ffn_down_exps };
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        size_t bytes = 0;
        for (const auto * w : weights) {
            if (!w || !w->buffer || !w->data || !ggml_is_contiguous(w) || !ggml_backend_buffer_is_host(w->buffer)) {
                throw std::runtime_error("layered prefill requires contiguous host-resident routed experts");
            }
            const size_t alignment = ggml_backend_buft_get_alignment(buft);
            const size_t padded = prefill_add(ggml_backend_buft_get_alloc_size(buft, w), alignment - 1) / alignment * alignment;
            bytes = prefill_add(bytes, padded);
        }
        if (bytes > budget) {
            throw std::runtime_error("layered prefill expert buffer exceeds max_expert_bytes");
        }
        ctx.reset(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw std::runtime_error("failed to allocate layered prefill expert metadata");
        }
        ggml_tensor ** copies[] = { &layer.ffn_gate_exps, &layer.ffn_up_exps, &layer.ffn_down_exps };
        for (int i = 0; i < 3; ++i) {
            *copies[i] = ggml_dup_tensor(ctx.get(), weights[i]);
            ggml_set_name(*copies[i], weights[i]->name);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buffer || ggml_backend_buffer_get_size(buffer.get()) > budget) {
            throw std::runtime_error("failed to allocate bounded layered prefill expert buffer");
        }
        ggml_backend_buffer_clear(buffer.get(), 0);
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_set(*copies[i], weights[i]->data, 0, ggml_nbytes(weights[i]));
            stats.expert_copy_bytes += ggml_nbytes(weights[i]);
        }
        ++stats.expert_layer_loads;
    }
};

void prefill_check_graph(ggml_backend_sched_t sched, ggml_cgraph * gf,
                         const llama_layer & layer, ggml_backend_t device,
                         const llm_graph_result & res) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        const bool expert = node->op == GGML_OP_MUL_MAT_ID &&
            (node->src[0] == layer.ffn_gate_exps ||
             node->src[0] == layer.ffn_up_exps ||
             node->src[0] == layer.ffn_down_exps);
        const bool conversion = node == res.t_prefill_out ||
            (res.t_prefill_in && node->src[0] == res.t_prefill_in);
        if ((expert || conversion) && ggml_backend_sched_get_tensor_backend(sched, node) != device) {
            throw std::runtime_error("resident experts and staging conversions must execute on the selected backend");
        }
    }
}

}

int llama_context::prefill_layered(
        const llama_token * tokens, int32_t n_tokens,
        const llama_layered_prefill_params & params, llama_layered_prefill_stats * stats_out) {
    if (stats_out) {
        *stats_out = {};
    }
    const auto & hp = model.hparams;
    if (!tokens || n_tokens <= 0 || model.arch != LLM_ARCH_GLM5_NEXT ||
        !memory || cparams.n_seq_max != 1 || n_tokens > (int64_t) cparams.n_ctx ||
        n_tokens > (int64_t) cparams.n_batch || params.n_ubatch == 0 ||
        params.n_ubatch > cparams.n_ubatch || params.max_host_bytes == 0 ||
        params.max_expert_bytes == 0 || params.max_compute_bytes == 0 ||
        (params.staging_type != GGML_TYPE_F32 && params.staging_type != GGML_TYPE_BF16) ||
        cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || !cparams.causal_attn ||
        cparams.embeddings || cparams.embeddings_nextn || cparams.n_rs_seq != 0 || cparams.cb_eval ||
        !sampling.samplers.empty() || !loras->empty() || moe_cache || !model.hdmd.rot.empty() ||
        std::any_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                    [](bool enabled) { return enabled; })) {
        LLAMA_LOG_ERROR("%s: unsupported request or configuration; see docs/development/layered-prefill.md\n", __func__);
        return -1;
    }
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if ((!hp.is_recr(il) && !hp.is_indexer_full(il)) || cvec->tensor_for(il)) {
            LLAMA_LOG_ERROR("%s: shared indexers and control vectors are not supported\n", __func__);
            return -1;
        }
    }
    if (memory->seq_pos_max(0) >= 0) {
        LLAMA_LOG_ERROR("%s: requires an empty context\n", __func__);
        return -1;
    }
    for (int32_t i = 0; i < n_tokens; ++i) {
        if (tokens[i] < 0 || tokens[i] >= model.vocab.n_tokens()) {
            LLAMA_LOG_ERROR("%s: invalid token at position %d\n", __func__, i);
            return -1;
        }
    }
    ggml_backend_t device = nullptr;
    for (auto backend : backend_ptrs) {
        const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            if (device) {
                LLAMA_LOG_ERROR("%s: only a single GPU is supported\n", __func__);
                return -1;
            }
            device = backend;
        }
    }
    if (!device && params.require_gpu) {
        LLAMA_LOG_ERROR("%s: no GPU backend; CPU execution must be explicitly requested for testing\n", __func__);
        return -1;
    }
    if (!device) {
        device = backend_cpu;
    }

    llama_layered_prefill_stats stats;
    bool started = false;
    try {
        const size_t row_bytes = ggml_row_size(params.staging_type, (int64_t) hp.n_embd * hp.dsv4_hc_mult);
        if ((size_t) n_tokens > params.max_host_bytes / 2 / row_bytes) {
            throw std::runtime_error("layered prefill activation buffers exceed max_host_bytes");
        }
        const size_t total_bytes = row_bytes * n_tokens;
        std::vector<uint8_t> input(total_bytes), output(total_bytes);
        llama_batch_ext batch(this);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const auto idx = batch.add_token(0);
            const llama_pos pos = i;
            if (idx < 0 || !batch.set_token_id(idx, tokens[i]) ||
                !batch.set_token_pos(idx, &pos) || !batch.set_output(idx, i == n_tokens - 1)) {
                throw std::runtime_error("failed to construct layered prefill batch");
            }
        }
        synchronize();
        const int64_t start_us = ggml_time_us();
        sched_reserve();
        if (output_reserve(1) < 1) {
            throw std::runtime_error("failed to reserve layered prefill logits");
        }
        started = true;
        for (uint32_t il = 0; il < hp.n_layer(); ++il) {
            ggml_backend_sched_reset(sched.get());
            gf_res_prev_active = nullptr;
            memory->clear(false);
            if (!balloc->init(batch, model.vocab, false)) {
                throw std::runtime_error("failed to initialize layered prefill batch");
            }
            auto mctx = memory->init_batch(*balloc, params.n_ubatch, false);
            if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
                throw std::runtime_error("failed to prepare layered prefill memory");
            }
            prefill_experts experts(model.layers[il], device, params.max_expert_bytes, stats);
            size_t offset = 0;
            do {
                const auto & ubatch = mctx->get_ubatch();
                if (!mctx->apply()) {
                    throw std::runtime_error("failed to apply layered prefill memory");
                }
                n_outputs = il + 1 == hp.n_layer() && offset + ubatch.n_tokens == (size_t) n_tokens ? 1 : 0;
                auto * res = get_gf_res_prev();
                ggml_backend_sched_reset(sched.get());
                res->reset();
                auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DECODER);
                gparams.moe_cache = nullptr;
                gparams.prefill_layer = il;
                gparams.prefill_type = params.staging_type;
                gparams.prefill_weights = &experts.layer;
                auto * gf = model.build_graph(gparams);
                if (!gf) {
                    throw std::runtime_error("failed to build layered prefill graph");
                }
                std::vector<size_t> sizes(backend_ptrs.size());
                ggml_backend_sched_reserve_size(sched.get(), gf, sizes.data());
                prefill_check_graph(sched.get(), gf, experts.layer, device, *res);
                size_t compute_bytes = 0;
                for (size_t bytes : sizes) {
                    compute_bytes = prefill_add(compute_bytes, bytes);
                }
                if (compute_bytes > params.max_compute_bytes) {
                    throw std::runtime_error("layered prefill workspace exceeds max_compute_bytes; reduce n_ubatch");
                }
                stats.peak_compute_bytes = std::max(stats.peak_compute_bytes, compute_bytes);
                // Splitting a measurement graph rewrites sources to scheduler-owned copies.
                ggml_backend_sched_reset(sched.get());
                res->reset();
                gf = model.build_graph(gparams);
                if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                    throw std::runtime_error("failed to allocate layered prefill graph");
                }
                prefill_check_graph(sched.get(), gf, experts.layer, device, *res);
                res->set_inputs(&ubatch);
                const size_t bytes = row_bytes * ubatch.n_tokens;
                if (res->t_prefill_in) {
                    ggml_backend_tensor_set(res->t_prefill_in, input.data() + offset * row_bytes, 0, bytes);
                    stats.activation_read_bytes += bytes;
                }
                const auto status = graph_compute(gf, true);
                synchronize();
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("layered prefill computation failed or was aborted");
                }
                ggml_backend_tensor_get(res->t_prefill_out, output.data() + offset * row_bytes, 0, bytes);
                stats.activation_write_bytes += bytes;
                if (n_outputs) {
                    ggml_backend_tensor_get(res->t_logits, logits.data, 0, model.vocab.n_tokens() * sizeof(float));
                }
                offset += ubatch.n_tokens;
                ++stats.microbatches;
            } while (mctx->next());
            if (offset != (size_t) n_tokens) {
                throw std::runtime_error("layered prefill did not consume the complete prompt");
            }
            ggml_backend_sched_reset(sched.get());
            for (auto & res : gf_res_prev) {
                if (res) {
                    res->reset();
                }
            }
            input.swap(output);
        }
        n_outputs = 1;
        std::fill(output_ids.begin(), output_ids.end(), -1);
        output_ids[n_tokens - 1] = 0;
        output_swaps.clear();
        embd_seq.clear();
        embd_batch_idxs.clear();
        t_compute_start_us = start_us;
        n_queued_tokens = n_tokens;
        synchronize();
        if (stats_out) {
            *stats_out = stats;
        }
        return 0;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        if (started) {
            synchronize();
            ggml_backend_sched_reset(sched.get());
            gf_res_prev_active = nullptr;
            for (auto & res : gf_res_prev) {
                if (res) {
                    res->reset();
                }
            }
            memory->clear(true);
            n_outputs = 0;
            std::fill(output_ids.begin(), output_ids.end(), -1);
        }
        return -2;
    }
}

int32_t llama_prefill_layered(
        llama_context * ctx, const llama_token * tokens, int32_t n_tokens,
        const llama_layered_prefill_params & params, llama_layered_prefill_stats * stats) {
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: null context\n", __func__);
        return -1;
    }
    return ctx->prefill_layered(tokens, n_tokens, params, stats);
}
