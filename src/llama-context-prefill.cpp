#include "llama-context.h"

#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

size_t prefill_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error("layered prefill size overflow");
    }
    return a + b;
}

struct prefill_pipeline {
    struct slot {
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        uint8_t * host_input = nullptr;
        uint8_t * host_output = nullptr;
        ggml_backend_event_ptr input_ready;
        ggml_backend_event_ptr output_ready;
        uint8_t * destination = nullptr;
        size_t input_bytes = 0;
        size_t output_bytes = 0;
    };

    ggml_backend_t compute;
    ggml_backend_ptr upload;
    ggml_backend_ptr download;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr device_buffer;
    ggml_backend_buffer_ptr host_buffer;
    std::array<slot, 2> slots;

    prefill_pipeline(ggml_backend_t backend, const llama_hparams & hp, uint32_t n_tokens,
                     const llama_layered_prefill_params & params, size_t host_budget,
                     llama_layered_prefill_stats & stats) : compute(backend) {
        const auto dev = ggml_backend_get_device(backend);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        const auto host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!props.caps.async || !props.caps.events || !props.caps.host_buffer || !host_buft) {
            throw std::runtime_error("layered prefill pipeline requires asynchronous copies, events and pinned host buffers");
        }
        const size_t row = ggml_row_size(params.staging_type, int64_t(hp.n_embd) * hp.dsv4_hc_mult);
        if (n_tokens > host_budget / 4 / row || n_tokens > params.max_transfer_bytes / 8 / row) {
            throw std::runtime_error("layered prefill pipeline exceeds its host or transfer buffer budget");
        }
        const size_t block = row * n_tokens;
        ctx.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw std::runtime_error("failed to allocate layered prefill pipeline metadata");
        }
        ggml_tensor * tensors[4];
        for (size_t i = 0; i < slots.size(); ++i) {
            auto & s = slots[i];
            s.input = tensors[2*i] = ggml_new_tensor_3d(ctx.get(), params.staging_type,
                                                       hp.n_embd, hp.dsv4_hc_mult, n_tokens);
            s.output = tensors[2*i + 1] = ggml_dup_tensor(ctx.get(), s.input);
            ggml_format_name(s.input, "prefill_pipeline.input.%zu", i);
            ggml_format_name(s.output, "prefill_pipeline.output.%zu", i);
            s.input_ready.reset(ggml_backend_event_new(dev));
            s.output_ready.reset(ggml_backend_event_new(dev));
            if (!s.input_ready || !s.output_ready) {
                throw std::runtime_error("failed to create layered prefill transfer events");
            }
        }
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t device_bytes = ggml_backend_buft_get_alloc_size_n(buft, tensors, 4);
        if (prefill_add(device_bytes, 4*block) > params.max_transfer_bytes) {
            throw std::runtime_error("layered prefill pipeline exceeds max_transfer_bytes");
        }
        device_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        host_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, 4*block));
        if (!device_buffer || !host_buffer || ggml_backend_buffer_get_type(host_buffer.get()) != host_buft) {
            throw std::runtime_error("failed to allocate layered prefill device or pinned staging buffers");
        }
        const size_t host_bytes = ggml_backend_buffer_get_size(host_buffer.get());
        const size_t all_bytes = prefill_add(ggml_backend_buffer_get_size(device_buffer.get()), host_bytes);
        if (host_bytes > host_budget || all_bytes > params.max_transfer_bytes) {
            throw std::runtime_error("allocated layered prefill pipeline exceeds its budget");
        }
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(host_buffer.get()));
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i].host_input = base + (2*i)*block;
            slots[i].host_output = base + (2*i + 1)*block;
        }
        upload.reset(ggml_backend_dev_init(dev, nullptr));
        download.reset(ggml_backend_dev_init(dev, nullptr));
        if (!upload || !download) {
            throw std::runtime_error("failed to initialize layered prefill transfer streams");
        }
        stats.pipeline_active = true;
        stats.transfer_buffer_bytes = all_bytes;
    }

    ~prefill_pipeline() {
        ggml_backend_synchronize(compute);
        if (upload) {
            ggml_backend_synchronize(upload.get());
        }
        if (download) {
            ggml_backend_synchronize(download.get());
        }
    }

    void finish_output(size_t i) {
        auto & s = slots[i];
        if (s.destination) {
            ggml_backend_event_synchronize(s.output_ready.get());
            std::memcpy(s.destination, s.host_output, s.output_bytes);
            s.destination = nullptr;
        }
    }

    void prefetch_input(size_t i, const uint8_t * source, size_t bytes) {
        auto & s = slots[i];
        if (s.input_bytes) {
            ggml_backend_event_synchronize(s.input_ready.get());
        }
        std::memcpy(s.host_input, source, bytes);
        ggml_backend_tensor_set_async(upload.get(), s.input, s.host_input, 0, bytes);
        ggml_backend_event_record(s.input_ready.get(), upload.get());
        s.input_bytes = bytes;
    }

    void wait_input(size_t i, size_t bytes) {
        auto & s = slots[i];
        if (s.input_bytes != bytes) {
            throw std::runtime_error("layered prefill pipeline requires consecutive fixed-size microbatches");
        }
        ggml_backend_event_wait(compute, s.input_ready.get());
    }

    void copy_output(size_t i, uint8_t * destination, size_t bytes) {
        auto & s = slots[i];
        ggml_backend_tensor_get_async(download.get(), s.output, s.host_output, 0, bytes);
        ggml_backend_event_record(s.output_ready.get(), download.get());
        s.destination = destination;
        s.output_bytes = bytes;
    }

    void drain() {
        ggml_backend_synchronize(upload.get());
        for (size_t i = 0; i < slots.size(); ++i) {
            finish_output(i);
            slots[i].input_bytes = 0;
        }
    }
};

// Device weights outlive every microbatch graph of the current layer.
struct prefill_experts {
    ggml_backend_t backend;
    llama_layer layer;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;

    static std::array<ggml_tensor *, 3> tensors(const llama_layer & layer) {
        return { layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps };
    }

    static size_t allocation_size(const llama_layer & layer, ggml_backend_t backend) {
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t bytes = 0;
        for (const auto * w : tensors(layer)) {
            if (!w || !w->buffer || !w->data || !ggml_is_contiguous(w) || !ggml_backend_buffer_is_host(w->buffer)) {
                throw std::runtime_error("layered prefill requires contiguous host-resident routed experts");
            }
            const size_t n = ggml_backend_buft_get_alloc_size(buft, w);
            bytes = prefill_add(bytes, prefill_add(n, alignment - 1) / alignment * alignment);
        }
        return bytes;
    }

    prefill_experts(const llama_layer & source, ggml_backend_t backend, size_t budget,
                    llama_layered_prefill_stats & stats, bool deferred = false, bool optional = false) :
        backend(backend), layer(source) {
        if (!source.ffn_gate_exps) {
            return;
        }
        const auto weights = tensors(source);
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t bytes = allocation_size(source, backend);
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
            buffer.reset();
            if (optional) {
                return;
            }
            throw std::runtime_error("failed to allocate bounded layered prefill expert buffer");
        }
        ggml_backend_buffer_clear(buffer.get(), 0);
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (!deferred) {
            for (int i = 0; i < 3; ++i) {
                ggml_backend_tensor_set(*copies[i], weights[i]->data, 0, ggml_nbytes(weights[i]));
                stats.expert_copy_bytes += ggml_nbytes(weights[i]);
            }
            ++stats.expert_layer_loads;
        }
    }

    ~prefill_experts() {
        ggml_backend_synchronize(backend);
    }
};

struct prefill_weight_prefetch {
    std::unique_ptr<prefill_experts> experts;
    std::array<ggml_tensor *, 3> source;
    ggml_backend_ptr upload;
    ggml_backend_buffer_ptr pinned;
    std::array<ggml_backend_event_ptr, 2> ready;
    std::array<bool, 2> pending = {};
    size_t block;
    size_t part = 0;
    size_t offset = 0;
    size_t submissions = 0;
    size_t copied = 0;
    size_t quota;

    prefill_weight_prefetch(std::unique_ptr<prefill_experts> bank, const llama_layer & layer,
                            size_t block, size_t steps) :
        experts(std::move(bank)), source(prefill_experts::tensors(layer)), block(block) {
        size_t total = 0;
        for (auto * t : source) {
            total = prefill_add(total, ggml_nbytes(t));
        }
        quota = std::max<size_t>(1 << 20, total / steps + (total % steps != 0));
        const auto dev = ggml_backend_get_device(experts->backend);
        const auto buft = ggml_backend_dev_host_buffer_type(dev);
        pinned.reset(ggml_backend_buft_alloc_buffer(buft, 2*block));
        if (!pinned || ggml_backend_buffer_get_type(pinned.get()) != buft) {
            return;
        }
        upload.reset(ggml_backend_dev_init(dev, nullptr));
        for (auto & event : ready) {
            event.reset(ggml_backend_event_new(dev));
        }
    }

    ~prefill_weight_prefetch() {
        if (upload) {
            ggml_backend_synchronize(upload.get());
        }
    }

    bool available() const {
        return pinned && upload && ready[0] && ready[1];
    }

    size_t advance(size_t bytes) {
        const size_t before = copied;
        const auto dest = prefill_experts::tensors(experts->layer);
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get()));
        while (part < source.size() && bytes > 0) {
            const size_t i = submissions % ready.size();
            if (pending[i]) {
                ggml_backend_event_synchronize(ready[i].get());
            }
            const size_t n = std::min({bytes, block, ggml_nbytes(source[part]) - offset});
            auto * bounce = base + i*block;
            std::memcpy(bounce, static_cast<const uint8_t *>(source[part]->data) + offset, n);
            ggml_backend_tensor_set_async(upload.get(), dest[part], bounce, offset, n);
            ggml_backend_event_record(ready[i].get(), upload.get());
            pending[i] = true;
            ++submissions;
            copied += n;
            bytes -= n;
            offset += n;
            if (offset == ggml_nbytes(source[part])) {
                ++part;
                offset = 0;
            }
        }
        return copied - before;
    }

    void finish(llama_layered_prefill_stats & stats) {
        advance(std::numeric_limits<size_t>::max());
        ggml_backend_synchronize(upload.get());
        stats.expert_copy_bytes += copied;
        stats.prefetched_expert_bytes += copied;
        ++stats.expert_layer_loads;
    }
};

std::unique_ptr<prefill_weight_prefetch> prefill_try_prefetch(
        const llama_layer & layer, ggml_backend_t device, ggml_backend_sched_t sched,
        const llama_layered_prefill_params & params, size_t host_available, size_t transfer_available,
        size_t steps, llama_layered_prefill_stats & stats) {
    if (!layer.ffn_gate_exps) {
        return nullptr;
    }
    const size_t bytes = prefill_experts::allocation_size(layer, device);
    size_t payload = 0;
    for (auto * t : prefill_experts::tensors(layer)) {
        payload = prefill_add(payload, ggml_nbytes(t));
    }
    const size_t block = std::min<size_t>(8 << 20, payload);
    size_t free_bytes = 0, total_bytes = 0;
    const auto dev = ggml_backend_get_device(device);
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    const size_t allocated_compute = ggml_backend_sched_get_buffer_size(sched, device);
    const size_t compute_growth = params.max_compute_bytes > allocated_compute ?
        params.max_compute_bytes - allocated_compute : 0;
    const bool fits = bytes <= params.max_prefetch_bytes && bytes <= params.max_expert_bytes &&
        bytes <= free_bytes && params.prefetch_headroom_bytes <= free_bytes - bytes &&
        compute_growth <= free_bytes - bytes - params.prefetch_headroom_bytes &&
        2*block <= host_available && 2*block <= transfer_available &&
        props.caps.async && props.caps.events && props.caps.host_buffer &&
        ggml_backend_dev_host_buffer_type(dev) != nullptr;
    if (!fits) {
        ++stats.weight_prefetch_skips;
        LLAMA_LOG_DEBUG("%s: skip optional next-layer prefetch: weights=%zu, free=%zu, compute growth=%zu, headroom=%zu\n",
                       __func__, bytes, free_bytes, compute_growth, params.prefetch_headroom_bytes);
        return nullptr;
    }
    ++stats.weight_prefetch_allocation_attempts;
    auto bank = std::make_unique<prefill_experts>(layer, device,
        std::min(params.max_prefetch_bytes, params.max_expert_bytes), stats, true, true);
    if (!bank->buffer) {
        ++stats.weight_prefetch_skips;
        LLAMA_LOG_WARN("%s: optional prefetch allocation failed after memory check; using normal layer loading\n", __func__);
        return nullptr;
    }
    auto prefetch = std::make_unique<prefill_weight_prefetch>(std::move(bank), layer, block, steps);
    if (!prefetch->available() || ggml_backend_buffer_get_size(prefetch->pinned.get()) > 2*block) {
        ++stats.weight_prefetch_skips;
        LLAMA_LOG_WARN("%s: optional pinned prefetch staging unavailable; using normal layer loading\n", __func__);
        return nullptr;
    }
    ++stats.weight_prefetches;
    stats.peak_prefetch_bytes = std::max(stats.peak_prefetch_bytes,
                                        ggml_backend_buffer_get_size(prefetch->experts->buffer.get()));
    stats.prefetch_pinned_bytes = std::max(stats.prefetch_pinned_bytes,
                                          ggml_backend_buffer_get_size(prefetch->pinned.get()));
    return prefetch;
}

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
    const bool pipelined = params.pipeline && device != backend_cpu;

    llama_layered_prefill_stats stats;
    bool started = false;
    try {
        const size_t row_bytes = ggml_row_size(params.staging_type, (int64_t) hp.n_embd * hp.dsv4_hc_mult);
        if ((size_t) n_tokens > params.max_host_bytes / 2 / row_bytes) {
            throw std::runtime_error("layered prefill activation buffers exceed max_host_bytes");
        }
        const size_t total_bytes = row_bytes * n_tokens;
        std::vector<uint8_t> input(total_bytes), output(total_bytes);
        std::unique_ptr<prefill_pipeline> pipeline;
        if (pipelined) {
            pipeline = std::make_unique<prefill_pipeline>(device, hp, std::min<uint32_t>(params.n_ubatch, n_tokens),
                                                         params, params.max_host_bytes - 2*total_bytes, stats);
        }
        std::unique_ptr<prefill_weight_prefetch> next_experts;
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
            std::unique_ptr<prefill_experts> experts;
            if (next_experts) {
                next_experts->finish(stats);
                experts = std::move(next_experts->experts);
                next_experts.reset();
            } else {
                experts = std::make_unique<prefill_experts>(model.layers[il], device, params.max_expert_bytes, stats);
            }
            size_t offset = 0;
            size_t i_batch = 0;
            if (pipeline && il > 0) {
                pipeline->prefetch_input(0, input.data(), row_bytes * mctx->get_ubatch().n_tokens);
            }
            do {
                const auto & ubatch = mctx->get_ubatch();
                const size_t slot = i_batch % 2;
                if (pipeline) {
                    pipeline->finish_output(slot);
                }
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
                gparams.prefill_weights = &experts->layer;
                if (pipeline) {
                    gparams.prefill_input = il > 0 ? pipeline->slots[slot].input : nullptr;
                    gparams.prefill_output = pipeline->slots[slot].output;
                }
                auto * gf = model.build_graph(gparams);
                if (!gf) {
                    throw std::runtime_error("failed to build layered prefill graph");
                }
                std::vector<size_t> sizes(backend_ptrs.size());
                ggml_backend_sched_reserve_size(sched.get(), gf, sizes.data());
                prefill_check_graph(sched.get(), gf, experts->layer, device, *res);
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
                prefill_check_graph(sched.get(), gf, experts->layer, device, *res);
                res->set_inputs(&ubatch);
                const size_t bytes = row_bytes * ubatch.n_tokens;
                if (res->t_prefill_in) {
                    if (pipeline) {
                        pipeline->wait_input(slot, bytes);
                    } else {
                        ggml_backend_tensor_set(res->t_prefill_in, input.data() + offset * row_bytes, 0, bytes);
                    }
                    stats.activation_read_bytes += bytes;
                }
                if (i_batch == 0 && params.weight_prefetch && device != backend_cpu && il + 1 < hp.n_layer()) {
                    const size_t pinned_bytes = pipeline ? ggml_backend_buffer_get_size(pipeline->host_buffer.get()) : 0;
                    next_experts = prefill_try_prefetch(model.layers[il + 1], device, sched.get(), params,
                        params.max_host_bytes - 2*total_bytes - pinned_bytes,
                        params.max_transfer_bytes > stats.transfer_buffer_bytes ?
                            params.max_transfer_bytes - stats.transfer_buffer_bytes : 0,
                        ((size_t) n_tokens + params.n_ubatch - 1) / params.n_ubatch, stats);
                }
                const auto status = graph_compute(gf, true);
                const size_t next = offset + ubatch.n_tokens;
                if (pipeline && il > 0 && next < (size_t) n_tokens && status == GGML_STATUS_SUCCESS) {
                    const size_t next_bytes = row_bytes * std::min<size_t>(params.n_ubatch, n_tokens - next);
                    pipeline->prefetch_input((i_batch + 1) % 2, input.data() + next * row_bytes, next_bytes);
                    ++stats.input_prefetches;
                }
                if (next_experts && status == GGML_STATUS_SUCCESS) {
                    stats.weight_prefetch_submit_bytes += next_experts->advance(next_experts->quota);
                }
                synchronize();
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("layered prefill computation failed or was aborted");
                }
                if (pipeline) {
                    pipeline->copy_output(slot, output.data() + offset * row_bytes, bytes);
                    ++stats.async_output_batches;
                } else {
                    ggml_backend_tensor_get(res->t_prefill_out, output.data() + offset * row_bytes, 0, bytes);
                }
                stats.activation_write_bytes += bytes;
                if (n_outputs) {
                    ggml_backend_tensor_get(res->t_logits, logits.data, 0, model.vocab.n_tokens() * sizeof(float));
                }
                offset += ubatch.n_tokens;
                ++stats.microbatches;
                ++i_batch;
            } while (mctx->next());
            if (offset != (size_t) n_tokens) {
                throw std::runtime_error("layered prefill did not consume the complete prompt");
            }
            if (pipeline) {
                pipeline->drain();
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
