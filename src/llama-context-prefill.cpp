#include "llama-context.h"

#include "llama-batch.h"
#include "llama-model.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-prefill-plan.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

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

    static size_t projected_device_bytes(ggml_backend_t backend, const llama_hparams & hp,
                                         uint32_t tokens, ggml_type type) {
        ggml_context_ptr metadata(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        if (!metadata) {
            throw llama_prefill_resource_error("failed to measure activation staging metadata");
        }
        std::array<ggml_tensor *, 4> tensors;
        for (auto & tensor : tensors) {
            tensor = ggml_new_tensor_3d(metadata.get(), type, hp.n_embd, hp.dsv4_hc_mult, tokens);
        }
        return ggml_backend_buft_get_alloc_size_n(ggml_backend_get_default_buffer_type(backend), tensors.data(), 4);
    }

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
            throw llama_prefill_resource_error("layered prefill pipeline exceeds its host or transfer buffer budget");
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
            throw llama_prefill_resource_error("layered prefill pipeline exceeds max_transfer_bytes");
        }
        device_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        host_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, 4*block));
        if (!device_buffer || !host_buffer || ggml_backend_buffer_get_type(host_buffer.get()) != host_buft) {
            throw llama_prefill_resource_error("failed to allocate layered prefill device or pinned staging buffers");
        }
        const size_t host_bytes = ggml_backend_buffer_get_size(host_buffer.get());
        const size_t all_bytes = prefill_add(ggml_backend_buffer_get_size(device_buffer.get()), host_bytes);
        if (host_bytes > host_budget || all_bytes > params.max_transfer_bytes) {
            throw llama_prefill_resource_error("allocated layered prefill pipeline exceeds its budget");
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
            throw llama_prefill_resource_error("layered prefill expert buffer exceeds max_expert_bytes");
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
            throw llama_prefill_resource_error("failed to allocate bounded layered prefill expert buffer");
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
        size_t steps, llama_layered_prefill_stats & stats, size_t compute_reserve = 0) {
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
    const size_t ceiling = compute_reserve ? compute_reserve : params.max_compute_bytes;
    const size_t compute_growth = ceiling > allocated_compute ? ceiling - allocated_compute : 0;
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
                         const std::vector<llama_layer> & layers, ggml_backend_t device,
                         const llm_graph_result & res) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        bool expert = false;
        if (node->op == GGML_OP_MUL_MAT_ID) {
            for (const auto & layer : layers) {
                expert |= node->src[0] == layer.ffn_gate_exps ||
                          node->src[0] == layer.ffn_up_exps ||
                          node->src[0] == layer.ffn_down_exps;
            }
        }
        const bool conversion = node == res.t_prefill_out ||
            (res.t_prefill_in && node->src[0] == res.t_prefill_in);
        if ((expert || conversion) && ggml_backend_sched_get_tensor_backend(sched, node) != device) {
            throw std::runtime_error("resident experts and staging conversions must execute on the selected backend");
        }
    }
}

uint32_t prefill_family(const llama_model & model, uint32_t il) {
    const auto & hp = model.hparams;
    const bool cpu = ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    return (cpu ? 4 : 0) + (il < hp.n_layer_dense_lead ? 0 : 2) + (hp.is_recr(il) ? 0 : 1);
}

llama_prefill_cost prefill_fit_cost(
        const llama_prefill_probe & probe, const llama_model & model, uint32_t batch, uint32_t tokens) {
    const auto & hp = model.hparams;
    llama_prefill_cost cost;
    cost.sample_tokens = tokens;
    cost.read_bytes_per_ms = probe.read_bytes / std::max(0.001, probe.read_ms);
    cost.write_bytes_per_ms = probe.write_bytes / std::max(0.001, probe.write_ms);
    cost.weight_bytes_per_ms = probe.weight_bytes / std::max(0.001, probe.weight_ms);
    std::array<uint32_t, 8> counts = {};
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        std::vector<const llama_prefill_probe_point *> points;
        for (const auto & point : probe.points) {
            if (point.layer == il && point.tokens == batch) {
                points.push_back(&point);
                cost.min_prefix = cost.min_prefix ? std::min(cost.min_prefix, point.prefix) : point.prefix;
            }
        }
        if (points.empty()) {
            continue;
        }
        const auto & first = *points[points.size() > 2 ? 1 : 0];
        const auto & last = *points.back();
        const uint32_t family = prefill_family(model, il);
        const double span = last.prefix > first.prefix ? last.prefix - first.prefix : 0;
        const double compute_slope = !hp.is_recr(il) && span > 0 ?
            std::max(0.0, (last.compute_ms - first.compute_ms) / span) : 0;
        const double setup_slope = span > 0 ? std::max(0.0, (last.setup_ms - first.setup_ms) / span) : 0;
        cost.compute_ms[family] += std::max(0.001, last.compute_ms - compute_slope * last.prefix);
        cost.compute_prefix_ms[family] += compute_slope;
        cost.setup_ms[family] += std::max(0.0, last.setup_ms - setup_slope * last.prefix);
        cost.setup_prefix_ms[family] += setup_slope;
        ++counts[family];
        cost.measured[family] = true;
    }
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i]) {
            cost.compute_ms[i] /= counts[i];
            cost.compute_prefix_ms[i] /= counts[i];
            cost.setup_ms[i] /= counts[i];
            cost.setup_prefix_ms[i] /= counts[i];
        }
    }
    return cost;
}

}

int llama_context::prefill_layered(
        const llama_token * tokens, int32_t n_tokens,
        const llama_layered_prefill_params & params, llama_layered_prefill_stats * stats_out) {
    if (stats_out) {
        *stats_out = {};
    }
    const auto & hp = model.hparams;
    if (!tokens || n_tokens <= 0 || model.arch != LLM_ARCH_GLM5_NEXT || hp.no_alloc ||
        !memory || cparams.n_seq_max != 1 || n_tokens > (int64_t) cparams.n_ctx ||
        n_tokens > (int64_t) cparams.n_batch || params.n_ubatch == 0 ||
        params.n_ubatch > cparams.n_ubatch || params.max_host_bytes == 0 ||
        params.max_expert_bytes == 0 || params.max_compute_bytes == 0 ||
        params.group_layers == 0 || params.max_group_layers == 0 ||
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
    auto * hybrid = static_cast<llama_memory_hybrid_idx *>(memory.get());
    if (!hybrid->has_complete_prefix(0)) {
        LLAMA_LOG_ERROR("%s: MLA/indexer/recurrent prefix metadata are incomplete, inconsistent, or have pending updates\n", __func__);
        return -1;
    }
    const int64_t prefix = int64_t(memory->seq_pos_max(0)) + 1;
    if (prefix > 0) {
        if (!params.reuse_prefix) {
            LLAMA_LOG_ERROR("%s: nonempty context requires complete MLA/indexer/KDA/convolution prefix state and reuse_prefix\n", __func__);
            return -1;
        }
        if (prefix + n_tokens > cparams.n_ctx || prefix + n_tokens > std::numeric_limits<llama_pos>::max()) {
            LLAMA_LOG_ERROR("%s: prefix plus suffix exceeds the context capacity\n", __func__);
            return -1;
        }
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
    if (params.auto_plan) {
        return prefill_layered_plan(tokens, n_tokens, params, stats_out, device);
    }
    std::vector<llama_layered_prefill_group> groups;
    for (uint32_t il = 0; il < hp.n_layer();) {
        const uint32_t count = std::min(params.group_layers, hp.n_layer() - il);
        groups.push_back({il, count, 0});
        il += count;
    }
    if (params.weight_prefetch) {
        for (size_t i = 0; i + 1 < groups.size(); ++i) {
            groups[i].n_prefetch_layers = std::min(params.max_prefetch_layers, groups[i + 1].n_layers);
        }
    }
    return prefill_layered_run(tokens, n_tokens, params, stats_out, device, groups);
}

int llama_context::prefill_layered_run(
        const llama_token * tokens, int32_t n_tokens,
        const llama_layered_prefill_params & params, llama_layered_prefill_stats * stats_out,
        ggml_backend_t device, const std::vector<llama_layered_prefill_group> & groups,
        llama_prefill_probe * probe) {
    const auto & hp = model.hparams;
    const bool pipelined = params.pipeline && device != backend_cpu;

    llama_layered_prefill_stats stats;
    stats.selected_ubatch = params.n_ubatch;
    stats.groups = groups;
    const llama_pos prefix = memory->seq_pos_max(0) + 1;
    stats.reused_prefix_tokens = prefix;
    stats.evaluated_tokens = n_tokens;
    size_t compute_ceiling = 0, device_ceiling = 0;
    for (const auto & group : groups) {
        compute_ceiling = std::max(compute_ceiling, group.compute_bytes);
        device_ceiling = std::max(device_ceiling, group.device_compute_bytes);
    }
    bool started = false;
    bool computed = false;
    auto * hybrid_memory = static_cast<llama_memory_hybrid_idx *>(memory.get());
    std::shared_ptr<const llama_memory_hybrid_idx::replay_state> replay;
    try {
        synchronize();
        replay = hybrid_memory->snapshot_metadata();
        const size_t row_bytes = ggml_row_size(params.staging_type, (int64_t) hp.n_embd * hp.dsv4_hc_mult);
        if ((size_t) n_tokens > params.max_host_bytes / 2 / row_bytes) {
            throw llama_prefill_resource_error("layered prefill activation buffers exceed max_host_bytes");
        }
        const size_t total_bytes = row_bytes * n_tokens;
        std::vector<uint8_t> input(total_bytes), output(total_bytes);
        std::unique_ptr<prefill_pipeline> pipeline;
        if (pipelined) {
            pipeline = std::make_unique<prefill_pipeline>(device, hp, std::min<uint32_t>(params.n_ubatch, n_tokens),
                                                         params, params.max_host_bytes - 2*total_bytes, stats);
        }
        std::vector<std::unique_ptr<prefill_weight_prefetch>> next_experts(hp.n_layer());
        llama_batch_ext batch(this);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const auto idx = batch.add_token(0);
            const llama_pos pos = prefix + i;
            if (idx < 0 || !batch.set_token_id(idx, tokens[i]) ||
                !batch.set_token_pos(idx, &pos) || !batch.set_output(idx, i == n_tokens - 1)) {
                throw std::runtime_error("failed to construct layered prefill batch");
            }
        }
        synchronize();
        const int64_t start_us = ggml_time_us();
        sched_reserve();
        if (output_reserve(1) < 1) {
            throw llama_prefill_resource_error("failed to reserve layered prefill logits");
        }
        started = true;
        for (size_t i_group = 0; i_group < groups.size(); ++i_group) {
            const auto & group = groups[i_group];
            const uint32_t il = group.first_layer;
            const uint32_t last = il + group.n_layers;
            ggml_backend_sched_reset(sched.get());
            gf_res_prev_active = nullptr;
            hybrid_memory->restore_metadata(*replay);
            if (!balloc->init(batch, model.vocab, false)) {
                throw std::runtime_error("failed to initialize layered prefill batch");
            }
            auto mctx = memory->init_batch(*balloc, params.n_ubatch, false);
            if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
                throw std::runtime_error("failed to prepare layered prefill memory");
            }
            size_t group_bytes = 0;
            for (uint32_t lid = il; lid < last; ++lid) {
                if (model.layers[lid].ffn_gate_exps) {
                    group_bytes = prefill_add(group_bytes, prefill_experts::allocation_size(model.layers[lid], device));
                }
            }
            if (params.max_group_expert_bytes && group_bytes > params.max_group_expert_bytes) {
                throw llama_prefill_resource_error("layered prefill group exceeds max_group_expert_bytes");
            }
            stats.groups[i_group].expert_bytes = group_bytes;
            std::vector<std::unique_ptr<prefill_experts>> banks;
            std::vector<llama_layer> weights;
            const int64_t load_start = ggml_time_us();
            const uint64_t copy_before = stats.expert_copy_bytes;
            for (uint32_t lid = il; lid < last; ++lid) {
                std::unique_ptr<prefill_experts> bank;
                if (next_experts[lid]) {
                    next_experts[lid]->finish(stats);
                    bank = std::move(next_experts[lid]->experts);
                    next_experts[lid].reset();
                } else {
                    bank = std::make_unique<prefill_experts>(model.layers[lid], device, params.max_expert_bytes, stats);
                }
                weights.push_back(bank->layer);
                banks.push_back(std::move(bank));
            }
            if (probe && stats.expert_copy_bytes > copy_before) {
                probe->weight_bytes += stats.expert_copy_bytes - copy_before;
                probe->weight_ms += (ggml_time_us() - load_start) * 1e-3;
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
                const int64_t prepare_start = ggml_time_us();
                if (!mctx->apply()) {
                    throw std::runtime_error("failed to apply layered prefill memory");
                }
                const int64_t prepare_end = ggml_time_us();
                n_outputs = last == hp.n_layer() && offset + ubatch.n_tokens == (size_t) n_tokens ? 1 : 0;
                auto * res = get_gf_res_prev();
                ggml_backend_sched_reset(sched.get());
                res->reset();
                auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DECODER);
                gparams.moe_cache = nullptr;
                gparams.prefill_layer = il;
                gparams.prefill_layers = group.n_layers;
                gparams.prefill_type = params.staging_type;
                gparams.prefill_weights = weights.data();
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
                prefill_check_graph(sched.get(), gf, weights, device, *res);
                size_t compute_bytes = 0;
                for (size_t i = 0; i < sizes.size(); ++i) {
                    compute_bytes = prefill_add(compute_bytes, sizes[i]);
                    if (backend_ptrs[i] == device) {
                        stats.groups[i_group].device_compute_bytes =
                            std::max(stats.groups[i_group].device_compute_bytes, sizes[i]);
                    }
                }
                if (compute_bytes > params.max_compute_bytes) {
                    throw llama_prefill_resource_error("layered prefill workspace exceeds max_compute_bytes; reduce n_ubatch");
                }
                if (compute_ceiling && compute_bytes > compute_ceiling) {
                    throw llama_prefill_resource_error("actual prefill graph exceeds the planner's workspace bound");
                }
                stats.groups[i_group].compute_bytes = std::max(stats.groups[i_group].compute_bytes, compute_bytes);
                stats.peak_compute_bytes = std::max(stats.peak_compute_bytes, compute_bytes);
                // Splitting a measurement graph rewrites sources to scheduler-owned copies.
                ggml_backend_sched_reset(sched.get());
                res->reset();
                gf = model.build_graph(gparams);
                if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                    throw llama_prefill_resource_error("failed to allocate layered prefill graph");
                }
                for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                    backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
                }
                prefill_check_graph(sched.get(), gf, weights, device, *res);
                const int64_t inputs_start = ggml_time_us();
                res->set_inputs(&ubatch);
                const int64_t inputs_end = ggml_time_us();
                const size_t bytes = row_bytes * ubatch.n_tokens;
                if (res->t_prefill_in) {
                    if (pipeline) {
                        pipeline->wait_input(slot, bytes);
                    } else {
                        ggml_backend_tensor_set(res->t_prefill_in, input.data() + offset * row_bytes, 0, bytes);
                    }
                    stats.activation_read_bytes += bytes;
                    if (probe) {
                        probe->read_bytes += bytes;
                        probe->read_ms += (ggml_time_us() - inputs_end) * 1e-3;
                    }
                }
                if (i_batch == 0 && params.weight_prefetch && device != backend_cpu && i_group + 1 < groups.size()) {
                    const size_t pinned_bytes = pipeline ? ggml_backend_buffer_get_size(pipeline->host_buffer.get()) : 0;
                    size_t extra_bytes = 0, extra_pinned = 0;
                    for (uint32_t p = 0; p < group.n_prefetch_layers; ++p) {
                        auto limits = params;
                        limits.max_prefetch_bytes -= extra_bytes;
                        next_experts[last + p] = prefill_try_prefetch(model.layers[last + p], device, sched.get(), limits,
                            params.max_host_bytes - 2*total_bytes - pinned_bytes - extra_pinned,
                            params.max_transfer_bytes > stats.transfer_buffer_bytes + extra_pinned ?
                                params.max_transfer_bytes - stats.transfer_buffer_bytes - extra_pinned : 0,
                            probe && probe->prefetch_steps ? probe->prefetch_steps :
                                ((size_t) n_tokens + params.n_ubatch - 1) / params.n_ubatch,
                            stats, device_ceiling ? device_ceiling : compute_ceiling);
                        if (next_experts[last + p]) {
                            extra_bytes += ggml_backend_buffer_get_size(next_experts[last + p]->experts->buffer.get());
                            extra_pinned += ggml_backend_buffer_get_size(next_experts[last + p]->pinned.get());
                        } else if (model.layers[last + p].ffn_gate_exps) {
                            break;
                        }
                    }
                }
                const int64_t compute_start = ggml_time_us();
                computed = true;
                const auto status = graph_compute(gf, true);
                const size_t next = offset + ubatch.n_tokens;
                if (pipeline && il > 0 && next < (size_t) n_tokens && status == GGML_STATUS_SUCCESS) {
                    const size_t next_bytes = row_bytes * std::min<size_t>(params.n_ubatch, n_tokens - next);
                    pipeline->prefetch_input((i_batch + 1) % 2, input.data() + next * row_bytes, next_bytes);
                    ++stats.input_prefetches;
                }
                for (uint32_t p = 0; p < group.n_prefetch_layers && status == GGML_STATUS_SUCCESS; ++p) {
                    if (next_experts[last + p]) {
                        stats.weight_prefetch_submit_bytes += next_experts[last + p]->advance(next_experts[last + p]->quota);
                    }
                }
                synchronize();
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("layered prefill computation failed or was aborted");
                }
                const int64_t compute_end = ggml_time_us();
                if (probe) {
                    const auto * hybrid = static_cast<const llama_memory_hybrid_idx_context *>(mctx.get());
                    probe->points.push_back({il, ubatch.n_tokens, hybrid->get_attn()->get_n_kv(),
                        (compute_end - compute_start) * 1e-3,
                        (prepare_end - prepare_start + inputs_end - inputs_start) * 1e-3});
                }
                if (pipeline) {
                    pipeline->copy_output(slot, output.data() + offset * row_bytes, bytes);
                    ++stats.async_output_batches;
                } else {
                    ggml_backend_tensor_get(res->t_prefill_out, output.data() + offset * row_bytes, 0, bytes);
                }
                stats.activation_write_bytes += bytes;
                if (probe) {
                    probe->write_bytes += bytes;
                    probe->write_ms += (ggml_time_us() - compute_end) * 1e-3;
                }
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
        n_outputs = probe ? 0 : 1;
        std::fill(output_ids.begin(), output_ids.end(), -1);
        if (!probe) {
            output_ids[n_tokens - 1] = 0;
        }
        output_swaps.clear();
        embd_seq.clear();
        embd_batch_idxs.clear();
        if (!probe) {
            t_compute_start_us = start_us;
            n_queued_tokens = n_tokens;
        }
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
            if (prefix > 0 && !computed && replay) {
                hybrid_memory->restore_metadata(*replay);
            } else {
                memory->clear(true);
            }
            if (prefix > 0 && computed) {
                LLAMA_LOG_ERROR("%s: execution failed after state updates; the cached prefix was invalidated\n", __func__);
            }
            n_outputs = 0;
            std::fill(output_ids.begin(), output_ids.end(), -1);
        }
        if (stats_out) {
            *stats_out = {};
            stats_out->prefix_invalidated = prefix > 0 && computed;
        }
        return probe && (dynamic_cast<const llama_prefill_resource_error *>(&e) ||
                         dynamic_cast<const std::bad_alloc *>(&e)) ? -3 : -2;
    }
}

int llama_context::prefill_layered_plan(
        const llama_token * tokens, int32_t n_tokens,
        const llama_layered_prefill_params & params, llama_layered_prefill_stats * stats_out,
        ggml_backend_t device, bool capacity_only) {
    const auto & hp = model.hparams;
    const bool gpu = device != backend_cpu;
    const bool pipelined = params.pipeline && gpu;
    const int64_t planning_start = ggml_time_us();
    llama_layered_prefill_stats diagnostic;
    const bool has_prefix = memory->seq_pos_max(0) >= 0;
    const uint32_t prefix_tokens = memory->seq_pos_max(0) + 1;
    auto * hybrid_memory = static_cast<llama_memory_hybrid_idx *>(memory.get());
    std::shared_ptr<const llama_memory_hybrid_idx::replay_state> original_metadata;
    const auto clean = [&]() {
        synchronize();
        ggml_backend_sched_reset(sched.get());
        gf_res_prev_active = nullptr;
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_reserve->reset();
        if (has_prefix) {
            if (original_metadata) {
                hybrid_memory->restore_metadata(*original_metadata);
            }
        } else {
            memory->clear(true);
        }
        n_outputs = 0;
        std::fill(output_ids.begin(), output_ids.end(), -1);
    };
    const auto fresh_scheduler = [&]() {
        clean();
        sched_new(gf_res_reserve->get_max_nodes(), false);
        sched_need_reserve = false;
        std::fill(backend_buf_exp_size.begin(), backend_buf_exp_size.end(), 0);
    };
    try {
        synchronize();
        if (has_prefix) {
            original_metadata = hybrid_memory->snapshot_metadata();
        }
        auto replay_probe = has_prefix ? original_metadata : hybrid_memory->snapshot_metadata();
        const int64_t replay_start = ggml_time_us();
        hybrid_memory->restore_metadata(*replay_probe);
        const double replay_ms = (ggml_time_us() - replay_start) * 1e-3;
        replay_probe.reset();
        for (const auto & item : memory_breakdown()) {
            if (ggml_backend_buft_get_device(item.first) == ggml_backend_get_device(device) &&
                (device == backend_cpu || !ggml_backend_buft_is_host(item.first))) {
                diagnostic.resident_model_bytes = prefill_add(diagnostic.resident_model_bytes, item.second.model);
                diagnostic.resident_state_bytes = prefill_add(diagnostic.resident_state_bytes, item.second.context);
            }
        }
        const size_t row = ggml_row_size(params.staging_type, int64_t(hp.n_embd) * hp.dsv4_hc_mult);
        LLAMA_LOG_DEBUG("%s: begin planning\n", __func__);
        if ((size_t) n_tokens > SIZE_MAX / 2 / row) {
            diagnostic.budget_shortfall_bytes = SIZE_MAX - params.max_host_bytes;
            diagnostic.shortfall_is_lower_bound = true;
            throw llama_prefill_resource_error("resident host staging exceeds the addressable size");
        }
        if ((size_t) n_tokens > params.max_host_bytes / 2 / row) {
            const size_t required = 2 * row * n_tokens;
            diagnostic.budget_shortfall_bytes = required - params.max_host_bytes;
            throw llama_prefill_resource_error(format("resident host staging needs %zu bytes, limit %zu, gap %zu",
                required, params.max_host_bytes, diagnostic.budget_shortfall_bytes));
        }
        const size_t activation_bytes = 2 * row * n_tokens;
        std::vector<uint32_t> families;
        std::vector<size_t> allocation, payload;
        std::array<bool, 8> seen = {};
        uint32_t probe_layers = std::min(2u, hp.n_layer());
        for (uint32_t il = 0; il < hp.n_layer(); ++il) {
            const uint32_t family = prefill_family(model, il);
            families.push_back(family);
            if (!seen[family]) {
                seen[family] = true;
                probe_layers = std::max(probe_layers, il + 1);
            }
            size_t bytes = 0, data = 0;
            if (model.layers[il].ffn_gate_exps) {
                bytes = prefill_experts::allocation_size(model.layers[il], device);
                for (auto * t : prefill_experts::tensors(model.layers[il])) {
                    data = prefill_add(data, ggml_nbytes(t));
                }
            }
            if (bytes > params.max_expert_bytes) {
                diagnostic.budget_shortfall_bytes = bytes - params.max_expert_bytes;
                throw llama_prefill_resource_error(format("resident expert layer needs %zu bytes, per-layer limit %zu, gap %zu",
                    bytes, params.max_expert_bytes, diagnostic.budget_shortfall_bytes));
            }
            if (params.max_group_expert_bytes && bytes > params.max_group_expert_bytes) {
                diagnostic.budget_shortfall_bytes = bytes - params.max_group_expert_bytes;
                throw llama_prefill_resource_error(format("resident expert layer needs %zu bytes, group limit %zu, gap %zu",
                    bytes, params.max_group_expert_bytes, diagnostic.budget_shortfall_bytes));
            }
            allocation.push_back(bytes);
            payload.push_back(data);
        }
        if (!prefill_profile || prefill_profile->staging_type != params.staging_type ||
            prefill_profile->threads != cparams.n_threads_batch) {
            prefill_profile = std::make_shared<llama_prefill_profile>();
            prefill_profile->staging_type = params.staging_type;
            prefill_profile->threads = cparams.n_threads_batch;
        }
        if (prefill_profile->ratio_tokens != (uint32_t) n_tokens) {
            prefill_profile->compute_ratios.clear();
            prefill_profile->ratio_tokens = n_tokens;
        }
        std::vector<uint32_t> batches;
        for (uint32_t b = std::min<uint32_t>(params.n_ubatch, n_tokens); b > 0; b /= 2) {
            batches.push_back(b);
        }
        std::map<uint32_t, llama_prefill_cost> costs;
        for (uint32_t b : batches) {
            if (capacity_only) {
                continue;
            }
            const uint32_t sample_tokens = std::min<uint64_t>(n_tokens, uint64_t(b) * 4);
            const auto cached = prefill_profile->costs.find(b);
            if (cached != prefill_profile->costs.end() && cached->second.sample_tokens >= sample_tokens) {
                costs[b] = cached->second;
                diagnostic.profile_reused = true;
                continue;
            }
            if (has_prefix || params.max_calibration_ms == 0 ||
                (diagnostic.calibration_ms >= params.max_calibration_ms && !costs.empty())) {
                continue;
            }
            fresh_scheduler();
            LLAMA_LOG_DEBUG("%s: calibrating batch %u\n", __func__, b);
            auto trial = params;
            trial.auto_plan = false;
            trial.n_ubatch = b;
            trial.pipeline = false;
            trial.weight_prefetch = false;
            std::vector<llama_layered_prefill_group> sample_groups;
            for (uint32_t il = 0; il < probe_layers; ++il) {
                sample_groups.push_back({il, 1, 0});
            }
            llama_prefill_probe probe;
            const int64_t begin = ggml_time_us();
            const int status = prefill_layered_run(tokens, sample_tokens, trial, nullptr, device, sample_groups, &probe);
            diagnostic.calibration_ms += (ggml_time_us() - begin) * 1e-3;
            ++diagnostic.calibration_probes;
            clean();
            LLAMA_LOG_DEBUG("%s: calibrated batch %u status %d\n", __func__, b, status);
            if (status == -3) {
                LLAMA_LOG_INFO("%s: discard calibration batch %u: resource limit\n", __func__, b);
                continue;
            }
            if (status != 0) {
                throw std::runtime_error("joint prefill calibration failed");
            }
            auto cost = prefill_fit_cost(probe, model, b, sample_tokens);
            for (uint32_t family : families) {
                if (!cost.measured[family]) {
                    throw std::runtime_error("joint prefill calibration did not cover every layer family");
                }
            }
            prefill_profile->costs[b] = cost;
            costs[b] = cost;
        }
        if (costs.empty()) {
            diagnostic.capacity_only = true;
            LLAMA_LOG_WARN("%s: no compatible timing profile; selecting a resident capacity plan without destructive calibration\n", __func__);
        }
        fresh_scheduler();
        LLAMA_LOG_DEBUG("%s: measuring candidate groups\n", __func__);
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(device), &free_bytes, &total_bytes);
        diagnostic.device_free_bytes = free_bytes;
        const size_t device_budget = gpu ?
            (free_bytes > params.prefetch_headroom_bytes ? free_bytes - params.prefetch_headroom_bytes : 0) : SIZE_MAX;
        struct measured_group {
            bool fits = false;
            llama_layered_prefill_group group;
        };
        std::map<std::tuple<uint32_t, uint32_t, uint32_t>, measured_group> measurements;
        std::map<uint32_t, llama_ubatch> sample_batches;
        std::map<uint32_t, size_t> transfer_sizes;
        const auto transfer_device_bytes = [&](uint32_t b) {
            if (!pipelined) {
                return size_t(0);
            }
            const auto old = transfer_sizes.find(b);
            if (old != transfer_sizes.end()) {
                return old->second;
            }
            const size_t bytes = prefill_pipeline::projected_device_bytes(device, hp, b, params.staging_type);
            transfer_sizes[b] = bytes;
            return bytes;
        };
        const auto measure = [&](uint32_t b, uint32_t first, uint32_t count) -> measured_group {
            const auto key = std::make_tuple(b, first, count);
            const auto old = measurements.find(key);
            if (old != measurements.end()) {
                return old->second;
            }
            measured_group result;
            result.group = {first, count, 0};
            try {
                size_t weight_bytes = 0;
                for (uint32_t il = first; il < first + count; ++il) {
                    weight_bytes = prefill_add(weight_bytes, allocation[il]);
                }
                if ((params.max_group_expert_bytes && weight_bytes > params.max_group_expert_bytes) ||
                    weight_bytes > device_budget) {
                    throw llama_prefill_resource_error("group expert budget");
                }
                result.group.expert_bytes = weight_bytes;
                if (sample_batches.find(b) == sample_batches.end()) {
                    llama_batch_ext batch(this);
                    for (uint32_t i = 0; i < b; ++i) {
                        const auto idx = batch.add_token(0);
                        const llama_pos pos = prefix_tokens + i;
                        if (idx < 0 || !batch.set_token_id(idx, tokens[i]) ||
                            !batch.set_token_pos(idx, &pos) || !batch.set_output(idx, i + 1 == b)) {
                            throw std::runtime_error("failed to build joint prefill measurement batch");
                        }
                    }
                    llama_batch_allocr allocator(hp.n_pos_per_embd());
                    if (!allocator.init(batch, model.vocab, false)) {
                        throw std::runtime_error("failed to split joint prefill measurement batch");
                    }
                    sample_batches[b] = allocator.split_equal(b, false, 0);
                }
                std::vector<std::unique_ptr<prefill_experts>> banks;
                std::vector<llama_layer> weights;
                llama_layered_prefill_stats ignored;
                for (uint32_t il = first; il < first + count; ++il) {
                    auto bank = std::make_unique<prefill_experts>(model.layers[il], device,
                                                                params.max_expert_bytes, ignored, true, true);
                    if (allocation[il] && !bank->buffer) {
                        throw llama_prefill_resource_error("joint prefill measurement bank allocation failed");
                    }
                    weights.push_back(bank->layer);
                    banks.push_back(std::move(bank));
                }
                const uint32_t kpool = hp.indexer_kpool;
                const uint32_t stale_pools = has_prefix &&
                    hybrid_memory->mem_idx_stale_get()[0] != llama_memory_hybrid_idx::POS_CLEAN ?
                        prefix_tokens / kpool : 0;
                auto full = hybrid_memory->init_measure(prefix_tokens + n_tokens,
                    stale_pools + b / kpool + 2);
                llm_graph_result res(gf_res_reserve->get_max_nodes());
                ggml_backend_sched_reset(sched.get());
                auto gp = graph_params(&res, sample_batches[b], full.get(), LLM_GRAPH_TYPE_DECODER);
                gp.n_outputs = 1;
                gp.moe_cache = nullptr;
                gp.prefill_layer = first;
                gp.prefill_layers = count;
                gp.prefill_type = params.staging_type;
                gp.prefill_weights = weights.data();
                const int64_t start = ggml_time_us();
                auto * gf = model.build_graph(gp);
                std::vector<size_t> sizes(backend_ptrs.size());
                ggml_backend_sched_reserve_size(sched.get(), gf, sizes.data());
                prefill_check_graph(sched.get(), gf, weights, device, res);
                for (size_t i = 0; i < sizes.size(); ++i) {
                    result.group.compute_bytes = prefill_add(result.group.compute_bytes, sizes[i]);
                    if (backend_ptrs[i] == device) {
                        result.group.device_compute_bytes = sizes[i];
                    }
                }
                ggml_backend_sched_reset(sched.get());
                res.reset();
                model.build_graph(gp);
                result.group.dispatch_ms = (ggml_time_us() - start) * 1e-3;
                ggml_backend_sched_reset(sched.get());
                const size_t transfer_device = transfer_device_bytes(b);
                const size_t transfer_host = pipelined ? 4 * row * b : 0;
                const size_t transfer_all = prefill_add(transfer_device, transfer_host);
                result.fits = result.group.compute_bytes <= params.max_compute_bytes &&
                    transfer_all <= params.max_transfer_bytes &&
                    transfer_host <= params.max_host_bytes - activation_bytes &&
                    weight_bytes <= device_budget &&
                    result.group.device_compute_bytes <= device_budget - weight_bytes &&
                    transfer_device <= device_budget - weight_bytes - result.group.device_compute_bytes;
            } catch (const llama_prefill_resource_error &) {
                ggml_backend_sched_reset(sched.get());
            }
            measurements[key] = result;
            return result;
        };
        struct candidate {
            uint32_t batch;
            std::vector<llama_layered_prefill_group> groups;
            double score;
        };
        std::vector<candidate> candidates;
        const auto plan_key = [](const candidate & plan) {
            std::vector<uint32_t> key = {plan.batch};
            for (const auto & group : plan.groups) {
                key.push_back(group.n_layers);
                key.push_back(group.n_prefetch_layers);
            }
            return key;
        };
        const auto score = [&](const candidate & plan, double ratio = 1.0, std::vector<double> * group_times = nullptr) {
            ++diagnostic.planner_candidates;
            if (diagnostic.capacity_only) {
                return 0.0;
            }
            auto cost = costs.at(plan.batch);
            cost.replay_ms = replay_ms;
            for (size_t f = 0; f < cost.measured.size(); ++f) {
                cost.compute_ms[f] *= ratio;
                cost.compute_prefix_ms[f] *= ratio;
            }
            return llama_prefill_estimate(cost, plan.groups, families, payload, n_tokens, plan.batch, row,
                                           pipelined, group_times, prefix_tokens);
        };
        std::vector<uint32_t> active_batches = batches;
        if (!diagnostic.capacity_only) {
            active_batches.clear();
            for (const auto & item : costs) {
                active_batches.push_back(item.first);
            }
        }
        for (uint32_t b : active_batches) {
            for (uint32_t cap = 1; cap <= std::min(params.max_group_layers, hp.n_layer()); ++cap) {
                candidate plan = {b, {}, 0};
                uint32_t first = 0;
                while (first < hp.n_layer()) {
                    uint32_t count = std::min(cap, hp.n_layer() - first);
                    auto measured = measure(b, first, count);
                    while (!measured.fits && count > 1) {
                        measured = measure(b, first, --count);
                    }
                    if (!measured.fits) {
                        break;
                    }
                    plan.groups.push_back(measured.group);
                    first += count;
                }
                if (first != hp.n_layer()) {
                    continue;
                }
                plan.score = score(plan);
                size_t device_compute = 0;
                for (const auto & group : plan.groups) {
                    device_compute = std::max(device_compute, group.device_compute_bytes);
                }
                if (!diagnostic.capacity_only && params.weight_prefetch && gpu) {
                    for (size_t g = 0; g + 1 < plan.groups.size(); ++g) {
                        std::vector<double> times;
                        score(plan, 1.0, &times);
                        const double threshold = std::max(0.001, times[g] * 0.01);
                        size_t extra = 0, pinned = 0;
                        for (uint32_t p = 1; p <= std::min(params.max_prefetch_layers, plan.groups[g + 1].n_layers); ++p) {
                            const uint32_t il = plan.groups[g + 1].first_layer + p - 1;
                            extra = prefill_add(extra, allocation[il]);
                            pinned = prefill_add(pinned, 2 * std::min<size_t>(8 << 20, payload[il]));
                            const size_t transfer_device = transfer_device_bytes(b);
                            const size_t transfer_host = pipelined ? 4 * row * b : 0;
                            const size_t current = prefill_add(plan.groups[g].expert_bytes,
                                                              prefill_add(device_compute, transfer_device));
                            if (extra > params.max_prefetch_bytes || current > device_budget ||
                                extra > device_budget - current ||
                                pinned > params.max_transfer_bytes - transfer_device - transfer_host ||
                                pinned > params.max_host_bytes - activation_bytes - transfer_host) {
                                break;
                            }
                            auto trial = plan;
                            trial.groups[g].n_prefetch_layers = p;
                            const double estimate = score(trial);
                            if (estimate + threshold < plan.score) {
                                plan = std::move(trial);
                                plan.score = estimate;
                            }
                        }
                    }
                }
                const auto ratio = prefill_profile->compute_ratios.find(plan_key(plan));
                if (!diagnostic.capacity_only && ratio != prefill_profile->compute_ratios.end()) {
                    plan.score = score(plan, ratio->second);
                }
                candidates.push_back(std::move(plan));
            }
        }
        if (candidates.empty()) {
            if (!diagnostic.capacity_only) {
                LLAMA_LOG_WARN("%s: timed candidates do not fit; searching smaller resident-only configurations\n", __func__);
                clean();
                llama_layered_prefill_stats fallback;
                const int status = prefill_layered_plan(tokens, n_tokens, params, &fallback, device, true);
                fallback.calibration_ms += diagnostic.calibration_ms;
                fallback.calibration_probes += diagnostic.calibration_probes;
                fallback.planner_candidates += diagnostic.planner_candidates;
                fallback.planning_ms = (ggml_time_us() - planning_start) * 1e-3;
                if (stats_out) {
                    *stats_out = std::move(fallback);
                }
                return status;
            }
            for (uint32_t il = 0; il < hp.n_layer(); ++il) {
                const auto minimum = measure(1, il, 1);
                const size_t transfer_device = transfer_device_bytes(1);
                const size_t required = prefill_add(allocation[il],
                    prefill_add(minimum.group.device_compute_bytes, transfer_device));
                diagnostic.minimum_additional_bytes = std::max(diagnostic.minimum_additional_bytes, required);
                if (allocation[il] > device_budget) {
                    diagnostic.shortfall_is_lower_bound = true;
                }
                if (minimum.group.compute_bytes > params.max_compute_bytes) {
                    diagnostic.budget_shortfall_bytes = std::max(diagnostic.budget_shortfall_bytes,
                        minimum.group.compute_bytes - params.max_compute_bytes);
                }
            }
            const bool overflow = params.prefetch_headroom_bytes > SIZE_MAX - diagnostic.minimum_additional_bytes;
            const size_t required = overflow ? SIZE_MAX :
                diagnostic.minimum_additional_bytes + params.prefetch_headroom_bytes;
            diagnostic.shortfall_is_lower_bound |= overflow;
            diagnostic.capacity_shortfall_bytes = gpu && required > free_bytes ? required - free_bytes : 0;
            throw llama_prefill_resource_error(format(
                "no resident plan fits: model=%zu state=%zu free=%zu minimum_extra=%zu headroom=%zu VRAM_gap=%zu%s policy_gap=%zu; no KV eviction",
                diagnostic.resident_model_bytes, diagnostic.resident_state_bytes, free_bytes,
                diagnostic.minimum_additional_bytes, params.prefetch_headroom_bytes,
                diagnostic.capacity_shortfall_bytes, diagnostic.shortfall_is_lower_bound ? " (lower bound)" : "",
                diagnostic.budget_shortfall_bytes));
        }
        const auto order = [&](const candidate & a, const candidate & b) {
            if (diagnostic.capacity_only) {
                return a.batch != b.batch ? a.batch > b.batch : a.groups.size() < b.groups.size();
            }
            return a.score < b.score;
        };
        std::sort(candidates.begin(), candidates.end(), order);
        for (size_t i = 0; i < std::min<size_t>(2, candidates.size()); ++i) {
            auto & plan = candidates[i];
            const auto key = plan_key(plan);
            if (has_prefix || diagnostic.capacity_only || prefill_profile->compute_ratios.count(key) ||
                diagnostic.calibration_ms >= params.max_calibration_ms) {
                continue;
            }
            auto trial = params;
            trial.auto_plan = false;
            trial.n_ubatch = plan.batch;
            const uint32_t sample_tokens = std::min<uint64_t>(n_tokens, uint64_t(plan.batch) * 4);
            auto sample_groups = plan.groups;
            while (!sample_groups.empty() && sample_groups.back().first_layer >= probe_layers) {
                sample_groups.pop_back();
            }
            sample_groups.back().n_layers = std::min(sample_groups.back().n_layers,
                                                     probe_layers - sample_groups.back().first_layer);
            sample_groups.back().n_prefetch_layers = 0;
            for (size_t g = 0; g + 1 < sample_groups.size(); ++g) {
                sample_groups[g].n_prefetch_layers = std::min(sample_groups[g].n_prefetch_layers, sample_groups[g + 1].n_layers);
            }
            fresh_scheduler();
            llama_prefill_probe probe;
            probe.prefetch_steps = uint32_t(n_tokens) / plan.batch + (uint32_t(n_tokens) % plan.batch != 0);
            const int64_t begin = ggml_time_us();
            const int status = prefill_layered_run(tokens, sample_tokens, trial, nullptr, device, sample_groups, &probe);
            diagnostic.calibration_ms += (ggml_time_us() - begin) * 1e-3;
            ++diagnostic.calibration_probes;
            clean();
            if (status != 0) {
                throw std::runtime_error("joint prefill finalist validation failed");
            }
            double observed = 0, predicted = 0;
            for (const auto & group : sample_groups) {
                bool first_point = true;
                for (const auto & point : probe.points) {
                    if (point.layer != group.first_layer) {
                        continue;
                    }
                    if (first_point) {
                        first_point = false;
                        continue;
                    }
                    double work = 0;
                    const auto & cost = costs.at(plan.batch);
                    for (uint32_t il = group.first_layer; il < group.first_layer + group.n_layers; ++il) {
                        const auto f = families[il];
                        work += cost.compute_ms[f] + cost.compute_prefix_ms[f] * point.prefix;
                    }
                    predicted += work * point.tokens / plan.batch;
                    observed += point.compute_ms;
                }
            }
            if (predicted > 0 && observed > 0) {
                const double ratio = observed / predicted;
                prefill_profile->compute_ratios[key] = ratio;
                plan.score = score(plan, ratio);
            }
        }
        std::sort(candidates.begin(), candidates.end(), order);
        size_t chosen = 0;
        const auto footprint = [&](const candidate & plan) {
            size_t peak = 0;
            for (size_t g = 0; g < plan.groups.size(); ++g) {
                size_t bytes = prefill_add(plan.groups[g].expert_bytes,
                    prefill_add(plan.groups[g].device_compute_bytes, transfer_device_bytes(plan.batch)));
                if (g + 1 < plan.groups.size() && plan.groups[g].n_prefetch_layers) {
                    for (uint32_t p = 0; p < plan.groups[g].n_prefetch_layers; ++p) {
                        bytes = prefill_add(bytes, allocation[plan.groups[g + 1].first_layer + p]);
                    }
                }
                peak = std::max(peak, bytes);
            }
            return peak;
        };
        for (size_t i = 1; !diagnostic.capacity_only && i < candidates.size() &&
             candidates[i].score <= candidates[0].score * 1.01; ++i) {
            if (footprint(candidates[i]) < footprint(candidates[chosen])) {
                chosen = i;
            }
        }
        auto selected = params;
        selected.auto_plan = false;
        selected.n_ubatch = candidates[chosen].batch;
        diagnostic.estimated_ms = candidates[chosen].score;
        fresh_scheduler();
        diagnostic.planning_ms = (ggml_time_us() - planning_start) * 1e-3;
        if (diagnostic.capacity_only) {
            LLAMA_LOG_INFO("%s: resident capacity plan: batch %u, %zu groups, no latency estimate, planning %.2f ms\n",
                           __func__, selected.n_ubatch, candidates[chosen].groups.size(), diagnostic.planning_ms);
        } else {
            LLAMA_LOG_INFO("%s: selected batch %u, %zu groups, estimated %.2f ms, planning %.2f ms\n", __func__,
                           selected.n_ubatch, candidates[chosen].groups.size(), diagnostic.estimated_ms, diagnostic.planning_ms);
        }
        llama_layered_prefill_stats result;
        const int status = prefill_layered_run(tokens, n_tokens, selected, &result, device, candidates[chosen].groups);
        if (stats_out) {
            result.planner_candidates = diagnostic.planner_candidates;
            result.calibration_probes = diagnostic.calibration_probes;
            result.profile_reused = diagnostic.profile_reused;
            result.calibration_ms = diagnostic.calibration_ms;
            result.planning_ms = diagnostic.planning_ms;
            result.estimated_ms = diagnostic.estimated_ms;
            result.capacity_only = diagnostic.capacity_only;
            result.resident_model_bytes = diagnostic.resident_model_bytes;
            result.resident_state_bytes = diagnostic.resident_state_bytes;
            result.device_free_bytes = diagnostic.device_free_bytes;
            *stats_out = std::move(result);
        }
        return status;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        clean();
        diagnostic.planning_ms = (ggml_time_us() - planning_start) * 1e-3;
        if (stats_out) {
            *stats_out = diagnostic;
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
