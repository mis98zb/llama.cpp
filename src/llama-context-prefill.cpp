#include "llama-context.h"

#include "llama-batch.h"
#include "llama-model.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid.h"
#include "llama-prefill-plan.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_set>

namespace {

size_t prefill_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error("layered prefill size overflow");
    }
    return a + b;
}

struct prefill_gpu_profile {
    using control_fn = bool (*)();
    using text_fn = void (*)(const char *);
    using pop_fn = void (*)();
    control_fn stop;
    text_fn push;
    text_fn mark;
    pop_fn pop;
    bool active = false;

    explicit prefill_gpu_profile(ggml_backend_t backend) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        const auto start = reinterpret_cast<control_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_profiler_start"));
        stop = reinterpret_cast<control_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_profiler_stop"));
        push = reinterpret_cast<text_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_profiler_push"));
        pop = reinterpret_cast<pop_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_profiler_pop"));
        mark = reinterpret_cast<text_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_profiler_mark"));
        if (!start || !stop || !push || !pop || !mark) {
            throw std::runtime_error("GPU timeline requires a CUDA backend built with NVTX3 support");
        }
        if (!start()) {
            throw std::runtime_error("failed to start CUDA profiler capture");
        }
        active = true;
        push("llama.prefill");
    }

    void finish() {
        mark("llama.prefill.success");
        pop();
        active = false;
        if (!stop()) {
            throw std::runtime_error("failed to stop CUDA profiler capture");
        }
    }

    ~prefill_gpu_profile() {
        if (active) {
            mark("llama.prefill.failed");
            pop();
            if (!stop()) {
                LLAMA_LOG_ERROR("failed to stop CUDA profiler after prefill failure\n");
            }
        }
    }
};

struct prefill_profile_range {
    prefill_gpu_profile * profile;

    prefill_profile_range(prefill_gpu_profile * profile, const char * name, int layer = -1, int batch = -1) : profile(profile) {
        if (profile) {
            char label[96];
            if (batch >= 0) {
                std::snprintf(label, sizeof(label), "%s L%d MB%d", name, layer, batch);
            } else if (layer >= 0) {
                std::snprintf(label, sizeof(label), "%s L%d", name, layer);
            } else {
                std::snprintf(label, sizeof(label), "%s", name);
            }
            profile->push(label);
        }
    }

    void close() {
        if (profile) {
            profile->pop();
            profile = nullptr;
        }
    }

    ~prefill_profile_range() { close(); }
};

struct prefill_pipeline {
    struct slot {
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        ggml_backend_event_ptr input_ready;
        ggml_backend_event_ptr output_ready;
        bool output_pending = false;
        size_t input_bytes = 0;
    };

    ggml_backend_t compute;
    ggml_backend_ptr upload;
    ggml_backend_ptr download;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr device_buffer;
    std::array<slot, 2> slots;

    static size_t projected_device_bytes(ggml_backend_t backend, const llama_hparams & hp,
                                         uint32_t tokens, ggml_type type) {
        ggml_context_ptr metadata(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        if (!metadata) {
            throw llama_prefill_resource_error("failed to measure activation staging metadata");
        }
        std::array<ggml_tensor *, 4> tensors;
        for (auto & tensor : tensors) {
            tensor = ggml_new_tensor_3d(metadata.get(), type, hp.n_embd, std::max(1u, hp.dsv4_hc_mult), tokens);
        }
        return ggml_backend_buft_get_alloc_size_n(ggml_backend_get_default_buffer_type(backend), tensors.data(), 4);
    }

    prefill_pipeline(ggml_backend_t backend, const llama_hparams & hp, uint32_t n_tokens,
                     const llama_layered_prefill_params & params,
                     llama_layered_prefill_stats & stats) : compute(backend) {
        const auto dev = ggml_backend_get_device(backend);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        const auto host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!props.caps.async || !props.caps.events || !props.caps.host_buffer || !host_buft) {
            throw std::runtime_error("layered prefill pipeline requires asynchronous copies, events and pinned host buffers");
        }
        const size_t row = ggml_row_size(params.staging_type, int64_t(hp.n_embd) * std::max(1u, hp.dsv4_hc_mult));
        if (n_tokens > params.max_transfer_bytes / 4 / row) {
            throw llama_prefill_resource_error("layered prefill pipeline exceeds its transfer buffer budget");
        }
        ctx.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw std::runtime_error("failed to allocate layered prefill pipeline metadata");
        }
        ggml_tensor * tensors[4];
        for (size_t i = 0; i < slots.size(); ++i) {
            auto & s = slots[i];
            s.input = tensors[2*i] = ggml_new_tensor_3d(ctx.get(), params.staging_type,
                                                       hp.n_embd, std::max(1u, hp.dsv4_hc_mult), n_tokens);
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
        if (device_bytes > params.max_transfer_bytes) {
            throw llama_prefill_resource_error("layered prefill pipeline exceeds max_transfer_bytes");
        }
        device_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!device_buffer) {
            throw llama_prefill_resource_error("failed to allocate layered prefill device slots");
        }
        const size_t all_bytes = ggml_backend_buffer_get_size(device_buffer.get());
        if (all_bytes > params.max_transfer_bytes) {
            throw llama_prefill_resource_error("allocated layered prefill pipeline exceeds its budget");
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
        if (s.output_pending) {
            ggml_backend_event_synchronize(s.output_ready.get());
            s.output_pending = false;
        }
    }

    void prefetch_input(size_t i, const uint8_t * source, size_t bytes) {
        auto & s = slots[i];
        if (bytes == 0 || bytes > ggml_nbytes(s.input)) {
            throw std::runtime_error("layered prefill input exceeds its pipeline slot");
        }
        if (s.input_bytes) {
            ggml_backend_event_synchronize(s.input_ready.get());
        }
        ggml_backend_tensor_set_async(upload.get(), s.input, source, 0, bytes);
        ggml_backend_event_record(s.input_ready.get(), upload.get());
        s.input_bytes = bytes;
    }

    void wait_input(size_t i, size_t bytes) {
        auto & s = slots[i];
        if (s.input_bytes != bytes) {
            throw std::runtime_error("layered prefill pipeline input does not match the prepared microbatch");
        }
        ggml_backend_event_wait(compute, s.input_ready.get());
    }

    void copy_output(size_t i, uint8_t * destination, size_t bytes) {
        auto & s = slots[i];
        if (bytes == 0 || bytes > ggml_nbytes(s.output) || s.output_pending) {
            throw std::runtime_error("layered prefill output exceeds its pipeline slot or overwrites a pending copy");
        }
        ggml_backend_tensor_get_async(download.get(), s.output, destination, 0, bytes);
        ggml_backend_event_record(s.output_ready.get(), download.get());
        s.output_pending = true;
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
    bool merged_gate_up = false;

    static bool can_merge(const llama_layer & source) {
        const auto * gate = source.ffn_gate_exps;
        const auto * up = source.ffn_up_exps;
        return !source.ffn_gate_up_exps && gate && up && gate->type == up->type &&
            ggml_are_same_shape(gate, up) && ggml_is_contiguous(gate) && ggml_is_contiguous(up) &&
            gate->ne[1] > 0 && gate->ne[1] <= INT64_MAX / 2 && gate->ne[2] > 0 && gate->ne[3] == 1 &&
            !source.ffn_gate_exps_s && !source.ffn_up_exps_s &&
            !source.ffn_gate_exps_b && !source.ffn_up_exps_b;
    }

    static std::vector<ggml_tensor *> tensors(const llama_layer & layer) {
        if (layer.ffn_gate_up_exps) {
            return { layer.ffn_gate_up_exps, layer.ffn_down_exps };
        }
        return { layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps };
    }

    static size_t allocation_size(const llama_layer & layer, ggml_backend_t backend, uint32_t slots = 0) {
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t bytes = 0;
        for (const auto * w : tensors(layer)) {
            if (!w || !w->buffer || !w->data || !ggml_is_contiguous(w) || !ggml_backend_buffer_is_host(w->buffer)) {
                throw std::runtime_error("layered prefill requires contiguous host-resident routed experts");
            }
            ggml_tensor compact = *w;
            if (slots) {
                compact.ne[2] = slots;
                compact.nb[3] = slots * compact.nb[2];
            }
            const size_t n = ggml_backend_buft_get_alloc_size(buft, &compact);
            bytes = prefill_add(bytes, prefill_add(n, alignment - 1) / alignment * alignment);
        }
        if (slots) {
            ggml_context_ptr metadata(ggml_init({ggml_tensor_overhead(), nullptr, true}));
            if (!metadata) {
                throw llama_prefill_resource_error("failed to measure expert slot metadata");
            }
            auto * map = ggml_new_tensor_2d(metadata.get(), GGML_TYPE_I32, 1, layer.ffn_down_exps->ne[2]);
            const size_t n = ggml_backend_buft_get_alloc_size(buft, map);
            bytes = prefill_add(bytes, prefill_add(n, alignment - 1) / alignment * alignment);
        }
        return bytes;
    }

    bool compatible(const llama_layer & source, uint32_t slots = 0) const {
        const auto current = tensors(layer);
        auto weights = tensors(source);
        ggml_tensor combined = {};
        if (merged_gate_up) {
            if (slots || !can_merge(source)) {
                return false;
            }
            combined = *source.ffn_gate_exps;
            combined.ne[1] *= 2;
            weights = {&combined, source.ffn_down_exps};
        }
        if (current.size() != weights.size()) {
            return false;
        }
        for (size_t i = 0; i < weights.size(); ++i) {
            if (!current[i] || !weights[i] || current[i]->type != weights[i]->type ||
                current[i]->ne[0] != weights[i]->ne[0] || current[i]->ne[1] != weights[i]->ne[1] ||
                current[i]->ne[2] != (slots ? slots : weights[i]->ne[2])) {
                return false;
            }
        }
        return true;
    }

    void rebind(const llama_layer & source, uint32_t slots, llama_layered_prefill_stats & stats) {
        ggml_backend_synchronize(backend);
        const auto copies = tensors(layer);
        const auto weights = tensors(source);
        layer = source;
        if (merged_gate_up) {
            layer.ffn_gate_exps = layer.ffn_up_exps = nullptr;
            layer.ffn_gate_up_exps = copies[0];
            layer.ffn_down_exps = copies[1];
            ggml_set_name(layer.ffn_down_exps, source.ffn_down_exps->name);
            ++stats.expert_reuses;
            return;
        }
        const std::vector<ggml_tensor **> fields = source.ffn_gate_up_exps ?
            std::vector<ggml_tensor **>{&layer.ffn_gate_up_exps, &layer.ffn_down_exps} :
            std::vector<ggml_tensor **>{&layer.ffn_gate_exps, &layer.ffn_up_exps, &layer.ffn_down_exps};
        for (size_t i = 0; i < copies.size(); ++i) {
            *fields[i] = copies[i];
            ggml_set_name(copies[i], weights[i]->name);
        }
        if (slots) {
            ggml_backend_buffer_clear(buffer.get(), 0);
        }
        ++stats.expert_reuses;
    }

    prefill_experts(const llama_layer & source, ggml_backend_t backend, size_t budget,
                    llama_layered_prefill_stats & stats, bool deferred = false, bool optional = false,
                    uint32_t slots = 0, bool merge_gate_up = false) :
        backend(backend), layer(source) {
        if (!source.ffn_gate_exps && !source.ffn_gate_up_exps) {
            return;
        }
        merged_gate_up = merge_gate_up && !source.ffn_gate_up_exps;
        if (merged_gate_up && (!can_merge(source) || slots || !deferred)) {
            throw std::runtime_error("merged experts require compatible gate/up weights and deferred complete-bank uploads");
        }
        const auto weights = tensors(source);
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t bytes = allocation_size(source, backend, slots);
        if (bytes > budget) {
            throw llama_prefill_resource_error("layered prefill expert buffer exceeds max_expert_bytes");
        }
        ctx.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw std::runtime_error("failed to allocate layered prefill expert metadata");
        }
        std::vector<ggml_tensor **> copies = source.ffn_gate_up_exps || merged_gate_up ?
            std::vector<ggml_tensor **>{&layer.ffn_gate_up_exps, &layer.ffn_down_exps} :
            std::vector<ggml_tensor **>{&layer.ffn_gate_exps, &layer.ffn_up_exps, &layer.ffn_down_exps};
        if (!merged_gate_up) {
            for (size_t i = 0; i < weights.size(); ++i) {
                *copies[i] = ggml_new_tensor_3d(ctx.get(), weights[i]->type,
                    weights[i]->ne[0], weights[i]->ne[1], slots ? slots : weights[i]->ne[2]);
                ggml_set_name(*copies[i], weights[i]->name);
            }
        }
        if (merged_gate_up) {
            layer.ffn_gate_up_exps = ggml_new_tensor_3d(ctx.get(), weights[0]->type,
                weights[0]->ne[0], 2 * weights[0]->ne[1], weights[0]->ne[2]);
            ggml_set_name(layer.ffn_gate_up_exps, "prefill_gate_up_exps");
            layer.ffn_down_exps = ggml_dup_tensor(ctx.get(), source.ffn_down_exps);
            ggml_set_name(layer.ffn_down_exps, source.ffn_down_exps->name);
            layer.ffn_gate_exps = layer.ffn_up_exps = nullptr;
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
        ++stats.expert_allocations;
        stats.peak_expert_bytes = std::max(stats.peak_expert_bytes, ggml_backend_buffer_get_size(buffer.get()));
        if (!deferred) {
            for (size_t i = 0; i < weights.size(); ++i) {
                ggml_backend_tensor_set(*copies[i], weights[i]->data, 0, ggml_nbytes(weights[i]));
                stats.expert_copy_bytes += ggml_nbytes(weights[i]);
            }
            ++stats.expert_layer_loads;
        }
    }

    ~prefill_experts() {
        ggml_backend_synchronize(backend);
    }

    void copy_source(const std::vector<ggml_tensor *> & source, size_t part, size_t offset,
                     size_t bytes, uint8_t * destination) const {
        if (!merged_gate_up || part != 0) {
            const auto * tensor = source[merged_gate_up ? 2 : part];
            std::memcpy(destination, static_cast<const uint8_t *>(tensor->data) + offset, bytes);
            return;
        }
        const size_t half = source[0]->nb[2];
        while (bytes) {
            const size_t index = offset / half;
            const size_t within = offset % half;
            const size_t n = std::min(bytes, half - within);
            const auto * data = static_cast<const uint8_t *>(source[index % 2]->data);
            std::memcpy(destination, data + (index / 2) * half + within, n);
            destination += n;
            offset += n;
            bytes -= n;
        }
    }
};

struct prefill_staging_pool {
    std::array<ggml_backend_buffer_ptr, 4> spare;
    size_t bytes = 0;
    size_t limit;
    llama_layered_prefill_stats & stats;

    prefill_staging_pool(size_t limit, llama_layered_prefill_stats & stats) : limit(limit), stats(stats) {}

    ggml_backend_buffer_ptr take(ggml_backend_buffer_type_t buft, size_t size, size_t available) {
        for (auto & buffer : spare) {
            if (buffer && ggml_backend_buffer_get_type(buffer.get()) == buft &&
                ggml_backend_buffer_get_size(buffer.get()) == size && size <= available) {
                ++stats.weight_staging_reuses;
                return std::move(buffer);
            }
        }
        for (auto & buffer : spare) {
            if (size <= limit - bytes) {
                break;
            }
            if (buffer) {
                bytes -= ggml_backend_buffer_get_size(buffer.get());
                buffer.reset();
            }
        }
        if (size > available || size > limit - bytes) {
            throw llama_prefill_resource_error("retained weight staging exceeds host or transfer budget");
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, size));
        if (!buffer || ggml_backend_buffer_get_type(buffer.get()) != buft ||
            ggml_backend_buffer_get_size(buffer.get()) > available ||
            ggml_backend_buffer_get_size(buffer.get()) > limit - bytes) {
            throw llama_prefill_resource_error("failed to allocate bounded retained weight staging");
        }
        bytes += ggml_backend_buffer_get_size(buffer.get());
        stats.peak_weight_staging_bytes = std::max(stats.peak_weight_staging_bytes, bytes);
        ++stats.weight_staging_allocations;
        return buffer;
    }

    void put(ggml_backend_buffer_ptr buffer) {
        for (auto & cached : spare) {
            if (!cached) {
                cached = std::move(buffer);
                return;
            }
        }
        bytes -= ggml_backend_buffer_get_size(buffer.get());
    }
};

struct prefill_staging_buffer {
    prefill_staging_pool * pool = nullptr;
    ggml_backend_buffer_ptr buffer;

    ~prefill_staging_buffer() { reset(); }

    void allocate(prefill_staging_pool * cache, ggml_backend_buffer_type_t buft, size_t bytes,
                  size_t available, llama_layered_prefill_stats & stats) {
        pool = cache;
        if (pool) {
            buffer = pool->take(buft, bytes, available);
        } else {
            buffer.reset(ggml_backend_buft_alloc_buffer(buft, bytes));
            if (buffer) {
                ++stats.weight_staging_allocations;
            }
        }
    }

    ggml_backend_buffer_t get() const { return buffer.get(); }
    explicit operator bool() const { return bool(buffer); }

    void reset() {
        if (pool && buffer) {
            pool->put(std::move(buffer));
        } else {
            buffer.reset();
        }
    }
};

struct prefill_weight_worker {
    std::mutex mutex;
    std::condition_variable cv;
    std::packaged_task<void()> next;
    bool stopping = false;
    std::thread thread;

    prefill_weight_worker() : thread([this]() {
        for (;;) {
            std::packaged_task<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() { return stopping || next.valid(); });
                if (!next.valid()) {
                    return;
                }
                job = std::move(next);
            }
            cv.notify_all();
            job();
        }
    }) {}

    ~prefill_weight_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv.notify_all();
        thread.join();
    }

    std::future<void> submit(std::function<void()> function) {
        std::packaged_task<void()> job(std::move(function));
        auto done = job.get_future();
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&]() { return !next.valid(); });
            next = std::move(job);
        }
        cv.notify_all();
        return done;
    }
};

struct prefill_static_weights {
    using member = ggml_tensor * llama_layer::*;

    static const std::vector<member> & fields() {
        static const std::vector<member> result = {
            &llama_layer::attn_norm, &llama_layer::attn_post_norm,
            &llama_layer::attn_q_norm, &llama_layer::attn_k_norm,
            &llama_layer::wq, &llama_layer::wk, &llama_layer::wv, &llama_layer::wo,
            &llama_layer::wqkv, &llama_layer::wqkv_gate,
            &llama_layer::wq_b, &llama_layer::wk_b, &llama_layer::wv_b, &llama_layer::wqkv_b,
            &llama_layer::wq_s, &llama_layer::wk_s, &llama_layer::wv_s, &llama_layer::wo_s,
            &llama_layer::wqkv_s, &llama_layer::wqkv_gate_s,
            &llama_layer::ssm_conv1d, &llama_layer::ssm_dt, &llama_layer::ssm_a,
            &llama_layer::ssm_beta, &llama_layer::ssm_alpha, &llama_layer::ssm_norm, &llama_layer::ssm_out,
            &llama_layer::ssm_beta_s, &llama_layer::ssm_alpha_s, &llama_layer::ssm_out_s,
            &llama_layer::ffn_gate_inp, &llama_layer::ffn_gate_inp_shexp,
            &llama_layer::ffn_up_shexp, &llama_layer::ffn_gate_shexp, &llama_layer::ffn_down_shexp,
            &llama_layer::ffn_up_shexp_s, &llama_layer::ffn_gate_shexp_s, &llama_layer::ffn_down_shexp_s,
            &llama_layer::ffn_up_exps_s, &llama_layer::ffn_gate_exps_s, &llama_layer::ffn_down_exps_s,
        };
        return result;
    }

    static size_t allocation_size(const llama_layer & layer, ggml_backend_t backend) {
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t size = 0;
        for (auto field : fields()) {
            const auto * tensor = layer.*field;
            if (!tensor) {
                continue;
            }
            if (!tensor->buffer || !tensor->data || !ggml_backend_buffer_is_host(tensor->buffer) ||
                !ggml_is_contiguous(tensor)) {
                throw std::runtime_error("streamed layer weights must be contiguous and host-resident");
            }
            const size_t n = ggml_backend_buft_get_alloc_size(buft, tensor);
            size = prefill_add(size, prefill_add(n, alignment - 1) / alignment * alignment);
        }
        return size;
    }

    ggml_backend_t compute;
    llama_layer layer;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    prefill_staging_buffer packed;
    ggml_backend_ptr upload;
    ggml_backend_event_ptr ready;
    ggml_tensor * blob = nullptr;
    bool submitted = false;

    prefill_static_weights(const llama_layer & source, ggml_backend_t backend, size_t budget,
                           size_t host_available, size_t transfer_available, bool asynchronous,
                           llama_layered_prefill_stats & stats, bool deferred = false,
                           prefill_staging_pool * pool = nullptr) : compute(backend), layer(source) {
        const size_t bytes = allocation_size(source, backend);
        if (!bytes || bytes > budget || bytes > host_available || bytes > transfer_available) {
            throw llama_prefill_resource_error("packed static layer weights exceed their device, host or transfer budget");
        }
        ctx.reset(ggml_init({(fields().size() + 1) * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw llama_prefill_resource_error("failed to allocate static layer weight metadata");
        }
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        buffer.reset(ggml_backend_buft_alloc_buffer(buft, bytes));
        const auto dev = ggml_backend_get_device(backend);
        const bool gpu = ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
        const auto host_buft = gpu ? ggml_backend_dev_host_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        if (!buffer || !host_buft) {
            throw llama_prefill_resource_error("failed to allocate static device bank or obtain pinned storage");
        }
        packed.allocate(pool, host_buft, bytes, std::min(host_available, transfer_available), stats);
        if (!packed || ggml_backend_buffer_get_type(packed.get()) != host_buft ||
            ggml_backend_buffer_get_size(buffer.get()) > budget ||
            ggml_backend_buffer_get_size(packed.get()) > host_available ||
            ggml_backend_buffer_get_size(packed.get()) > transfer_available) {
            throw llama_prefill_resource_error("failed to allocate bounded packed static layer weights");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * host = static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed.get()));
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get()));
        std::memset(host, 0, bytes);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t offset = 0;
        for (auto field : fields()) {
            const auto * tensor = source.*field;
            if (!tensor) {
                continue;
            }
            auto * copy = ggml_dup_tensor(ctx.get(), tensor);
            ggml_set_name(copy, tensor->name);
            ggml_backend_tensor_alloc(buffer.get(), copy, base + offset);
            std::memcpy(host + offset, tensor->data, ggml_nbytes(tensor));
            layer.*field = copy;
            const size_t n = ggml_backend_buft_get_alloc_size(buft, tensor);
            offset += (n + alignment - 1) / alignment * alignment;
        }
        blob = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, bytes);
        ggml_backend_tensor_alloc(buffer.get(), blob, base);
        if (asynchronous) {
            upload.reset(ggml_backend_dev_init(dev, nullptr));
            ready.reset(ggml_backend_event_new(dev));
            if (!upload || !ready) {
                throw llama_prefill_resource_error("failed to create static weight prefetch stream/event");
            }
            if (!deferred) {
                enqueue(stats);
            }
        } else {
            ggml_backend_tensor_set(blob, host, 0, bytes);
            submitted = true;
            packed.reset();
        }
        stats.static_copy_bytes += bytes;
        ++stats.static_layer_loads;
    }

    ~prefill_static_weights() {
        ggml_backend_synchronize(compute);
        if (upload) {
            ggml_backend_synchronize(upload.get());
        }
    }

    size_t size() const { return ggml_backend_buffer_get_size(buffer.get()); }

    void enqueue(llama_layered_prefill_stats & stats) {
        if (submitted) {
            return;
        }
        ggml_backend_tensor_set_async(upload.get(), blob, ggml_backend_buffer_get_base(packed.get()), 0, ggml_nbytes(blob));
        ggml_backend_event_record(ready.get(), upload.get());
        submitted = true;
        ++stats.static_prefetches;
    }

    void finish() {
        if (!submitted) {
            throw std::runtime_error("next-layer static weights were prepared but not submitted");
        }
        if (ready) {
            ggml_backend_event_synchronize(ready.get());
            packed.reset();
        }
    }

    void apply(llama_layer & target) {
        finish();
        for (auto field : fields()) {
            target.*field = layer.*field;
        }
    }
};

struct prefill_demand {
    prefill_experts & bank;
    const llama_layer & source;
    llama_layered_prefill_stats & stats;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * map;
    ggml_tensor * ids = nullptr;
    std::vector<int32_t> slots;
    std::vector<int32_t> mapping;
    std::string error;
    std::string route_name;
    std::string shared_name;
    size_t cursor = 0;
    ggml_backend_ptr upload;
    ggml_backend_buffer_ptr pinned;
    ggml_backend_event_ptr ready;
    bool pending = false;
    bool pending_weights = false;

    prefill_demand(prefill_experts & bank, const llama_layer & source,
                   uint32_t count, uint32_t il, const llama_layered_prefill_params & params,
                   bool asynchronous, size_t host_available, llama_layered_prefill_stats & stats) :
        bank(bank), source(source), stats(stats), slots(count, -1),
        mapping(source.ffn_down_exps->ne[2], -1),
        route_name(format("prefill_expert_route-%u", il)), shared_name(format("prefill_expert_ready-%u", il)) {
        ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw llama_prefill_resource_error("failed to allocate expert slot metadata");
        }
        map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, mapping.size());
        ggml_set_name(map, "prefill_expert_map");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), bank.backend));
        if (!buffer) {
            throw llama_prefill_resource_error("failed to allocate expert slot map");
        }
        const size_t bytes = prefill_add(ggml_backend_buffer_get_size(bank.buffer.get()),
                                         ggml_backend_buffer_get_size(buffer.get()));
        if (bytes > params.max_expert_bytes) {
            throw llama_prefill_resource_error("expert bank and slot map exceed max_expert_bytes");
        }
        stats.peak_expert_bytes = std::max(stats.peak_expert_bytes, bytes);
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(map, mapping.data(), 0, ggml_nbytes(map));
        if (asynchronous && source.ffn_up_shexp) {
            const auto dev = ggml_backend_get_device(bank.backend);
            const auto buft = ggml_backend_dev_host_buffer_type(dev);
            size_t staging = ggml_nbytes(map);
            for (auto * w : prefill_experts::tensors(source)) {
                staging = prefill_add(staging, count * w->nb[2]);
            }
            if (!buft || staging > host_available || stats.transfer_buffer_bytes > params.max_transfer_bytes ||
                staging > params.max_transfer_bytes - stats.transfer_buffer_bytes) {
                throw llama_prefill_resource_error("asynchronous expert staging exceeds host or transfer budget");
            }
            pinned.reset(ggml_backend_buft_alloc_buffer(buft, staging));
            if (!pinned || ggml_backend_buffer_get_type(pinned.get()) != buft ||
                ggml_backend_buffer_get_size(pinned.get()) > host_available ||
                ggml_backend_buffer_get_size(pinned.get()) > params.max_transfer_bytes - stats.transfer_buffer_bytes) {
                throw llama_prefill_resource_error("failed to allocate bounded pinned expert staging");
            }
            upload.reset(ggml_backend_dev_init(dev, nullptr));
            ready.reset(ggml_backend_event_new(dev));
            if (!upload || !ready) {
                throw llama_prefill_resource_error("failed to create asynchronous expert upload stream and event");
            }
            stats.transfer_buffer_bytes += ggml_backend_buffer_get_size(pinned.get());
            stats.demand_pinned_bytes = std::max(stats.demand_pinned_bytes, ggml_backend_buffer_get_size(pinned.get()));
        }
        ++stats.expert_layer_loads;
    }

    ~prefill_demand() {
        ggml_backend_synchronize(bank.backend);
        if (upload) {
            ggml_backend_synchronize(upload.get());
            stats.transfer_buffer_bytes -= ggml_backend_buffer_get_size(pinned.get());
        }
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        auto & self = *static_cast<prefill_demand *>(data);
        const bool route = tensor->name == self.route_name;
        const bool shared = tensor->name == self.shared_name;
        const bool selected = route || shared;
        if (ask) {
            return selected;
        }
        if (!selected) {
            return true;
        }
        try {
            if (route) {
                if (!self.ids) {
                    throw std::runtime_error("missing routed expert indices");
                }
                self.load(self.ids);
            } else if (self.pending) {
                ggml_backend_event_synchronize(self.ready.get());
                self.pending = false;
                if (self.pending_weights) {
                    ++self.stats.demand_overlap_batches;
                }
            }
            return true;
        } catch (const std::exception & e) {
            self.error = e.what();
            LLAMA_LOG_ERROR("layered prefill route upload: %s\n", e.what());
            return false;
        }
    }

    void load(ggml_tensor * ids) {
        if (pending) {
            throw std::runtime_error("previous expert upload did not reach the shared-compute barrier");
        }
        if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            ids->ne[2] != 1 || ids->ne[3] != 1) {
            throw std::runtime_error("expert routing requires a two-dimensional I32 tensor");
        }
        std::vector<int32_t> selected(ggml_nbytes(ids) / sizeof(int32_t));
        ggml_backend_tensor_get(ids, selected.data(), 0, ggml_nbytes(ids));
        std::vector<bool> used(mapping.size(), false);
        size_t count = 0;
        for (int64_t row = 0; row < ids->ne[1]; ++row) {
            for (int64_t col = 0; col < ids->ne[0]; ++col) {
                const auto id = selected[row * ids->nb[1] / sizeof(int32_t) + col];
                if (id < 0 || size_t(id) >= mapping.size()) {
                    throw std::runtime_error("expert route is out of range");
                }
                if (!used[id]) {
                    used[id] = true;
                    ++count;
                }
            }
        }
        if (count > slots.size()) {
            throw llama_prefill_resource_error("routed expert union exceeds expert_slots; reduce n_ubatch or increase slots");
        }
        const auto weights = prefill_experts::tensors(source);
        const auto dest = prefill_experts::tensors(bank.layer);
        const auto stream = upload ? upload.get() : bank.backend;
        auto * staging = pinned ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get())) : nullptr;
        size_t staged = 0;
        const auto before = stats.expert_copy_bytes;
        for (size_t id = 0; id < used.size(); ++id) {
            if (!used[id]) {
                continue;
            }
            if (mapping[id] >= 0) {
                ++stats.expert_cache_hits;
                continue;
            }
            while (slots[cursor] >= 0 && used[slots[cursor]]) {
                cursor = (cursor + 1) % slots.size();
            }
            if (slots[cursor] >= 0) {
                mapping[slots[cursor]] = -1;
            }
            slots[cursor] = id;
            mapping[id] = cursor;
            for (size_t i = 0; i < weights.size(); ++i) {
                const size_t bytes = weights[i]->nb[2];
                const auto * input = static_cast<const uint8_t *>(weights[i]->data) + id * bytes;
                if (staging) {
                    std::memcpy(staging + staged, input, bytes);
                    input = staging + staged;
                    staged += bytes;
                }
                ggml_backend_tensor_set_async(stream, dest[i], input, cursor * bytes, bytes);
                stats.expert_copy_bytes += bytes;
            }
            ++stats.expert_cache_misses;
            cursor = (cursor + 1) % slots.size();
        }
        if (staging) {
            std::memcpy(staging + staged, mapping.data(), ggml_nbytes(map));
            ggml_backend_tensor_set_async(stream, map, staging + staged, 0, ggml_nbytes(map));
            ggml_backend_event_record(ready.get(), stream);
            pending = true;
            pending_weights = stats.expert_copy_bytes > before;
            stats.demand_async_bytes += stats.expert_copy_bytes - before;
        } else {
            ggml_backend_tensor_set_async(stream, map, mapping.data(), 0, ggml_nbytes(map));
            ggml_backend_synchronize(stream);
        }
    }
};

// Tensor identities stay unchanged while their storage is bound to the current device bank.
struct prefill_layer_storage {
    struct binding {
        ggml_tensor * tensor;
        ggml_backend_buffer_t buffer;
        void * data;
        ggml_tensor * copy;
    };

    ggml_backend_t backend;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::vector<binding> bindings;

    static size_t allocation_size(ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors) {
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t bytes = 0;
        std::unordered_set<ggml_tensor *> seen;
        for (auto * tensor : tensors) {
            if (tensor && !tensor->view_src && seen.insert(tensor).second) {
                const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
                bytes = prefill_add(bytes, prefill_add(size, alignment - 1) / alignment * alignment);
            }
        }
        return bytes;
    }

    prefill_layer_storage(ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors,
                          size_t budget, size_t headroom, bool weights, uint64_t & uploaded,
                          bool measure = false) : backend(backend) {
        if (tensors.empty()) {
            return;
        }
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t bytes = allocation_size(backend, tensors);
        std::unordered_set<ggml_tensor *> seen;
        ctx.reset(ggml_init({tensors.size() * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw llama_prefill_resource_error("failed to allocate full-layer tensor metadata");
        }
        for (auto * tensor : tensors) {
            if (!tensor || !seen.insert(tensor).second) {
                continue;
            }
            if (!tensor->buffer || !tensor->data || !ggml_backend_buffer_is_host(tensor->buffer)) {
                throw std::runtime_error("full-layer prefill requires CPU-owned weights and state");
            }
            ggml_tensor * copy = nullptr;
            if (!tensor->view_src) {
                if (!ggml_is_contiguous(tensor)) {
                    throw std::runtime_error("full-layer prefill requires contiguous tensor storage");
                }
                copy = ggml_dup_tensor(ctx.get(), tensor);
                ggml_set_name(copy, tensor->name);
            } else if (!seen.count(tensor->view_src)) {
                throw std::runtime_error("full-layer storage view must follow its storage tensor");
            }
            bindings.push_back({tensor, tensor->buffer, tensor->data, copy});
        }
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_bytes, &total_bytes);
        if (bytes > budget || headroom > free_bytes || bytes > free_bytes - headroom) {
            throw llama_prefill_resource_error("full-layer bank exceeds its device budget");
        }
        // Dry graphs need device placement, not storage. Their host addresses are never read by a backend.
        buffer.reset(measure ? ggml_backend_buft_alloc_buffer(buft, 0) :
            ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buffer || ggml_backend_buffer_get_size(buffer.get()) > budget) {
            throw llama_prefill_resource_error("failed to allocate full-layer bank");
        }
        if (weights) {
            ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        for (auto & item : bindings) {
            if (item.copy && !measure) {
                ggml_backend_tensor_set(item.copy, item.data, 0, ggml_nbytes(item.tensor));
                uploaded += ggml_nbytes(item.tensor);
            }
        }
        for (auto & item : bindings) {
            item.tensor->buffer = buffer.get();
            item.tensor->data = measure ? item.data : item.copy ? item.copy->data :
                static_cast<uint8_t *>(item.tensor->view_src->data) + item.tensor->view_offs;
        }
    }

    void download(uint64_t & bytes) {
        ggml_backend_synchronize(backend);
        for (const auto & item : bindings) {
            if (item.copy) {
                ggml_backend_tensor_get(item.copy, item.data, 0, ggml_nbytes(item.tensor));
                bytes += ggml_nbytes(item.tensor);
            }
        }
    }

    ~prefill_layer_storage() {
        ggml_backend_synchronize(backend);
        for (auto & item : bindings) {
            item.tensor->buffer = item.buffer;
            item.tensor->data = item.data;
        }
    }
};

std::vector<ggml_tensor *> prefill_layer_weights(const llama_model & model, uint32_t first, uint32_t count) {
    std::vector<ggml_tensor *> result;
    for (const auto & entry : model.tensors_by_name) {
        for (uint32_t il = first; il < first + count; ++il) {
            const auto prefix = "blk." + std::to_string(il) + ".";
            if (entry.first.compare(0, prefix.size(), prefix) == 0) {
                result.push_back(entry.second);
                break;
            }
        }
    }
    if (first == 0) {
        result.push_back(model.tok_embd);
    }
    if (first + count == model.hparams.n_layer()) {
        result.push_back(model.output_norm);
        result.push_back(model.output);
    }
    return result;
}

std::vector<ggml_tensor *> prefill_layer_state(llama_memory_hybrid_idx & memory, uint32_t first, uint32_t count) {
    auto result = memory.get_mem_attn()->get_layer_storage(first, count);
    auto indexer = memory.get_mem_idx()->get_layer_storage(first, count);
    result.insert(result.end(), indexer.begin(), indexer.end());
    auto * recurrent = memory.get_mem_recr();
    for (const auto * storage : {&recurrent->r_l, &recurrent->s_l, &recurrent->p_l}) {
        for (uint32_t il = first; il < first + count && il < storage->size(); ++il) {
            if ((*storage)[il]) {
                result.push_back((*storage)[il]);
            }
        }
    }
    return result;
}

void prefill_place_full_graph(ggml_backend_sched_t sched, ggml_cgraph * gf, ggml_backend_t device) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_NONE) {
            continue;
        }
        if (!ggml_backend_supports_op(device, node)) {
            throw std::runtime_error(format("full-layer backend does not support %s (%s), src0=%s %lld/%lld/%lld/%lld src1=%s",
                node->name, ggml_op_name(node->op), ggml_type_name(node->src[0]->type),
                (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1],
                (long long) node->src[0]->ne[2], (long long) node->src[0]->ne[3],
                node->src[1] ? ggml_type_name(node->src[1]->type) : "none"));
        }
        ggml_backend_sched_set_tensor_backend(sched, node, device);
    }
}

struct prefill_weight_prefetch {
    static bool can_upload_direct(const llama_layer & layer, ggml_backend_t backend, bool merge_gate_up) {
        const auto dev = ggml_backend_get_device(backend);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0 ||
            (merge_gate_up && !layer.ffn_gate_up_exps)) {
            return false;
        }
        const auto buft = ggml_backend_dev_host_buffer_type(dev);
        const auto tensors = prefill_experts::tensors(layer);
        return buft && !tensors.empty() && std::all_of(tensors.begin(), tensors.end(), [buft](const ggml_tensor * t) {
            return t && t->data && t->buffer && ggml_is_contiguous(t) &&
                ggml_backend_buffer_get_type(t->buffer) == buft;
        });
    }

    std::unique_ptr<prefill_experts> experts;
    std::vector<ggml_tensor *> source;
    ggml_backend_ptr upload;
    prefill_staging_buffer pinned;
    std::array<ggml_backend_event_ptr, 2> ready;
    std::array<bool, 2> pending = {};
    size_t block;
    size_t part = 0;
    size_t offset = 0;
    size_t submissions = 0;
    size_t copied = 0;
    size_t quota;
    bool direct;
    std::future<void> worker_done;

    prefill_weight_prefetch(std::unique_ptr<prefill_experts> bank, const llama_layer & layer,
                            size_t block, size_t steps, llama_layered_prefill_stats & stats,
                            prefill_staging_pool * pool = nullptr, bool direct = false) :
        experts(std::move(bank)), source(prefill_experts::tensors(layer)), block(block), direct(direct) {
        size_t total = 0;
        for (auto * t : prefill_experts::tensors(experts->layer)) {
            total = prefill_add(total, ggml_nbytes(t));
        }
        quota = std::max<size_t>(1 << 20, total / steps + (total % steps != 0));
        const auto dev = ggml_backend_get_device(experts->backend);
        const auto buft = ggml_backend_dev_host_buffer_type(dev);
        if (!direct) {
            pinned.allocate(pool, buft, 2*block, 2*block, stats);
            if (!pinned || ggml_backend_buffer_get_type(pinned.get()) != buft) {
                return;
            }
        }
        upload.reset(ggml_backend_dev_init(dev, nullptr));
        if (!direct) {
            for (auto & event : ready) {
                event.reset(ggml_backend_event_new(dev));
            }
        }
    }

    ~prefill_weight_prefetch() {
        // The worker owns the upload stream and staging slots until this task completes.
        if (worker_done.valid()) {
            worker_done.wait();
        }
        if (upload) {
            ggml_backend_synchronize(upload.get());
        }
    }

    bool available() const {
        return upload && (direct || (pinned && ready[0] && ready[1]));
    }

    size_t pinned_size() const { return pinned ? ggml_backend_buffer_get_size(pinned.get()) : 0; }

    void submit(prefill_weight_worker & worker) {
        if (!worker_done.valid()) {
            worker_done = worker.submit([this]() { advance(std::numeric_limits<size_t>::max()); });
        }
    }

    size_t advance(size_t bytes) {
        const size_t before = copied;
        const auto dest = prefill_experts::tensors(experts->layer);
        auto * base = pinned ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get())) : nullptr;
        while (part < dest.size() && bytes > 0) {
            const size_t i = submissions % ready.size();
            if (!direct && pending[i]) {
                ggml_backend_event_synchronize(ready[i].get());
            }
            const size_t n = std::min({bytes, direct ? bytes : block, ggml_nbytes(dest[part]) - offset});
            if (direct) {
                // Model storage stays alive until the upload stream has completed.
                const auto * data = static_cast<const uint8_t *>(source[part]->data);
                ggml_backend_tensor_set_async(upload.get(), dest[part], data + offset, offset, n);
            } else {
                auto * bounce = base + i*block;
                experts->copy_source(source, part, offset, n, bounce);
                ggml_backend_tensor_set_async(upload.get(), dest[part], bounce, offset, n);
                ggml_backend_event_record(ready[i].get(), upload.get());
                pending[i] = true;
            }
            ++submissions;
            copied += n;
            bytes -= n;
            offset += n;
            if (offset == ggml_nbytes(dest[part])) {
                ++part;
                offset = 0;
            }
        }
        return copied - before;
    }

    void finish(llama_layered_prefill_stats & stats) {
        if (worker_done.valid()) {
            worker_done.get();
            stats.weight_worker_bytes += copied;
        } else {
            advance(std::numeric_limits<size_t>::max());
        }
        ggml_backend_synchronize(upload.get());
        stats.expert_copy_bytes += copied;
        stats.direct_expert_bytes += direct ? copied : 0;
        stats.prefetched_expert_bytes += copied;
        ++stats.expert_layer_loads;
    }
};

std::unique_ptr<prefill_weight_prefetch> prefill_try_prefetch(
        const llama_layer & layer, ggml_backend_t device, ggml_backend_sched_t sched,
        const llama_layered_prefill_params & params, size_t host_available, size_t transfer_available,
        size_t steps, llama_layered_prefill_stats & stats, size_t compute_reserve = 0,
        std::unique_ptr<prefill_experts> * spare = nullptr, prefill_staging_pool * pool = nullptr) {
    if (!layer.ffn_gate_exps && !layer.ffn_gate_up_exps) {
        return nullptr;
    }
    const size_t bytes = prefill_experts::allocation_size(layer, device);
    size_t payload = 0;
    for (auto * t : prefill_experts::tensors(layer)) {
        payload = prefill_add(payload, ggml_nbytes(t));
    }
    const size_t block = std::min<size_t>(8 << 20, payload);
    const bool direct = params.stream_weights && params.test_weight_staging < 0 &&
        prefill_weight_prefetch::can_upload_direct(layer, device, params.merge_gate_up);
    const size_t pinned_bytes = direct ? 0 : 2*block;
    if (spare && *spare && !(*spare)->compatible(layer)) {
        spare->reset();
    }
    const bool reuse = spare && *spare;
    const size_t additional = reuse ? 0 : bytes;
    size_t free_bytes = 0, total_bytes = 0;
    const auto dev = ggml_backend_get_device(device);
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    const size_t allocated_compute = ggml_backend_sched_get_buffer_size(sched, device);
    const size_t ceiling = compute_reserve ? compute_reserve : params.max_compute_bytes;
    const size_t compute_growth = ceiling > allocated_compute ? ceiling - allocated_compute : 0;
    const bool fits = bytes <= params.max_prefetch_bytes && bytes <= params.max_expert_bytes &&
        additional <= free_bytes && params.prefetch_headroom_bytes <= free_bytes - additional &&
        compute_growth <= free_bytes - additional - params.prefetch_headroom_bytes &&
        pinned_bytes <= host_available && pinned_bytes <= transfer_available &&
        props.caps.async && props.caps.events && props.caps.host_buffer &&
        ggml_backend_dev_host_buffer_type(dev) != nullptr;
    if (!fits) {
        if (spare) {
            spare->reset();
        }
        ++stats.weight_prefetch_skips;
        LLAMA_LOG_DEBUG("%s: skip optional next-layer prefetch: weights=%zu, free=%zu, compute growth=%zu, headroom=%zu\n",
                       __func__, bytes, free_bytes, compute_growth, params.prefetch_headroom_bytes);
        return nullptr;
    }
    std::unique_ptr<prefill_experts> bank;
    if (reuse) {
        bank = std::move(*spare);
        bank->rebind(layer, 0, stats);
    } else {
        ++stats.weight_prefetch_allocation_attempts;
        bank = std::make_unique<prefill_experts>(layer, device,
            std::min(params.max_prefetch_bytes, params.max_expert_bytes), stats, true, true, 0, params.merge_gate_up);
    }
    if (!bank->buffer) {
        ++stats.weight_prefetch_skips;
        LLAMA_LOG_WARN("%s: optional prefetch allocation failed after memory check; using normal layer loading\n", __func__);
        return nullptr;
    }
    std::unique_ptr<prefill_weight_prefetch> prefetch;
    try {
        prefetch = std::make_unique<prefill_weight_prefetch>(std::move(bank), layer, block, steps, stats, pool, direct);
    } catch (const llama_prefill_resource_error & e) {
        ++stats.weight_prefetch_skips;
        LLAMA_LOG_WARN("%s: optional prefetch staging unavailable: %s; using normal layer loading\n", __func__, e.what());
        return nullptr;
    }
    if (!prefetch->available() || prefetch->pinned_size() > pinned_bytes) {
        ++stats.weight_prefetch_skips;
        LLAMA_LOG_WARN("%s: optional pinned prefetch staging unavailable; using normal layer loading\n", __func__);
        return nullptr;
    }
    ++stats.weight_prefetches;
    stats.peak_prefetch_bytes = std::max(stats.peak_prefetch_bytes,
                                        ggml_backend_buffer_get_size(prefetch->experts->buffer.get()));
    stats.prefetch_pinned_bytes = std::max(stats.prefetch_pinned_bytes,
                                          prefetch->pinned_size());
    return prefetch;
}

void prefill_place_staging(ggml_backend_sched_t sched, ggml_cgraph * gf,
                          ggml_backend_t device, const llm_graph_result & res) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (node == res.t_prefill_out || (res.t_prefill_in && node->src[0] == res.t_prefill_in)) {
            if (!ggml_backend_supports_op(device, node)) {
                throw std::runtime_error("selected backend does not support activation staging");
            }
            ggml_backend_sched_set_tensor_backend(sched, node, device);
        }
    }
}

void prefill_check_graph(ggml_backend_sched_t sched, ggml_cgraph * gf,
                         const std::vector<llama_layer> & layers, ggml_backend_t device,
                         const llm_graph_result & res, llama_layered_prefill_stats * stats = nullptr) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (stats) {
            stats->f16_indexer_active |= node->op == GGML_OP_LIGHTNING_INDEXER && node->src[1]->type == GGML_TYPE_F16;
            stats->compact_attention_active |= node->op == GGML_OP_FLASH_ATTN_EXT && node->src[5];
            if (node->op == GGML_OP_LIGHTNING_INDEXER) {
                stats->max_indexer_queries = std::max(stats->max_indexer_queries, uint32_t(node->src[0]->ne[2]));
            }
            for (const auto * source : node->src) {
                if (source) {
                    stats->compact_causal_input |= std::strcmp(source->name, "prefill_causal_positions") == 0;
                    stats->compact_pool_input |= std::strcmp(source->name, "prefill_pool_ranges") == 0;
                }
            }
        }
        bool expert = false;
        if (node->op == GGML_OP_MUL_MAT_ID) {
            for (const auto & layer : layers) {
                expert |= node->src[0] == layer.ffn_gate_exps ||
                          node->src[0] == layer.ffn_gate_up_exps ||
                          node->src[0] == layer.ffn_up_exps ||
                          node->src[0] == layer.ffn_down_exps;
            }
        }
        const bool conversion = node == res.t_prefill_out ||
            (res.t_prefill_in && node->src[0] == res.t_prefill_in);
        const auto assigned = ggml_backend_sched_get_tensor_backend(sched, node);
        if ((expert || conversion) && assigned != device) {
            throw std::runtime_error(format("resident expert/staging node %s (%s) assigned to %s instead of %s",
                node->name, ggml_op_desc(node), assigned ? ggml_backend_name(assigned) : "none", ggml_backend_name(device)));
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
    const bool qwen = model.arch == LLM_ARCH_QWEN35MOE;
    if (!tokens || n_tokens <= 0 || (!qwen && model.arch != LLM_ARCH_GLM5_NEXT) || hp.no_alloc ||
        !memory || cparams.n_seq_max != 1 || n_tokens > (int64_t) cparams.n_ctx ||
        n_tokens > (int64_t) cparams.n_batch || params.n_ubatch == 0 ||
        params.n_ubatch > cparams.n_ubatch ||
        (cparams.n_rs_seq > 0 && params.n_ubatch <= cparams.n_rs_seq + 1) || params.max_host_bytes == 0 ||
        params.max_expert_bytes == 0 || params.max_compute_bytes == 0 ||
        params.group_layers == 0 || params.max_group_layers == 0 ||
        params.test_weight_staging < -1 || params.test_weight_staging > 3 ||
        (params.staging_type != GGML_TYPE_F32 && params.staging_type != GGML_TYPE_BF16) ||
        cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || !cparams.causal_attn ||
        cparams.embeddings || (cparams.embeddings_nextn && (!params.full_layers || cparams.embeddings_nextn_masked)) ||
        (cparams.n_rs_seq != 0 && !params.full_layers) || cparams.cb_eval ||
        !sampling.samplers.empty() || !loras->empty() || moe_cache || !model.hdmd.rot.empty() ||
        std::any_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                    [](bool enabled) { return enabled; })) {
        LLAMA_LOG_ERROR("%s: unsupported request or configuration; see docs/development/layered-prefill.md\n", __func__);
        return -1;
    }
    if ((params.full_layers && qwen) ||
        (params.stream_weights && (!qwen || params.group_layers != 1 || params.max_static_bytes == 0)) ||
        (qwen && params.auto_plan) ||
        (params.expert_slots && (!qwen || params.group_layers != 1 || params.auto_plan ||
                                 params.expert_slots > hp.n_expert || params.expert_slots < hp.n_expert_used()))) {
        LLAMA_LOG_ERROR("%s: full layers require GLM; Qwen requires manual planning and expert slots require single-layer groups with at least top-k slots\n", __func__);
        return -1;
    }
    if (params.stream_weights) {
        for (const auto & item : model.tensors_by_name) {
            if (item.second->buffer && !ggml_backend_buffer_is_host(item.second->buffer)) {
                LLAMA_LOG_ERROR("%s: streamed weights require every model tensor to remain on the host\n", __func__);
                return -1;
            }
        }
    }
    if ((params.merge_gate_up || params.defer_static_upload) &&
        (!params.stream_weights || params.expert_slots)) {
        LLAMA_LOG_ERROR("%s: streamed weight optimizations require complete expert banks\n", __func__);
        return -1;
    }
    if (params.merge_gate_up) {
        for (uint32_t il = 0; il < hp.n_layer(); ++il) {
            const auto & layer = model.layers[il];
            if (!layer.ffn_gate_up_exps &&
                (!prefill_experts::can_merge(layer) || hp.swiglu_clamp_exp[il] > 1e-6f)) {
                LLAMA_LOG_ERROR("%s: gate/up packing requires matching shapes/types without separate scales, biases or activation clamping\n", __func__);
                return -1;
            }
        }
    }
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if ((!qwen && !hp.is_recr(il) && !hp.is_indexer_full(il)) || cvec->tensor_for(il)) {
            LLAMA_LOG_ERROR("%s: shared indexers and control vectors are not supported\n", __func__);
            return -1;
        }
    }
    if (params.full_layers && (params.weight_prefetch || cparams.offload_kqv)) {
        LLAMA_LOG_ERROR("%s: full-layer prefill requires no expert prefetch and CPU state\n", __func__);
        return -1;
    }
    auto * hybrid = static_cast<llama_memory_hybrid *>(memory.get());
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
    if ((params.merge_gate_up || params.defer_static_upload) &&
        device == backend_cpu) {
        LLAMA_LOG_ERROR("%s: streamed compute optimizations require a GPU backend\n", __func__);
        return -1;
    }
    if (params.profile_gpu && (device == backend_cpu || params.auto_plan)) {
        LLAMA_LOG_ERROR("%s: GPU timeline requires a manual GPU prefill, not CPU execution or calibration\n", __func__);
        return -1;
    }
    if (params.full_layers && params.pipeline && params.max_transfer_bytes == 0) {
        auto bounded = params;
        try {
            const size_t row = ggml_row_size(params.staging_type, int64_t(hp.n_embd) * hp.dsv4_hc_mult);
            const uint32_t batch = std::max(std::min<uint32_t>(params.n_ubatch, n_tokens),
                cparams.n_rs_seq > 0 ? cparams.n_rs_seq + 2 : 1);
            if (batch > SIZE_MAX / 8 / row) {
                throw llama_prefill_resource_error("automatic transfer buffer size overflow");
            }
            bounded.max_transfer_bytes = prefill_add(
                prefill_pipeline::projected_device_bytes(device, hp, batch, params.staging_type), 4 * row * batch);
            LLAMA_LOG_INFO("%s: automatic transfer budget %zu bytes, batch ceiling %u\n",
                __func__, bounded.max_transfer_bytes, batch);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
            return -2;
        }
        return prefill_layered(tokens, n_tokens, bounded, stats_out);
    }
    if (params.auto_plan) {
        return prefill_layered_plan(tokens, n_tokens, params, stats_out, device, params.full_layers);
    }
    std::vector<llama_layered_prefill_group> groups;
    for (uint32_t il = 0; il < hp.n_layer();) {
        const uint32_t count = std::min(params.group_layers, hp.n_layer() - il);
        groups.push_back({il, count, 0});
        il += count;
    }
    if (params.weight_prefetch && !params.expert_slots) {
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
    const int64_t execution_start = ggml_time_us();
    const auto & hp = model.hparams;
    const bool pipelined = params.pipeline && device != backend_cpu;
    auto * mask_cache = static_cast<llama_memory_hybrid *>(memory.get())->get_mem_attn();
    mask_cache->reset_mask_stats();

    llama_layered_prefill_stats stats;
    stats.selected_ubatch = params.n_ubatch;
    stats.groups = groups;
    const llama_pos prefix = memory->seq_pos_max(0) + 1;
    stats.reused_prefix_tokens = prefix;
    stats.evaluated_tokens = n_tokens;
    for (const auto & item : memory->memory_breakdown()) {
        if (!ggml_backend_buft_is_host(item.first)) {
            stats.resident_state_bytes = prefill_add(stats.resident_state_bytes, item.second);
        }
    }
    size_t compute_ceiling = 0, device_ceiling = 0, host_ceiling = 0;
    for (const auto & group : groups) {
        device_ceiling = std::max(device_ceiling, group.device_compute_bytes);
        host_ceiling = std::max(host_ceiling, group.compute_bytes - group.device_compute_bytes);
    }
    compute_ceiling = prefill_add(device_ceiling, host_ceiling);
    bool started = false;
    bool computed = false;
    auto * hybrid_memory = model.arch == LLM_ARCH_GLM5_NEXT ?
        static_cast<llama_memory_hybrid_idx *>(memory.get()) : nullptr;
    std::shared_ptr<const llama_memory_hybrid_idx::replay_state> replay;
    std::function<void()> restore_metadata;
    std::unique_ptr<prefill_gpu_profile> profile;
    try {
        synchronize();
        if (params.profile_gpu) {
            profile = std::make_unique<prefill_gpu_profile>(device);
        }
        if (hybrid_memory) {
            replay = hybrid_memory->snapshot_metadata();
            restore_metadata = [&]() { hybrid_memory->restore_metadata(*replay); };
        } else {
            auto * hybrid = static_cast<llama_memory_hybrid *>(memory.get());
            const auto attention = hybrid->get_mem_attn()->snapshot_metadata();
            const auto recurrent = hybrid->get_mem_recr()->snapshot_metadata();
            restore_metadata = [hybrid, attention, recurrent]() {
                hybrid->get_mem_attn()->restore_metadata(attention);
                hybrid->get_mem_recr()->restore_metadata(recurrent);
            };
        }
        const size_t row_bytes = ggml_row_size(params.staging_type, (int64_t) hp.n_embd * std::max(1u, hp.dsv4_hc_mult));
        if ((size_t) n_tokens > params.max_host_bytes / row_bytes) {
            throw llama_prefill_resource_error("layered prefill activation buffer exceeds max_host_bytes");
        }
        const size_t total_bytes = row_bytes * n_tokens;
        const auto activation_buft = device == backend_cpu ? ggml_backend_cpu_buffer_type() :
            ggml_backend_dev_host_buffer_type(ggml_backend_get_device(device));
        if (!activation_buft) {
            throw llama_prefill_resource_error("layered prefill requires pinned host activation storage");
        }
        ggml_backend_buffer_ptr activation_buffer(ggml_backend_buft_alloc_buffer(activation_buft, total_bytes));
        if (!activation_buffer || ggml_backend_buffer_get_type(activation_buffer.get()) != activation_buft ||
            ggml_backend_buffer_get_size(activation_buffer.get()) > params.max_host_bytes) {
            throw llama_prefill_resource_error("failed to allocate the bounded host activation buffer; GPU execution requires pinned memory");
        }
        auto * activations = static_cast<uint8_t *>(ggml_backend_buffer_get_base(activation_buffer.get()));
        stats.activation_host_bytes = ggml_backend_buffer_get_size(activation_buffer.get());
        stats.activation_pinned_bytes = device == backend_cpu ? 0 : stats.activation_host_bytes;
        const size_t host_available = params.max_host_bytes - stats.activation_host_bytes;
        std::unique_ptr<prefill_pipeline> pipeline;
        if (pipelined) {
            pipeline = std::make_unique<prefill_pipeline>(device, hp, std::min<uint32_t>(params.n_ubatch, n_tokens),
                                                         params, stats);
        }
        const bool cuda_staging = params.stream_weights && !params.expert_slots &&
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(device))), "CUDA") == 0;
        const int staging_mode = params.test_weight_staging < 0 ? 3 : params.test_weight_staging;
        std::unique_ptr<prefill_staging_pool> staging_pool;
        if (cuda_staging && (staging_mode & 1)) {
            const size_t transfer_available = params.max_transfer_bytes - stats.transfer_buffer_bytes;
            staging_pool = std::make_unique<prefill_staging_pool>(std::min(host_available, transfer_available), stats);
        }
        std::unique_ptr<prefill_weight_worker> weight_worker;
        std::vector<std::unique_ptr<prefill_weight_prefetch>> next_experts(hp.n_layer());
        std::vector<std::unique_ptr<prefill_static_weights>> next_static(hp.n_layer());
        std::unique_ptr<prefill_experts> spare_experts;
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
        stats.timing.setup_ms = (ggml_time_us() - execution_start) * 1e-3;
        started = true;
        llama_memory_context_ptr mctx;
        for (size_t i_group = 0; i_group < groups.size(); ++i_group) {
            const int64_t group_start = ggml_time_us();
            auto & timing = stats.groups[i_group].timing;
            timing = {};
            const auto & group = groups[i_group];
            const uint32_t il = group.first_layer;
            const uint32_t last = il + group.n_layers;
            prefill_profile_range layer_range(profile.get(), "llama.layer", int(il));
            ggml_backend_sched_reset(sched.get());
            gf_res_prev_active = nullptr;
            const int64_t restore_start = ggml_time_us();
            prefill_profile_range restore_range(profile.get(), "llama.metadata-restore", int(il));
            restore_metadata();
            restore_range.close();
            const int64_t batch_start = ggml_time_us();
            stats.metadata_restore_ms += (batch_start - restore_start) * 1e-3;
            prefill_profile_range batch_range(profile.get(), "llama.batch-init", int(il));
            if (i_group == 0 && !balloc->init(batch, model.vocab, false)) {
                throw std::runtime_error("failed to initialize layered prefill batch");
            }
            batch_range.close();
            const int64_t memory_start = ggml_time_us();
            stats.batch_init_ms += (memory_start - batch_start) * 1e-3;
            prefill_profile_range memory_range(profile.get(), "llama.memory-init", int(il));
            if (i_group == 0) {
                mctx = memory->init_batch(*balloc, params.n_ubatch, false);
            } else {
                static_cast<llama_memory_hybrid_context *>(mctx.get())->rewind();
                ++stats.memory_plan_reuses;
            }
            if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
                throw std::runtime_error("failed to prepare layered prefill memory");
            }
            memory_range.close();
            stats.memory_init_ms += (ggml_time_us() - memory_start) * 1e-3;
            timing.memory_ms += (ggml_time_us() - group_start) * 1e-3;
            size_t group_bytes = 0;
            for (uint32_t lid = il; lid < last; ++lid) {
                if (model.layers[lid].ffn_gate_exps || model.layers[lid].ffn_gate_up_exps) {
                    group_bytes = prefill_add(group_bytes,
                        prefill_experts::allocation_size(model.layers[lid], device, params.expert_slots));
                }
            }
            if (params.max_group_expert_bytes && group_bytes > params.max_group_expert_bytes) {
                throw llama_prefill_resource_error("layered prefill group exceeds max_group_expert_bytes");
            }
            stats.groups[i_group].expert_bytes = group_bytes;
            if (params.max_device_bytes && device != backend_cpu) {
                size_t minimum = prefill_add(group_bytes, stats.resident_state_bytes);
                minimum = prefill_add(minimum, ggml_backend_sched_get_buffer_size(sched.get(), device));
                if (pipeline) {
                    minimum = prefill_add(minimum, ggml_backend_buffer_get_size(pipeline->device_buffer.get()));
                }
                if (params.stream_weights) {
                    minimum = prefill_add(minimum, prefill_static_weights::allocation_size(model.layers[il], device));
                }
                if (minimum > params.max_device_bytes) {
                    throw llama_prefill_resource_error(format("minimum layer working set needs %zu bytes, limit %zu",
                        minimum, params.max_device_bytes));
                }
            }
            if (device != backend_cpu) {
                if (spare_experts && !next_experts[il] &&
                    !spare_experts->compatible(model.layers[il], params.expert_slots)) {
                    spare_experts.reset();
                }
                size_t additional = group_bytes;
                for (uint32_t lid = il; lid < last; ++lid) {
                    if (next_experts[lid]) {
                        additional -= prefill_experts::allocation_size(model.layers[lid], device);
                    }
                    if (spare_experts && !next_experts[il] &&
                        spare_experts->compatible(model.layers[il], params.expert_slots)) {
                        additional -= ggml_backend_buffer_get_size(spare_experts->buffer.get());
                    }
                }
                size_t free_bytes = 0, total_bytes = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(device), &free_bytes, &total_bytes);
                if (additional > free_bytes || params.prefetch_headroom_bytes > free_bytes - additional) {
                    throw llama_prefill_resource_error("current expert group exceeds free device memory and safety headroom");
                }
            }
            stats.peak_expert_bytes = std::max(stats.peak_expert_bytes, group_bytes);
            std::vector<std::unique_ptr<prefill_experts>> banks;
            std::vector<std::unique_ptr<prefill_static_weights>> static_banks;
            std::vector<llama_layer> weights;
            std::unique_ptr<prefill_layer_storage> full_weights;
            std::unique_ptr<prefill_layer_storage> full_state;
            const int64_t load_start = ggml_time_us();
            prefill_profile_range load_range(profile.get(), "llama.weights", int(il));
            const uint64_t copy_before = stats.expert_copy_bytes;
            if (next_static[il]) {
                next_static[il]->finish();
            }
            if (params.full_layers) {
                for (uint32_t lid = il; lid < last; ++lid) {
                    if (prefill_layer_storage::allocation_size(device, prefill_layer_weights(model, lid, 1)) >
                        params.max_expert_bytes) {
                        throw llama_prefill_resource_error("complete layer exceeds the per-layer weight budget");
                    }
                }
                full_weights = std::make_unique<prefill_layer_storage>(device,
                    prefill_layer_weights(model, il, group.n_layers),
                    params.max_group_expert_bytes ? params.max_group_expert_bytes : SIZE_MAX,
                    params.prefetch_headroom_bytes, true, stats.layer_weight_bytes);
                const size_t weight_bytes = ggml_backend_buffer_get_size(full_weights->buffer.get());
                if (params.max_group_expert_bytes && weight_bytes > params.max_group_expert_bytes) {
                    throw llama_prefill_resource_error("complete layer group exceeds the configured group budget");
                }
                const int64_t state_start = ggml_time_us();
                full_state = std::make_unique<prefill_layer_storage>(device,
                    prefill_layer_state(*hybrid_memory, il, group.n_layers), params.max_compute_bytes,
                    params.prefetch_headroom_bytes, false, stats.state_upload_bytes);
                timing.state_upload_ms += (ggml_time_us() - state_start) * 1e-3;
                const size_t state_bytes = full_state->buffer ? ggml_backend_buffer_get_size(full_state->buffer.get()) : 0;
                stats.groups[i_group].expert_bytes = weight_bytes;
                stats.groups[i_group].state_bytes = state_bytes;
                stats.peak_layer_bytes = std::max(stats.peak_layer_bytes,
                    prefill_add(ggml_backend_buffer_get_size(full_weights->buffer.get()), state_bytes));
                LLAMA_LOG_INFO("%s: full layers [%u,%u), peak bank=%zu bytes\n",
                    __func__, il, last, stats.peak_layer_bytes);
            }
            for (uint32_t lid = il; lid < last; ++lid) {
                if (params.full_layers) {
                    weights.push_back(model.layers[lid]);
                    continue;
                }
                std::unique_ptr<prefill_experts> bank;
                if (next_experts[lid]) {
                    next_experts[lid]->finish(stats);
                    bank = std::move(next_experts[lid]->experts);
                    next_experts[lid].reset();
                } else {
                    if (spare_experts && spare_experts->compatible(model.layers[lid], params.expert_slots)) {
                        bank = std::move(spare_experts);
                        bank->rebind(model.layers[lid], params.expert_slots, stats);
                    } else {
                        bank = std::make_unique<prefill_experts>(model.layers[lid], device, params.max_expert_bytes, stats,
                            params.expert_slots != 0 || (params.stream_weights && device != backend_cpu),
                            false, params.expert_slots, params.merge_gate_up);
                    }
                    if (params.stream_weights && !params.expert_slots && device != backend_cpu) {
                        const size_t block = 8 << 20;
                        const bool direct = params.test_weight_staging < 0 &&
                            prefill_weight_prefetch::can_upload_direct(model.layers[lid], device, params.merge_gate_up);
                        const size_t pinned_bytes = direct ? 0 : 2*block;
                        if (params.max_transfer_bytes < stats.transfer_buffer_bytes ||
                            pinned_bytes > params.max_transfer_bytes - stats.transfer_buffer_bytes ||
                            pinned_bytes > host_available ||
                            !ggml_backend_dev_host_buffer_type(ggml_backend_get_device(device))) {
                            throw llama_prefill_resource_error("pinned expert upload exceeds host or transfer budget");
                        }
                        prefill_weight_prefetch loading(std::move(bank), model.layers[lid], block, 1, stats, staging_pool.get(), direct);
                        if (!loading.available() || loading.pinned_size() > pinned_bytes) {
                            throw llama_prefill_resource_error("failed to allocate pinned expert upload buffers");
                        }
                        llama_layered_prefill_stats copied;
                        loading.finish(copied);
                        stats.expert_copy_bytes += copied.expert_copy_bytes;
                        stats.direct_expert_bytes += copied.direct_expert_bytes;
                        stats.expert_layer_loads += copied.expert_layer_loads;
                        bank = std::move(loading.experts);
                    }
                }
                weights.push_back(bank->layer);
                stats.merged_expert_layers += bank->merged_gate_up;
                banks.push_back(std::move(bank));
            }
            if (params.stream_weights) {
                auto current = std::move(next_static[il]);
                if (!current) {
                    current = std::make_unique<prefill_static_weights>(model.layers[il], device, params.max_static_bytes,
                        host_available,
                        params.max_transfer_bytes - stats.transfer_buffer_bytes, false, stats, false, staging_pool.get());
                }
                current->apply(weights[0]);
                stats.peak_static_bytes = std::max(stats.peak_static_bytes, current->size());
                static_banks.push_back(std::move(current));
            }
            std::unique_ptr<prefill_demand> demand;
            if (params.expert_slots) {
                demand = std::make_unique<prefill_demand>(*banks[0], model.layers[il], params.expert_slots, il, params,
                    pipelined, host_available, stats);
            }
            const auto device_usage = [&](size_t workspace) {
                size_t bytes = stats.resident_state_bytes;
                bytes = prefill_add(bytes, workspace);
                if (pipeline) {
                    bytes = prefill_add(bytes, ggml_backend_buffer_get_size(pipeline->device_buffer.get()));
                }
                if (full_weights) {
                    bytes = prefill_add(bytes, ggml_backend_buffer_get_size(full_weights->buffer.get()));
                }
                if (full_state && full_state->buffer) {
                    bytes = prefill_add(bytes, ggml_backend_buffer_get_size(full_state->buffer.get()));
                }
                for (const auto & bank : banks) {
                    if (bank->buffer) {
                        bytes = prefill_add(bytes, ggml_backend_buffer_get_size(bank->buffer.get()));
                    }
                }
                for (const auto & bank : static_banks) {
                    bytes = prefill_add(bytes, bank->size());
                }
                if (demand) {
                    bytes = prefill_add(bytes, ggml_backend_buffer_get_size(demand->buffer.get()));
                }
                if (spare_experts) {
                    bytes = prefill_add(bytes, ggml_backend_buffer_get_size(spare_experts->buffer.get()));
                }
                for (uint32_t lid = last; lid < hp.n_layer(); ++lid) {
                    if (next_experts[lid]) {
                        bytes = prefill_add(bytes, ggml_backend_buffer_get_size(next_experts[lid]->experts->buffer.get()));
                    }
                    if (next_static[lid]) {
                        bytes = prefill_add(bytes, next_static[lid]->size());
                    }
                }
                return bytes;
            };
            if (probe && stats.expert_copy_bytes > copy_before) {
                probe->weight_bytes += stats.expert_copy_bytes - copy_before;
                probe->weight_ms += (ggml_time_us() - load_start) * 1e-3;
            }
            stats.weight_load_ms += (ggml_time_us() - load_start) * 1e-3;
            load_range.close();
            timing.weight_ms += (ggml_time_us() - load_start) * 1e-3 - timing.state_upload_ms;
            if (params.full_layers && params.staging_type == GGML_TYPE_F32 && compute_ceiling) {
                const int64_t reserve_start = ggml_time_us();
                const uint32_t stale = prefix > 0 && hybrid_memory->mem_idx_stale_get()[0] !=
                    llama_memory_hybrid_idx::POS_CLEAN ? prefix / hp.indexer_kpool : 0;
                auto measured = hybrid_memory->init_measure(prefix + n_tokens,
                    stale + params.n_ubatch / hp.indexer_kpool + 2);
                llm_graph_result layout(gf_res_reserve->get_max_nodes());
                auto gp = graph_params(&layout, mctx->get_ubatch(), measured.get(), LLM_GRAPH_TYPE_DECODER);
                gp.n_outputs = 1;
                gp.moe_cache = nullptr;
                gp.prefill_layer = il;
                gp.prefill_layers = group.n_layers;
                gp.prefill_type = params.staging_type;
                gp.prefill_weights = weights.data();
                if (pipeline) {
                    gp.prefill_input = il > 0 ? pipeline->slots[0].input : nullptr;
                    gp.prefill_output = last < hp.n_layer() ? pipeline->slots[0].output : nullptr;
                }
                auto * graph = model.build_graph(gp);
                if (!graph) {
                    throw std::runtime_error("failed to build bounded prefill workspace graph");
                }
                prefill_place_full_graph(sched.get(), graph, device);
                if (!ggml_backend_sched_reserve(sched.get(), graph)) {
                    throw llama_prefill_resource_error("failed to reserve bounded prefill workspace");
                }
                size_t reserved = 0;
                for (auto * backend : backend_ptrs) {
                    reserved = prefill_add(reserved, ggml_backend_sched_get_buffer_size(sched.get(), backend));
                }
                if (reserved > params.max_compute_bytes || reserved > compute_ceiling) {
                    throw llama_prefill_resource_error("reserved prefill workspace exceeds its admitted bound");
                }
                stats.reserved_workspace_bytes = std::max(stats.reserved_workspace_bytes, reserved);
                ggml_backend_sched_reset(sched.get());
                const double elapsed = (ggml_time_us() - reserve_start) * 1e-3;
                timing.workspace_reserve_ms += elapsed;
                timing.graph_ms += elapsed;
            }
            size_t offset = 0;
            size_t i_batch = 0;
            if (pipeline && il > 0) {
                const int64_t upload_start = ggml_time_us();
                pipeline->prefetch_input(0, activations, row_bytes * mctx->get_ubatch().n_tokens);
                timing.activation_upload_ms += (ggml_time_us() - upload_start) * 1e-3;
            }
            bool more;
            do {
                const auto & ubatch = mctx->get_ubatch();
                const size_t slot = i_batch % 2;
                if (pipeline) {
                    const int64_t download_start = ggml_time_us();
                    pipeline->finish_output(slot);
                    timing.activation_download_ms += (ggml_time_us() - download_start) * 1e-3;
                }
                const int64_t prepare_start = ggml_time_us();
                prefill_profile_range setup_range(profile.get(), "llama.setup", int(il), int(i_batch));
                prefill_profile_range apply_range(profile.get(), "llama.memory-apply", int(il), int(i_batch));
                if (!mctx->apply()) {
                    throw std::runtime_error("failed to apply layered prefill memory");
                }
                const int64_t prepare_end = ggml_time_us();
                apply_range.close();
                stats.memory_apply_ms += (prepare_end - prepare_start) * 1e-3;
                prefill_profile_range build_range(profile.get(), "llama.graph-build", int(il), int(i_batch));
                timing.memory_ms += (prepare_end - prepare_start) * 1e-3;
                n_outputs = last == hp.n_layer() && offset + ubatch.n_tokens == (size_t) n_tokens ? 1 : 0;
                auto * res = get_gf_res_prev();
                auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DECODER);
                if (model.arch == LLM_ARCH_QWEN35MOE) {
                    const auto callback = gparams.cb;
                    gparams.cb = [&, callback](const llama_ubatch & batch, ggml_tensor * node, const char * name, int lid) {
                        callback(batch, node, name, lid);
                        // Keep routing and shared arithmetic placement independent of expert scheduling.
                        const bool shared = std::strncmp(name, "ffn_", 4) == 0 &&
                                            std::strncmp(name, "ffn_moe_", 8) != 0;
                        const bool routing = std::strncmp(name, "ffn_moe_weights", 15) == 0 ||
                                             std::strncmp(name, "ffn_moe_logits", 14) == 0 ||
                                             std::strncmp(name, "ffn_moe_probs", 13) == 0 ||
                                             std::strcmp(name, "ffn_moe_argsort") == 0 ||
                                             std::strcmp(name, "ffn_moe_topk") == 0;
                        if (lid >= 0 && (shared || routing || std::strncmp(name, "shared_expert_", 14) == 0 ||
                                         std::strcmp(name, "prefill_expert_ready") == 0 ||
                                         std::strcmp(name, "l_out") == 0)) {
                            for (auto backend : backend_ptrs) {
                                if (ggml_backend_get_device(backend) == (params.stream_weights ?
                                        ggml_backend_get_device(device) : model.dev_layer(lid)) &&
                                    ggml_backend_supports_op(backend, node)) {
                                    ggml_backend_sched_set_tensor_backend(sched.get(), node, backend);
                                    break;
                                }
                                if (params.stream_weights && lid >= 0) {
                                    if (!ggml_backend_supports_op(device, node)) {
                                        throw std::runtime_error(format("streamed layer operation %s is not supported on %s",
                                            name, ggml_backend_name(device)));
                                    }
                                    ggml_backend_sched_set_tensor_backend(sched.get(), node, device);
                                }
                            }
                        }
                    };
                }
                gparams.moe_cache = nullptr;
                gparams.prefill_layer = il;
                gparams.prefill_layers = group.n_layers;
                gparams.prefill_type = params.staging_type;
                gparams.prefill_weights = weights.data();
                gparams.prefill_expert_map = demand ? demand->map : nullptr;
                if (pipeline) {
                    gparams.prefill_input = il > 0 ? pipeline->slots[slot].input : nullptr;
                    gparams.prefill_output = last < hp.n_layer() ? pipeline->slots[slot].output : nullptr;
                }
                ggml_cgraph * gf = nullptr;
                const bool use_admitted = params.full_layers && params.staging_type == GGML_TYPE_F32 && compute_ceiling > 0;
                size_t measured_compute = stats.groups[i_group].compute_bytes;
                if (params.full_layers && !graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
                    gf = res->get_gf();
                    ++stats.graph_reuses;
                    ++n_reused;
                } else {
                    gf_res_prev_active = nullptr;
                    ggml_backend_sched_reset(sched.get());
                    res->reset();
                    const auto build = [&]() {
                        const int64_t begin = ggml_time_us();
                        auto * graph = model.build_graph(gparams);
                        if (!graph) {
                            throw std::runtime_error("failed to build layered prefill graph");
                        }
                        if (params.full_layers) {
                            prefill_place_full_graph(sched.get(), graph, device);
                        }
                        prefill_place_staging(sched.get(), graph, device, *res);
                        const double elapsed = (ggml_time_us() - begin) * 1e-3;
                        timing.graph_build_ms += elapsed;
                        stats.graph_build_ms += elapsed;
                        ++stats.graph_builds;
                        return graph;
                    };
                    gf = build();
                    if (!use_admitted) {
                        const int64_t measure_start = ggml_time_us();
                        std::vector<size_t> sizes(backend_ptrs.size());
                        ggml_backend_sched_reserve_size(sched.get(), gf, sizes.data());
                        prefill_check_graph(sched.get(), gf, weights, device, *res, &stats);
                        measured_compute = 0;
                        for (size_t i = 0; i < sizes.size(); ++i) {
                            measured_compute = prefill_add(measured_compute, sizes[i]);
                            if (backend_ptrs[i] == device) {
                                stats.groups[i_group].device_compute_bytes =
                                    std::max(stats.groups[i_group].device_compute_bytes, sizes[i]);
                            }
                        }
                        if (measured_compute > params.max_compute_bytes ||
                            (compute_ceiling && measured_compute > compute_ceiling)) {
                            throw llama_prefill_resource_error("layered prefill workspace exceeds max_compute_bytes; reduce n_ubatch");
                        }
                        const double elapsed = (ggml_time_us() - measure_start) * 1e-3;
                        timing.graph_measure_ms += elapsed;
                        stats.graph_reserve_ms += elapsed;
                        ++stats.graph_measurements;
                        // Measurement rewrites graph sources to scheduler-owned copies.
                        ggml_backend_sched_reset(sched.get());
                        res->reset();
                        gf = build();
                    }
                    const int64_t alloc_start = ggml_time_us();
                    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                        throw llama_prefill_resource_error("failed to allocate layered prefill graph");
                    }
                    const double elapsed = (ggml_time_us() - alloc_start) * 1e-3;
                    timing.graph_alloc_ms += elapsed;
                    stats.graph_alloc_ms += elapsed;
                    gf_res_prev_active = res;
                }
                build_range.close();
                size_t compute_bytes = 0;
                for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                    backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
                    compute_bytes = prefill_add(compute_bytes, backend_buf_exp_size[i]);
                    if (use_admitted && backend_ptrs[i] == device) {
                        stats.groups[i_group].device_compute_bytes =
                            std::max(stats.groups[i_group].device_compute_bytes, backend_buf_exp_size[i]);
                    }
                }
                const size_t allocated_compute = compute_bytes;
                if (allocated_compute > params.max_compute_bytes) {
                    throw llama_prefill_resource_error(format("retained prefill workspace needs %zu bytes, limit %zu",
                        allocated_compute, params.max_compute_bytes));
                }
                if (!use_admitted) {
                    compute_bytes = measured_compute;
                }
                if (device != backend_cpu) {
                    const size_t workspace = std::max(stats.groups[i_group].device_compute_bytes,
                        ggml_backend_sched_get_buffer_size(sched.get(), device));
                    const size_t total_device = device_usage(workspace);
                    stats.peak_device_bytes = std::max(stats.peak_device_bytes, total_device);
                    if (params.max_device_bytes && total_device > params.max_device_bytes) {
                        throw llama_prefill_resource_error(format("device working set needs %zu bytes, limit %zu; reduce n_ubatch",
                            total_device, params.max_device_bytes));
                    }
                }
                if (compute_bytes > params.max_compute_bytes || (compute_ceiling && compute_bytes > compute_ceiling)) {
                    throw llama_prefill_resource_error(format(
                        "allocated prefill graph exceeds its admitted workspace bound: actual=%zu planned=%zu limit=%zu",
                        compute_bytes, compute_ceiling, params.max_compute_bytes));
                }
                stats.groups[i_group].compute_bytes = std::max(stats.groups[i_group].compute_bytes, compute_bytes);
                stats.peak_compute_bytes = std::max(stats.peak_compute_bytes, std::max(compute_bytes, allocated_compute));
                stats.max_backend_segments = std::max(stats.max_backend_segments,
                    uint32_t(ggml_backend_sched_get_n_splits(sched.get())));
                prefill_check_graph(sched.get(), gf, weights, device, *res, &stats);
                const int64_t inputs_start = ggml_time_us();
                timing.graph_ms += (inputs_start - prepare_end) * 1e-3;
                prefill_profile_range inputs_range(profile.get(), "llama.inputs", int(il), int(i_batch));
                res->set_inputs(&ubatch);
                inputs_range.close();
                const int64_t inputs_end = ggml_time_us();
                stats.graph_inputs_ms += (inputs_end - inputs_start) * 1e-3;
                prefill_profile_range input_wait_range(profile.get(), "llama.input-wait", int(il), int(i_batch));
                timing.inputs_ms += (inputs_end - inputs_start) * 1e-3;
                const size_t bytes = row_bytes * ubatch.n_tokens;
                if (res->t_prefill_in) {
                    if (pipeline) {
                        pipeline->wait_input(slot, bytes);
                    } else {
                        ggml_backend_tensor_set(res->t_prefill_in, activations + offset * row_bytes, 0, bytes);
                    }
                    stats.activation_read_bytes += bytes;
                    if (probe) {
                        probe->read_bytes += bytes;
                        probe->read_ms += (ggml_time_us() - inputs_end) * 1e-3;
                    }
                }
                input_wait_range.close();
                const int64_t prefetch_start = ggml_time_us();
                stats.input_wait_ms += (prefetch_start - inputs_end) * 1e-3;
                timing.activation_upload_ms += (prefetch_start - inputs_end) * 1e-3;
                prefill_profile_range prefetch_range(profile.get(), "llama.prefetch-setup", int(il), int(i_batch));
                if (i_batch == 0 && params.weight_prefetch &&
                    !demand && device != backend_cpu && i_group + 1 < groups.size()) {
                    size_t extra_bytes = 0, extra_pinned = 0;
                    for (uint32_t p = 0; p < group.n_prefetch_layers; ++p) {
                        auto limits = params;
                        limits.max_prefetch_bytes -= extra_bytes;
                        if (params.stream_weights && params.max_group_expert_bytes) {
                            const size_t occupied = prefill_add(group_bytes, extra_bytes);
                            limits.max_prefetch_bytes = std::min(limits.max_prefetch_bytes,
                                params.max_group_expert_bytes > occupied ? params.max_group_expert_bytes - occupied : 0);
                        }
                        if (params.max_device_bytes) {
                            const size_t used = device_usage(std::max(params.max_compute_bytes,
                                ggml_backend_sched_get_buffer_size(sched.get(), device)));
                            size_t available = params.max_device_bytes > used ? params.max_device_bytes - used : 0;
                            if (spare_experts && spare_experts->compatible(model.layers[last + p])) {
                                available = prefill_add(available, ggml_backend_buffer_get_size(spare_experts->buffer.get()));
                            }
                            limits.max_prefetch_bytes = std::min(limits.max_prefetch_bytes, available);
                        }
                        next_experts[last + p] = prefill_try_prefetch(model.layers[last + p], device, sched.get(), limits,
                            host_available - extra_pinned,
                            params.max_transfer_bytes > stats.transfer_buffer_bytes + extra_pinned ?
                                params.max_transfer_bytes - stats.transfer_buffer_bytes - extra_pinned : 0,
                            probe && probe->prefetch_steps ? probe->prefetch_steps :
                                ((size_t) n_tokens + params.n_ubatch - 1) / params.n_ubatch,
                            stats, device_ceiling ? device_ceiling : compute_ceiling,
                            params.stream_weights ? &spare_experts : nullptr, staging_pool.get());
                        if (next_experts[last + p]) {
                            extra_bytes += ggml_backend_buffer_get_size(next_experts[last + p]->experts->buffer.get());
                            extra_pinned += next_experts[last + p]->pinned_size();
                        } else if (model.layers[last + p].ffn_gate_exps || model.layers[last + p].ffn_gate_up_exps) {
                            break;
                        }
                    }
                }
                if (params.stream_weights && params.weight_prefetch && i_batch == 0 &&
                    last < hp.n_layer() && device != backend_cpu) {
                    const size_t bytes = prefill_static_weights::allocation_size(model.layers[last], device);
                    size_t free_bytes = 0, total_device = 0;
                    ggml_backend_dev_memory(ggml_backend_get_device(device), &free_bytes, &total_device);
                    const size_t allocated = ggml_backend_sched_get_buffer_size(sched.get(), device);
                    const size_t growth = params.max_compute_bytes > allocated ? params.max_compute_bytes - allocated : 0;
                    const size_t current = static_banks[0]->size();
                    size_t expert_pinned = 0;
                    for (uint32_t lid = last; lid < hp.n_layer(); ++lid) {
                        if (next_experts[lid]) {
                            expert_pinned = prefill_add(expert_pinned,
                                next_experts[lid]->pinned_size());
                        }
                    }
                    const size_t transfer_used = prefill_add(stats.transfer_buffer_bytes, expert_pinned);
                    const size_t demand_pinned = demand && demand->pinned ?
                        ggml_backend_buffer_get_size(demand->pinned.get()) : 0;
                    const size_t host_used = prefill_add(stats.activation_host_bytes, prefill_add(expert_pinned, demand_pinned));
                    const bool fits = current <= params.max_static_bytes && bytes <= params.max_static_bytes - current &&
                        bytes <= free_bytes && params.prefetch_headroom_bytes <= free_bytes - bytes &&
                        growth <= free_bytes - bytes - params.prefetch_headroom_bytes &&
                        transfer_used <= params.max_transfer_bytes &&
                        bytes <= params.max_transfer_bytes - transfer_used &&
                        host_used <= params.max_host_bytes && bytes <= params.max_host_bytes - host_used &&
                        (!params.max_device_bytes || prefill_add(device_usage(std::max(allocated, params.max_compute_bytes)), bytes) <= params.max_device_bytes);
                    if (fits) {
                        try {
                            next_static[last] = std::make_unique<prefill_static_weights>(model.layers[last], device,
                                params.max_static_bytes - current,
                                params.max_host_bytes - host_used,
                                params.max_transfer_bytes - transfer_used, true, stats, params.defer_static_upload, staging_pool.get());
                            stats.peak_static_bytes = std::max(stats.peak_static_bytes, current + next_static[last]->size());
                        } catch (const llama_prefill_resource_error & e) {
                            ++stats.static_prefetch_skips;
                            LLAMA_LOG_WARN("%s: optional static prefetch skipped: %s\n", __func__, e.what());
                        }
                    } else {
                        ++stats.static_prefetch_skips;
                    }
                }
                const int64_t compute_start = ggml_time_us();
                prefetch_range.close();
                stats.prefetch_setup_ms += (compute_start - prefetch_start) * 1e-3;
                setup_range.close();
                stats.graph_setup_ms += (compute_start - prepare_start) * 1e-3;
                timing.weight_ms += (compute_start - prefetch_start) * 1e-3;
                computed = true;
                if (demand) {
                    demand->ids = res->t_prefill_ids;
                }
                ggml_backend_sched_set_eval_callback(sched.get(), demand ? prefill_demand::callback : nullptr, demand.get());
                prefill_profile_range submit_range(profile.get(), "llama.submit", int(il), int(i_batch));
                const auto status = graph_compute(gf, true);
                submit_range.close();
                ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
                prefill_profile_range enqueue_range(profile.get(), "llama.prefetch-enqueue", int(il), int(i_batch));
                const size_t next = offset + ubatch.n_tokens;
                if (pipeline && il > 0 && next < (size_t) n_tokens && status == GGML_STATUS_SUCCESS) {
                    const auto * following = static_cast<const llama_memory_hybrid_context *>(mctx.get())->get_next_ubatch();
                    if (!following || following->n_tokens == 0 || following->n_tokens > (size_t) n_tokens - next) {
                        throw std::runtime_error("layered prefill pipeline has an invalid next microbatch");
                    }
                    const size_t next_bytes = row_bytes * following->n_tokens;
                    pipeline->prefetch_input((i_batch + 1) % 2, activations + next * row_bytes, next_bytes);
                    ++stats.input_prefetches;
                }
                if (status == GGML_STATUS_SUCCESS && last < hp.n_layer() &&
                    next_static[last] && params.defer_static_upload) {
                    next_static[last]->enqueue(stats);
                }
                for (uint32_t p = 0; p < group.n_prefetch_layers && status == GGML_STATUS_SUCCESS; ++p) {
                    if (next_experts[last + p]) {
                        if (cuda_staging && (staging_mode & 2)) {
                            if (!weight_worker) {
                                weight_worker = std::make_unique<prefill_weight_worker>();
                            }
                            next_experts[last + p]->submit(*weight_worker);
                        } else {
                            stats.weight_prefetch_submit_bytes += next_experts[last + p]->advance(next_experts[last + p]->quota);
                        }
                    }
                }
                enqueue_range.close();
                prefill_profile_range wait_range(profile.get(), "llama.wait", int(il), int(i_batch));
                synchronize();
                wait_range.close();
                if (demand && !demand->error.empty()) {
                    throw std::runtime_error(demand->error);
                }
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("layered prefill computation failed or was aborted");
                }
                const int64_t compute_end = ggml_time_us();
                stats.compute_wait_ms += (compute_end - compute_start) * 1e-3;
                timing.compute_ms += (compute_end - compute_start) * 1e-3;
                if (probe) {
                    const auto * hybrid = static_cast<const llama_memory_hybrid_idx_context *>(mctx.get());
                    probe->points.push_back({il, ubatch.n_tokens, hybrid->get_attn()->get_n_kv(),
                        (compute_end - compute_start) * 1e-3,
                        (prepare_end - prepare_start + inputs_end - inputs_start) * 1e-3});
                }
                const int64_t download_start = ggml_time_us();
                if (res->t_prefill_out) {
                    if (pipeline) {
                        pipeline->copy_output(slot, activations + offset * row_bytes, bytes);
                        ++stats.async_output_batches;
                    } else {
                        ggml_backend_tensor_get(res->t_prefill_out, activations + offset * row_bytes, 0, bytes);
                    }
                    stats.activation_write_bytes += bytes;
                    if (probe) {
                        probe->write_bytes += bytes;
                        probe->write_ms += (ggml_time_us() - compute_end) * 1e-3;
                    }
                }
                const int64_t output_start = ggml_time_us();
                timing.activation_download_ms += (output_start - download_start) * 1e-3;
                if (n_outputs) {
                    ggml_backend_tensor_get(res->t_logits, logits.data, 0, model.vocab.n_tokens() * sizeof(float));
                }
                if (cparams.embeddings_nextn && last == hp.n_layer()) {
                    if (!res->t_h_nextn || !embd_nextn.data) {
                        throw std::runtime_error("full-layer prefill did not produce MTP hidden states");
                    }
                    ggml_backend_tensor_get(res->t_h_nextn, embd_nextn.data + offset * hp.n_embd,
                                            0, size_t(ubatch.n_tokens) * hp.n_embd * sizeof(float));
                }
                timing.output_ms += (ggml_time_us() - output_start) * 1e-3;
                stats.activation_copy_ms += (ggml_time_us() - compute_end) * 1e-3;
                offset += ubatch.n_tokens;
                ++stats.microbatches;
                ++i_batch;
                const int64_t next_start = ggml_time_us();
                more = mctx->next();
                timing.memory_ms += (ggml_time_us() - next_start) * 1e-3;
            } while (more);
            if (offset != (size_t) n_tokens) {
                throw std::runtime_error("layered prefill did not consume the complete prompt");
            }
            if (pipeline) {
                const int64_t drain_start = ggml_time_us();
                pipeline->drain();
                timing.activation_download_ms += (ggml_time_us() - drain_start) * 1e-3;
            }
            const int64_t reset_start = ggml_time_us();
            ggml_backend_sched_reset(sched.get());
            for (auto & res : gf_res_prev) {
                if (res) {
                    res->reset();
                }
            }
            if (params.stream_weights && device != backend_cpu) {
                spare_experts = std::move(banks[0]);
            }
            timing.cleanup_ms += (ggml_time_us() - reset_start) * 1e-3;
            if (full_state) {
                const int64_t state_start = ggml_time_us();
                full_state->download(stats.state_download_bytes);
                timing.state_download_ms += (ggml_time_us() - state_start) * 1e-3;
            }
            const int64_t release_start = ggml_time_us();
            full_state.reset();
            full_weights.reset();
            demand.reset();
            banks.clear();
            static_banks.clear();
            timing.cleanup_ms += (ggml_time_us() - release_start) * 1e-3;
            timing.total_ms = (ggml_time_us() - group_start) * 1e-3;
            stats.timing.add(timing);
        }
        const int64_t finish_start = ggml_time_us();
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
        const auto mask_after = mask_cache->get_mask_stats();
        stats.mask_ms = mask_after.time_us * 1e-3;
        stats.mask_calls = mask_after.calls;
        stats.mask_parallel_calls = mask_after.parallel_calls;
        stats.mask_max_threads = mask_after.max_threads;
        next_experts.clear();
        weight_worker.reset();
        next_static.clear();
        spare_experts.reset();
        staging_pool.reset();
        pipeline.reset();
        activation_buffer.reset();
        restore_metadata = {};
        replay.reset();
        stats.timing.cleanup_ms += (ggml_time_us() - finish_start) * 1e-3;
        stats.timing.total_ms = (ggml_time_us() - execution_start) * 1e-3;
        if (stats_out) {
            *stats_out = stats;
        }
        if (profile) {
            profile->finish();
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
            ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
            if (prefix > 0 && !computed && restore_metadata) {
                restore_metadata();
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
    diagnostic.capacity_only = params.full_layers;
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
        if ((size_t) n_tokens > SIZE_MAX / row) {
            diagnostic.budget_shortfall_bytes = SIZE_MAX - params.max_host_bytes;
            diagnostic.shortfall_is_lower_bound = true;
            throw llama_prefill_resource_error("resident host staging exceeds the addressable size");
        }
        if ((size_t) n_tokens > params.max_host_bytes / row) {
            const size_t required = row * n_tokens;
            diagnostic.budget_shortfall_bytes = required - params.max_host_bytes;
            throw llama_prefill_resource_error(format("resident host staging needs %zu bytes, limit %zu, gap %zu",
                required, params.max_host_bytes, diagnostic.budget_shortfall_bytes));
        }
        const size_t activation_bytes = row * n_tokens;
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
            if (params.full_layers) {
                bytes = prefill_layer_storage::allocation_size(device, prefill_layer_weights(model, il, 1));
            } else if (model.layers[il].ffn_gate_exps) {
                bytes = prefill_experts::allocation_size(model.layers[il], device);
                for (auto * t : prefill_experts::tensors(model.layers[il])) {
                    data = prefill_add(data, ggml_nbytes(t));
                }
            }
            if (bytes > params.max_expert_bytes) {
                diagnostic.budget_shortfall_bytes = bytes - params.max_expert_bytes;
                throw llama_prefill_resource_error(format("%s layer needs %zu bytes, per-layer limit %zu, gap %zu",
                    params.full_layers ? "complete" : "resident expert", bytes,
                    params.max_expert_bytes, diagnostic.budget_shortfall_bytes));
            }
            if (params.max_group_expert_bytes && bytes > params.max_group_expert_bytes) {
                diagnostic.budget_shortfall_bytes = bytes - params.max_group_expert_bytes;
                throw llama_prefill_resource_error(format("%s layer needs %zu bytes, group limit %zu, gap %zu",
                    params.full_layers ? "complete" : "resident expert", bytes,
                    params.max_group_expert_bytes, diagnostic.budget_shortfall_bytes));
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
        const uint32_t min_batch = cparams.n_rs_seq > 0 ? cparams.n_rs_seq + 2 : 1;
        const uint32_t largest_batch = std::max(std::min<uint32_t>(params.n_ubatch, n_tokens), min_batch);
        for (uint32_t b = largest_batch; b >= min_batch; b /= 2) {
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
            LLAMA_LOG_WARN("%s: selecting a %s capacity plan without destructive calibration\n",
                __func__, params.full_layers ? "full-layer" : "resident");
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
                const auto full_tensors = params.full_layers ? prefill_layer_weights(model, first, count) :
                    std::vector<ggml_tensor *>();
                const auto state_tensors = params.full_layers ? prefill_layer_state(*hybrid_memory, first, count) :
                    std::vector<ggml_tensor *>();
                if (params.full_layers) {
                    weight_bytes = prefill_layer_storage::allocation_size(device, full_tensors);
                    result.group.state_bytes = prefill_layer_storage::allocation_size(device, state_tensors);
                }
                const size_t bank_bytes = prefill_add(weight_bytes, result.group.state_bytes);
                if ((params.max_group_expert_bytes && weight_bytes > params.max_group_expert_bytes) ||
                    bank_bytes > device_budget || result.group.state_bytes > params.max_compute_bytes) {
                    throw llama_prefill_resource_error("group expert budget");
                }
                result.group.expert_bytes = weight_bytes;
                if (sample_batches.find(b) == sample_batches.end()) {
                    llama_batch_ext batch(this);
                    const uint32_t sample_tokens = std::min<uint32_t>(b, n_tokens);
                    for (uint32_t i = 0; i < sample_tokens; ++i) {
                        const auto idx = batch.add_token(0);
                        const llama_pos pos = prefix_tokens + i;
                        if (idx < 0 || !batch.set_token_id(idx, tokens[i]) ||
                            !batch.set_token_pos(idx, &pos) || !batch.set_output(idx, i + 1 == sample_tokens)) {
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
                std::unique_ptr<prefill_layer_storage> full_weights;
                std::unique_ptr<prefill_layer_storage> full_state;
                if (params.full_layers) {
                    // Zero-sized placement markers keep dry graphs free of device allocations and transfers.
                    full_weights = std::make_unique<prefill_layer_storage>(device, full_tensors,
                        weight_bytes, params.prefetch_headroom_bytes, true, ignored.layer_weight_bytes, true);
                    full_state = std::make_unique<prefill_layer_storage>(device, state_tensors,
                        params.max_compute_bytes, params.prefetch_headroom_bytes, false, ignored.state_upload_bytes, true);
                }
                for (uint32_t il = first; il < first + count; ++il) {
                    if (params.full_layers) {
                        weights.push_back(model.layers[il]);
                        continue;
                    }
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
                ++diagnostic.planner_graphs;
                auto * gf = model.build_graph(gp);
                std::vector<size_t> sizes(backend_ptrs.size());
                if (params.full_layers) {
                    prefill_place_full_graph(sched.get(), gf, device);
                }
                ggml_backend_sched_reserve_size(sched.get(), gf, sizes.data());
                prefill_check_graph(sched.get(), gf, weights, device, res);
                for (size_t i = 0; i < sizes.size(); ++i) {
                    result.group.compute_bytes = prefill_add(result.group.compute_bytes, sizes[i]);
                    if (backend_ptrs[i] == device) {
                        result.group.device_compute_bytes = sizes[i];
                    }
                }
                if (!params.full_layers) {
                    ggml_backend_sched_reset(sched.get());
                    res.reset();
                    model.build_graph(gp);
                }
                result.group.dispatch_ms = (ggml_time_us() - start) * 1e-3;
                ggml_backend_sched_reset(sched.get());
                const size_t transfer_device = transfer_device_bytes(b);
                result.fits = result.group.compute_bytes <= params.max_compute_bytes &&
                    transfer_device <= params.max_transfer_bytes &&
                    bank_bytes <= device_budget &&
                    result.group.device_compute_bytes <= device_budget - bank_bytes &&
                    transfer_device <= device_budget - bank_bytes - result.group.device_compute_bytes;
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
        if (params.full_layers) {
            std::vector<size_t> state_sizes;
            for (uint32_t il = 0; il < hp.n_layer(); ++il) {
                state_sizes.push_back(prefill_layer_storage::allocation_size(
                    device, prefill_layer_state(*hybrid_memory, il, 1)));
            }
            for (uint32_t b : batches) {
                const size_t transfer_device = transfer_device_bytes(b);
                size_t workspace = std::min(params.max_compute_bytes, gpu ? device_budget / 4 : params.max_compute_bytes);
                for (uint32_t il = 0; il < hp.n_layer(); ++il) {
                    const size_t fixed = prefill_add(prefill_add(allocation[il], state_sizes[il]), transfer_device);
                    workspace = std::min(workspace, fixed < device_budget ? device_budget - fixed : 0);
                }
                while (true) {
                    candidate plan = {b, {}, 0};
                    ++diagnostic.planner_candidates;
                    uint32_t first = 0;
                    while (first < hp.n_layer()) {
                        llama_layered_prefill_group group;
                        group.first_layer = first;
                        group.n_layers = 0;
                        while (group.n_layers < params.max_group_layers && first + group.n_layers < hp.n_layer()) {
                            const uint32_t il = first + group.n_layers;
                            const size_t weight = prefill_add(group.expert_bytes, allocation[il]);
                            const size_t state = prefill_add(group.state_bytes, state_sizes[il]);
                            const size_t required = prefill_add(prefill_add(weight, state),
                                prefill_add(workspace, transfer_device));
                            if ((params.max_group_expert_bytes && weight > params.max_group_expert_bytes) ||
                                state > params.max_compute_bytes || required > device_budget) {
                                break;
                            }
                            group.expert_bytes = weight;
                            group.state_bytes = state;
                            ++group.n_layers;
                        }
                        if (group.n_layers == 0) {
                            break;
                        }
                        first += group.n_layers;
                        plan.groups.push_back(group);
                    }
                    if (first != hp.n_layer()) {
                        break;
                    }
                    bool fits = true;
                    size_t measured_workspace = 0, host_workspace = 0;
                    for (auto & group : plan.groups) {
                        const auto admitted = measure(b, group.first_layer, group.n_layers);
                        group = admitted.group;
                        fits &= admitted.fits;
                        measured_workspace = std::max(measured_workspace, group.device_compute_bytes);
                        host_workspace = std::max(host_workspace, group.compute_bytes - group.device_compute_bytes);
                    }
                    bool capacity = true;
                    for (const auto & group : plan.groups) {
                        const size_t required = prefill_add(prefill_add(group.expert_bytes, group.state_bytes),
                            prefill_add(measured_workspace, transfer_device));
                        capacity &= required <= device_budget;
                    }
                    if (fits && capacity && prefill_add(measured_workspace, host_workspace) <= params.max_compute_bytes) {
                        candidates.push_back(std::move(plan));
                        break;
                    }
                    if (measured_workspace > workspace && measured_workspace <= params.max_compute_bytes) {
                        LLAMA_LOG_INFO("%s: full-layer dry graph raises workspace reserve from %zu to %zu bytes\n",
                            __func__, workspace, measured_workspace);
                        workspace = measured_workspace;
                        continue;
                    }
                    break;
                }
                if (!candidates.empty()) {
                    break;
                }
                LLAMA_LOG_INFO("%s: full-layer plan at batch %u exceeds capacity; trying a smaller batch\n", __func__, b);
            }
        }
        if (params.full_layers) {
            active_batches.clear();
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
                if (params.full_layers) {
                    bool fits = true;
                    for (const auto & group : plan.groups) {
                        const size_t required = prefill_add(prefill_add(group.expert_bytes, group.state_bytes),
                            prefill_add(device_compute, transfer_device_bytes(b)));
                        fits &= required <= device_budget;
                    }
                    if (!fits) {
                        continue;
                    }
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
                            const size_t current = prefill_add(plan.groups[g].expert_bytes,
                                                              prefill_add(device_compute, transfer_device));
                            if (extra > params.max_prefetch_bytes || current > device_budget ||
                                extra > device_budget - current ||
                                pinned > params.max_transfer_bytes - transfer_device ||
                                pinned > params.max_host_bytes - activation_bytes) {
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
                fallback.planner_graphs += diagnostic.planner_graphs;
                fallback.planning_ms = (ggml_time_us() - planning_start) * 1e-3;
                if (stats_out) {
                    *stats_out = std::move(fallback);
                }
                return status;
            }
            for (uint32_t il = 0; il < hp.n_layer(); ++il) {
                const auto minimum = measure(min_batch, il, 1);
                const size_t transfer_device = transfer_device_bytes(min_batch);
                const size_t required = prefill_add(prefill_add(allocation[il], minimum.group.state_bytes),
                    prefill_add(minimum.group.device_compute_bytes, transfer_device));
                diagnostic.minimum_additional_bytes = std::max(diagnostic.minimum_additional_bytes, required);
                if (allocation[il] > device_budget) {
                    diagnostic.shortfall_is_lower_bound = true;
                }
                if (minimum.group.compute_bytes > params.max_compute_bytes) {
                    diagnostic.budget_shortfall_bytes = std::max(diagnostic.budget_shortfall_bytes,
                        minimum.group.compute_bytes - params.max_compute_bytes);
                }
                if (minimum.group.state_bytes > params.max_compute_bytes) {
                    diagnostic.budget_shortfall_bytes = std::max(diagnostic.budget_shortfall_bytes,
                        minimum.group.state_bytes - params.max_compute_bytes);
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
            result.planner_graphs = diagnostic.planner_graphs;
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
