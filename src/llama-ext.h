#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Experimental: single-sequence text prefill, final-token logits only.
// Limits are bytes and exclude model and persistent context allocations.
// The host, expert and compute limits must be nonzero.
// max_compute_bytes bounds scheduler buffers, not backend-private kernel workspaces.
struct llama_layered_prefill_group {
    uint32_t first_layer = 0;
    uint32_t n_layers = 1;
    uint32_t n_prefetch_layers = 0;
    size_t expert_bytes = 0;
    size_t compute_bytes = 0;
    size_t device_compute_bytes = 0;
    double dispatch_ms = 0;
};

struct llama_layered_prefill_params {
    uint32_t n_ubatch = 256;
    ggml_type staging_type = GGML_TYPE_BF16;
    size_t max_host_bytes = size_t(16) << 30; // one suffix activation buffer and pinned weight staging
    size_t max_expert_bytes = size_t(6) << 30;
    size_t max_compute_bytes = size_t(8) << 30;
    bool require_gpu = true; // false enables CPU correctness testing
    bool pipeline = true;
    size_t max_transfer_bytes = size_t(512) << 20; // activation device slots and pinned weight staging; full activations use max_host_bytes
    bool weight_prefetch = true;
    size_t max_prefetch_bytes = size_t(6) << 30; // additional next-layer device weights
    size_t prefetch_headroom_bytes = size_t(512) << 20; // resident-planning and prefetch safety margin
    uint32_t group_layers = 1;
    uint32_t max_prefetch_layers = 4;
    size_t max_group_expert_bytes = 0; // zero: no extra group limit beyond device capacity
    bool auto_plan = false;
    uint32_t max_group_layers = 4;
    uint32_t max_calibration_ms = 2000; // soft admission limit; one probe may finish after it
    bool reuse_prefix = false; // tokens are only the suffix; reuse the context's complete sequence-0 state
    uint32_t expert_slots = 0; // zero: complete banks; nonzero: route-driven single-layer cache
    bool stream_weights = false; // Qwen: pack and temporarily upload all non-expert layer weights
    size_t max_static_bytes = size_t(256) << 20; // current and prefetched non-expert banks
    size_t max_device_bytes = 0; // optional state + temporary weights + activation slots + device scheduler cap
    bool profile_gpu = false; // CUDA profiler capture range and NVTX labels; use an external activity collector
    bool merge_gate_up = false; // pack compatible streamed gate/up experts for one matrix product
    bool defer_static_upload = false; // prepare next-layer static weights before compute, submit DMA afterwards
};

struct llama_layered_prefill_stats {
    uint64_t expert_copy_bytes = 0;
    uint64_t activation_read_bytes = 0;
    uint64_t activation_write_bytes = 0;
    size_t activation_host_bytes = 0;
    size_t activation_pinned_bytes = 0;
    uint32_t expert_layer_loads = 0;
    uint32_t microbatches = 0;
    size_t peak_compute_bytes = 0;
    size_t transfer_buffer_bytes = 0;
    uint32_t input_prefetches = 0;
    uint32_t async_output_batches = 0;
    bool pipeline_active = false;
    uint32_t weight_prefetches = 0;
    uint32_t weight_prefetch_skips = 0;
    uint32_t weight_prefetch_allocation_attempts = 0;
    uint64_t prefetched_expert_bytes = 0;
    uint64_t weight_prefetch_submit_bytes = 0;
    size_t peak_prefetch_bytes = 0;
    size_t prefetch_pinned_bytes = 0;
    uint32_t selected_ubatch = 0;
    std::vector<llama_layered_prefill_group> groups;
    uint32_t planner_candidates = 0;
    uint32_t calibration_probes = 0;
    bool profile_reused = false;
    double calibration_ms = 0;
    double planning_ms = 0;
    double estimated_ms = 0;
    uint32_t reused_prefix_tokens = 0;
    uint32_t evaluated_tokens = 0;
    bool capacity_only = false;
    size_t resident_model_bytes = 0;
    size_t resident_state_bytes = 0;
    size_t device_free_bytes = 0;
    size_t minimum_additional_bytes = 0;
    size_t capacity_shortfall_bytes = 0;
    bool shortfall_is_lower_bound = false;
    size_t budget_shortfall_bytes = 0;
    bool prefix_invalidated = false;
    uint64_t expert_cache_hits = 0;
    uint64_t expert_cache_misses = 0;
    size_t peak_expert_bytes = 0;
    uint64_t demand_async_bytes = 0;
    uint32_t demand_overlap_batches = 0;
    size_t demand_pinned_bytes = 0;
    uint64_t static_copy_bytes = 0;
    uint32_t static_layer_loads = 0;
    uint32_t static_prefetches = 0;
    uint32_t static_prefetch_skips = 0;
    size_t peak_static_bytes = 0;
    double weight_load_ms = 0;
    double graph_setup_ms = 0;
    double compute_wait_ms = 0;
    double activation_copy_ms = 0;
    uint32_t expert_allocations = 0;
    uint32_t expert_reuses = 0;
    size_t peak_device_bytes = 0;
    uint32_t merged_expert_layers = 0;
};

// Allocate persistent state and scheduler metadata, without full-model compute buffers.
LLAMA_API llama_context * llama_init_from_model_layered(llama_model * model, llama_context_params params);

// Returns 0 on success, -1 on unsupported/invalid input, -2 on allocation/compute failure.
// With reuse_prefix, tokens contain only the consecutive suffix of the live sequence-0 state.
// An execution failure invalidates the context; admission/planning failures preserve an existing prefix.
LLAMA_API int32_t llama_prefill_layered(
        struct llama_context * ctx,
        const llama_token * tokens,
        int32_t n_tokens,
        const llama_layered_prefill_params & params,
        llama_layered_prefill_stats * stats = nullptr);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
