#pragma once

#include "llama-ext.h"

#include <array>
#include <map>
#include <stdexcept>
#include <vector>

struct llama_prefill_resource_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct llama_prefill_probe_point {
    uint32_t layer;
    uint32_t tokens;
    uint32_t prefix;
    double compute_ms;
    double setup_ms;
};

struct llama_prefill_probe {
    std::vector<llama_prefill_probe_point> points;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    uint64_t weight_bytes = 0;
    double read_ms = 0;
    double write_ms = 0;
    double weight_ms = 0;
    uint32_t prefetch_steps = 0;
};

struct llama_prefill_cost {
    uint32_t sample_tokens = 0;
    uint32_t min_prefix = 0;
    std::array<double, 8> compute_ms = {};
    std::array<double, 8> compute_prefix_ms = {};
    std::array<double, 8> setup_ms = {};
    std::array<double, 8> setup_prefix_ms = {};
    std::array<bool, 8> measured = {};
    double read_bytes_per_ms = 0;
    double write_bytes_per_ms = 0;
    double weight_bytes_per_ms = 0;
    double replay_ms = 0;
};

struct llama_prefill_profile {
    ggml_type staging_type = GGML_TYPE_COUNT;
    uint32_t threads = 0;
    uint32_t ratio_tokens = 0;
    std::map<uint32_t, llama_prefill_cost> costs;
    std::map<std::vector<uint32_t>, double> compute_ratios;
};

double llama_prefill_estimate(
        const llama_prefill_cost & cost,
        const std::vector<llama_layered_prefill_group> & groups,
        const std::vector<uint32_t> & families,
        const std::vector<size_t> & weights,
        uint32_t tokens, uint32_t ubatch, size_t row_bytes, bool pipeline,
        std::vector<double> * group_times = nullptr, uint32_t prefix_tokens = 0);
