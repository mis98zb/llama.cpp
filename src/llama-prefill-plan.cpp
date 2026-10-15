#include "llama-prefill-plan.h"

#include <algorithm>
#include <cmath>

static double transfer_ms(double bytes, double bandwidth) {
    if (bytes == 0) {
        return 0;
    }
    if (!std::isfinite(bandwidth) || bandwidth <= 0) {
        throw std::runtime_error("missing measured prefill transfer bandwidth");
    }
    return bytes / bandwidth;
}

double llama_prefill_estimate(
        const llama_prefill_cost & cost,
        const std::vector<llama_layered_prefill_group> & groups,
        const std::vector<uint32_t> & families,
        const std::vector<size_t> & weights,
        uint32_t tokens, uint32_t ubatch, size_t row_bytes, bool pipeline,
        std::vector<double> * group_times, uint32_t prefix_tokens) {
    if (tokens == 0 || ubatch == 0 || row_bytes == 0 || groups.empty() || families.size() != weights.size()) {
        throw std::runtime_error("invalid joint prefill cost input");
    }
    if (!std::isfinite(cost.replay_ms) || cost.replay_ms < 0) {
        throw std::runtime_error("invalid measured metadata replay time");
    }
    const uint32_t batches = tokens / ubatch + (tokens % ubatch != 0);
    const uint32_t slices = std::min(batches, 256u);
    if (group_times) {
        group_times->clear();
    }
    double total = 0;
    double incoming = 0;
    uint32_t expected_layer = 0;
    for (size_t g = 0; g < groups.size(); ++g) {
        const auto & group = groups[g];
        const double before = total;
        if (!std::isfinite(group.dispatch_ms) || group.dispatch_ms < 0) {
            throw std::runtime_error("invalid measured prefill dispatch time");
        }
        if (group.first_layer != expected_layer || group.n_layers == 0 ||
            group.n_layers > weights.size() - expected_layer) {
            throw std::runtime_error("nonconsecutive joint prefill groups");
        }
        expected_layer += group.n_layers;
        double weight_bytes = 0;
        for (uint32_t il = group.first_layer; il < expected_layer; ++il) {
            weight_bytes += weights[il];
        }
        total += cost.replay_ms + transfer_ms(std::max(0.0, weight_bytes - incoming), cost.weight_bytes_per_ms);
        double prefetched = 0;
        if (g + 1 < groups.size()) {
            if (groups[g + 1].first_layer != expected_layer ||
                group.n_prefetch_layers > groups[g + 1].n_layers ||
                group.n_prefetch_layers > weights.size() - expected_layer) {
                throw std::runtime_error("prefetch extends beyond the next compute group");
            }
            for (uint32_t i = 0; i < group.n_prefetch_layers; ++i) {
                prefetched += weights[expected_layer + i];
            }
        } else if (group.n_prefetch_layers != 0) {
            throw std::runtime_error("last compute group cannot prefetch");
        }
        double host = 0, compute = 0, upload = 0, download = 0;
        std::array<double, 2> output_ready = {};
        const uint32_t first_count = std::min<uint64_t>(tokens, uint64_t(batches / slices) * ubatch);
        double input_ready = pipeline && group.first_layer ? transfer_ms(first_count * double(row_bytes),
                                                                        cost.read_bytes_per_ms) : 0;
        upload = input_ready;
        for (uint32_t b = 0, offset = 0; b < slices; ++b) {
            const uint32_t next_batch = uint64_t(batches) * (b + 1) / slices;
            const uint32_t previous_batch = uint64_t(batches) * b / slices;
            const uint32_t step_count = next_batch - previous_batch;
            const uint32_t end = std::min<uint64_t>(tokens, uint64_t(next_batch) * ubatch);
            const uint32_t count = end - offset;
            const double prefix = std::max(double(cost.min_prefix), prefix_tokens + (double(offset) + end) / 2);
            double work = 0, setup = 0;
            for (uint32_t il = group.first_layer; il < expected_layer; ++il) {
                const auto f = families[il];
                if (f >= cost.measured.size() || !cost.measured[f]) {
                    throw std::runtime_error("missing measured prefill layer family");
                }
                if (!std::isfinite(cost.compute_ms[f]) || cost.compute_ms[f] < 0 ||
                    !std::isfinite(cost.compute_prefix_ms[f]) || cost.compute_prefix_ms[f] < 0 ||
                    !std::isfinite(cost.setup_ms[f]) || cost.setup_ms[f] < 0 ||
                    !std::isfinite(cost.setup_prefix_ms[f]) || cost.setup_prefix_ms[f] < 0) {
                    throw std::runtime_error("invalid measured prefill layer timing");
                }
                work += std::max(0.001, cost.compute_ms[f] + cost.compute_prefix_ms[f] * prefix);
                setup = std::max(setup, cost.setup_ms[f] + cost.setup_prefix_ms[f] * prefix);
            }
            work *= double(count) / ubatch;
            host = std::max(host, pipeline ? output_ready[b % 2] : download) +
                (setup + group.dispatch_ms) * step_count;
            if (!pipeline && group.first_layer) {
                input_ready = std::max(host, upload) + transfer_ms(count * double(row_bytes), cost.read_bytes_per_ms);
                upload = input_ready;
            }
            const double begin = std::max({host, compute, input_ready});
            compute = begin + work;
            offset += count;
            if (pipeline && group.first_layer && offset < tokens) {
                const uint32_t following_batch = uint64_t(batches) * (b + 2) / slices;
                const uint32_t following_end = std::min<uint64_t>(tokens, uint64_t(following_batch) * ubatch);
                input_ready = std::max(upload, begin) +
                    transfer_ms((following_end - offset) * double(row_bytes), cost.read_bytes_per_ms);
                upload = input_ready;
            }
            if (prefetched != 0) {
                upload = std::max(upload, begin) +
                    transfer_ms(prefetched * step_count / batches, cost.weight_bytes_per_ms);
            }
            download = std::max(download, compute) + transfer_ms(count * double(row_bytes), cost.write_bytes_per_ms);
            output_ready[b % 2] = download;
            host = pipeline ? compute : download;
        }
        total += std::max({host, compute, upload, download});
        if (group_times) {
            group_times->push_back(total - before);
        }
        incoming = prefetched;
    }
    if (expected_layer != weights.size() || !std::isfinite(total)) {
        throw std::runtime_error("incomplete or invalid joint prefill cost");
    }
    return total;
}
