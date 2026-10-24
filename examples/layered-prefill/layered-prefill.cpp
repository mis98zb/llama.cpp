#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

int main(int argc, char ** argv) {
    llama_layered_prefill_params pp;
    uint32_t prefix_tokens = 0;
    uint32_t prompt_tokens = 0;
    std::string logits_file;
    std::vector<uint32_t> ubatch_sweep;
    bool compare_optimizations = false;
    std::vector<char *> common_args = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--prompt") == 0 ||
             std::strcmp(argv[i], "-f") == 0 || std::strcmp(argv[i], "--file") == 0) && i + 1 < argc) {
            common_args.push_back(argv[i]);
            common_args.push_back(argv[++i]);
        } else if (std::strcmp(argv[i], "--logits-file") == 0) {
            if (++i == argc) {
                std::fprintf(stderr, "--logits-file requires a path\n");
                return 1;
            }
            logits_file = argv[i];
        } else if (std::strcmp(argv[i], "--sweep-ubatch") == 0) {
            if (++i == argc) {
                std::fprintf(stderr, "--sweep-ubatch requires comma-separated positive counts\n");
                return 1;
            }
            try {
                const std::string text = argv[i];
                if (text.empty() || text.back() == ',') {
                    throw std::invalid_argument("empty microbatch count");
                }
                std::istringstream input(text);
                std::string part;
                while (std::getline(input, part, ',')) {
                    if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos) {
                        throw std::invalid_argument("invalid microbatch count");
                    }
                    const auto count = std::stoull(part);
                    if (!count || count > UINT32_MAX || ubatch_sweep.size() >= 32) {
                        throw std::invalid_argument("microbatch count or sweep length out of range");
                    }
                    ubatch_sweep.push_back(uint32_t(count));
                }
            } catch (const std::exception & e) {
                std::fprintf(stderr, "--sweep-ubatch: %s\n", e.what());
                return 1;
            }
        } else if (std::strcmp(argv[i], "--staging-type") == 0) {
            if (++i == argc || (std::strcmp(argv[i], "bf16") != 0 && std::strcmp(argv[i], "f32") != 0)) {
                std::fprintf(stderr, "--staging-type requires bf16 or f32\n");
                return 1;
            }
            pp.staging_type = std::strcmp(argv[i], "bf16") == 0 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        } else if (std::strcmp(argv[i], "--no-pipeline") == 0) {
            pp.pipeline = false;
        } else if (std::strcmp(argv[i], "--no-weight-prefetch") == 0) {
            pp.weight_prefetch = false;
        } else if (std::strcmp(argv[i], "--auto-plan") == 0) {
            pp.auto_plan = true;
        } else if (std::strcmp(argv[i], "--stream-weights") == 0) {
            pp.stream_weights = true;
        } else if (std::strcmp(argv[i], "--profile-gpu") == 0) {
            pp.profile_gpu = true;
        } else if (std::strcmp(argv[i], "--merge-gate-up") == 0) {
            pp.merge_gate_up = true;
        } else if (std::strcmp(argv[i], "--defer-static-upload") == 0) {
            pp.defer_static_upload = true;
        } else if (std::strcmp(argv[i], "--compare-optimizations") == 0) {
            compare_optimizations = true;
        } else if (std::strcmp(argv[i], "--host-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--compute-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--prefetch-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--prefetch-headroom-mib") == 0 ||
                   std::strcmp(argv[i], "--device-headroom-mib") == 0 ||
                   std::strcmp(argv[i], "--group-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--group-layers") == 0 ||
                   std::strcmp(argv[i], "--max-group-layers") == 0 ||
                   std::strcmp(argv[i], "--max-prefetch-layers") == 0 ||
                   std::strcmp(argv[i], "--expert-slots") == 0 ||
                   std::strcmp(argv[i], "--static-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--transfer-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--device-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--prompt-tokens") == 0 ||
                   std::strcmp(argv[i], "--prefix-tokens") == 0 ||
                   std::strcmp(argv[i], "--calibration-ms") == 0) {
            const char * option = argv[i];
            const bool mib = std::strstr(option, "-mib") != nullptr;
            if (++i == argc) {
                std::fprintf(stderr, "%s requires a nonnegative count\n", option);
                return 1;
            }
            try {
                size_t end = 0;
                const auto value = std::stoull(argv[i], &end);
                if (argv[i][0] == '-' || end != std::strlen(argv[i]) ||
                    value > (mib ? std::numeric_limits<size_t>::max() >> 20 : UINT32_MAX)) {
                    throw std::invalid_argument("invalid count");
                }
                const size_t bytes = mib ? size_t(value) << 20 : size_t(value);
                if (std::strcmp(option, "--host-budget-mib") == 0) {
                    pp.max_host_bytes = bytes;
                } else if (std::strcmp(option, "--compute-budget-mib") == 0) {
                    pp.max_compute_bytes = bytes;
                } else if (std::strcmp(option, "--prefetch-budget-mib") == 0) {
                    pp.max_prefetch_bytes = bytes;
                } else if (std::strcmp(option, "--prefetch-headroom-mib") == 0 ||
                           std::strcmp(option, "--device-headroom-mib") == 0) {
                    pp.prefetch_headroom_bytes = bytes;
                } else if (std::strcmp(option, "--group-budget-mib") == 0) {
                    pp.max_group_expert_bytes = bytes;
                } else if (std::strcmp(option, "--group-layers") == 0) {
                    pp.group_layers = bytes;
                } else if (std::strcmp(option, "--max-group-layers") == 0) {
                    pp.max_group_layers = bytes;
                } else if (std::strcmp(option, "--max-prefetch-layers") == 0) {
                    pp.max_prefetch_layers = bytes;
                } else if (std::strcmp(option, "--prefix-tokens") == 0) {
                    prefix_tokens = bytes;
                } else if (std::strcmp(option, "--expert-slots") == 0) {
                    pp.expert_slots = bytes;
                } else if (std::strcmp(option, "--static-budget-mib") == 0) {
                    pp.max_static_bytes = bytes;
                } else if (std::strcmp(option, "--transfer-budget-mib") == 0) {
                    pp.max_transfer_bytes = bytes;
                } else if (std::strcmp(option, "--device-budget-mib") == 0) {
                    pp.max_device_bytes = bytes;
                } else if (std::strcmp(option, "--prompt-tokens") == 0) {
                    prompt_tokens = bytes;
                } else {
                    pp.max_calibration_ms = bytes;
                }
            } catch (const std::exception & e) {
                std::fprintf(stderr, "%s: %s\n", option, e.what());
                return 1;
            }
        } else {
            common_args.push_back(argv[i]);
        }
    }
    common_params params;
    params.n_parallel = 1;
    params.n_ubatch = 256;
    params.n_predict = 32;
    common_init();
    const int common_argc = common_args.size();
    common_args.push_back(nullptr);
    if (!common_params_parse(common_argc, common_args.data(), params, LLAMA_EXAMPLE_COMMON,
                            [](int, char **) {
                                LOG("Layered prefill: --staging-type bf16|f32, --no-pipeline, --no-weight-prefetch\n");
                                LOG("Memory: --compute-budget-mib N, --prefetch-budget-mib N, --prefetch-headroom-mib N\n");
                                LOG("Host activation and pinned weight staging limit: --host-budget-mib N\n");
                                LOG("Planner: --auto-plan, --group-layers N, --max-group-layers N, --max-prefetch-layers N\n");
                                LOG("Planner limits: --group-budget-mib N, --calibration-ms N\n");
                                LOG("Resident safety margin: --device-headroom-mib N (alias of --prefetch-headroom-mib)\n");
                                LOG("Prefix demonstration: --prefix-tokens N (seed and reuse in this process)\n");
                                LOG("Qwen3.5 route-driven cache: --expert-slots N (single-layer groups, no speculative prefetch)\n");
                                LOG("Correctness comparison: --logits-file PATH (final-token F32 logits)\n");
                                LOG("All layer weights on CPU: --stream-weights, --static-budget-mib N, --transfer-budget-mib N\n");
                                LOG("Accounted total GPU working set: --device-budget-mib N\n");
                                LOG("Long-prompt benchmark: --prompt-tokens N (requires at least N input tokens)\n");
                                LOG("Reuse host model across contexts: --sweep-ubatch 128,256,512,1024,2048 (-n 0)\n");
                                LOG("Nsight CUDA activity capture: --profile-gpu (one manual prefill, no sweep/prefix)\n");
                                LOG("Streamed complete-bank experiments: --merge-gate-up, --defer-static-upload\n");
                                LOG("Compare all four experiments twice with one host model: --compare-optimizations (-n 0)\n");
                                LOG("CUDA streamed complete banks automatically reuse weight staging and upload admitted expert prefetches on a worker\n");
                            })) {
        return 1;
    }
    if (params.n_parallel != 1 || params.n_predict < 0 || params.moe_cache_size != 0 ||
        params.embedding || !params.lora_adapters.empty()) {
        LOG_ERR("layered prefill requires one sequence, finite generation, no adapters and no decode expert cache\n");
        return 1;
    }
    if (!ubatch_sweep.empty() && (params.n_predict != 0 || prefix_tokens)) {
        LOG_ERR("--sweep-ubatch requires -n 0 and no prefix demonstration\n");
        return 1;
    }
    if (pp.profile_gpu && (!ubatch_sweep.empty() || prefix_tokens || pp.auto_plan)) {
        LOG_ERR("--profile-gpu requires one manual prefill without a sweep or prefix demonstration\n");
        return 1;
    }
    if (compare_optimizations) {
        if (!pp.stream_weights || pp.expert_slots || pp.auto_plan || pp.profile_gpu ||
            prefix_tokens || params.n_predict != 0 || !ubatch_sweep.empty()) {
            LOG_ERR("--compare-optimizations requires manual streamed complete banks, -n 0 and no other sweep/profile/prefix\n");
            return 1;
        }
        ubatch_sweep.assign(8, params.n_ubatch);
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    int exit_status = 0;
    try {
        auto mp = common_model_params_to_llama(params);
        std::vector<llama_model_tensor_buft_override> overrides = {
            { pp.stream_weights ? ".*" : LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        };
        if (pp.stream_weights) {
            mp.n_gpu_layers = INT32_MAX;
            LOG_INF("all model weights stay in CPU buffers; logical GPU layers place KV/recurrent state only\n");
        }
        if (mp.tensor_buft_overrides) {
            for (auto * o = mp.tensor_buft_overrides; o->pattern; ++o) {
                overrides.push_back(*o);
            }
        }
        overrides.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = overrides.data();
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mp));
        if (!model) {
            throw std::runtime_error("failed to load model");
        }
        auto tokens = common_tokenize(llama_model_get_vocab(model.get()), params.prompt, true);
        if (prompt_tokens) {
            if (tokens.size() < prompt_tokens) {
                throw std::runtime_error("input is shorter than --prompt-tokens; provide a longer prompt");
            }
            tokens.resize(prompt_tokens);
        }
        if (tokens.empty() || tokens.size() > (size_t) INT32_MAX - params.n_predict) {
            throw std::runtime_error("invalid prompt length");
        }
        const bool sweeping = !ubatch_sweep.empty();
        if (!sweeping) {
            ubatch_sweep.push_back(params.n_ubatch);
        }
        for (size_t point = 0; point < ubatch_sweep.size(); ++point) {
            const uint32_t benchmark_ubatch = ubatch_sweep[point];
            if (compare_optimizations) {
                const unsigned mask = unsigned(point < 4 ? point : 7 - point);
                pp.merge_gate_up = (mask & 1) != 0;
                pp.defer_static_upload = (mask & 2) != 0;
                LOG_INF("comparison case=%zu, merged=%d, deferred=%d\n", point, pp.merge_gate_up, pp.defer_static_upload);
            }
            if (sweeping) {
                LOG_INF("sweep ubatch=%u begin\n", benchmark_ubatch);
            }
            try {
                auto cp = common_context_params_to_llama(params);
                cp.n_ubatch = benchmark_ubatch;
                cp.n_batch = std::max<uint32_t>(cp.n_batch, tokens.size());
                if (cp.n_ctx == 0) {
                    cp.n_ctx = tokens.size() + params.n_predict;
                }
                if (tokens.size() + params.n_predict > cp.n_ctx) {
                    throw std::runtime_error("context cannot hold prompt and requested generation");
                }
                llama_context_ptr ctx(pp.stream_weights ?
                    llama_init_from_model_layered(model.get(), cp) : llama_init_from_model(model.get(), cp));
                if (!ctx) {
                    throw std::runtime_error("failed to create the resident context before planning; KV/state eviction is not implemented");
                }
                pp.n_ubatch = cp.n_ubatch;
                if (pp.expert_slots && pp.weight_prefetch) {
                    LOG_INF("route-driven mode waits for routing; routed-bank prefetch is disabled, static prefetch remains budgeted\n");
                }
                if (pp.staging_type == GGML_TYPE_BF16) {
                    LOG_WRN("BF16 staging rounds residuals and may change sparse selection; use --staging-type f32 as a reference\n");
                }
                llama_layered_prefill_stats stats;
                if (prefix_tokens >= tokens.size()) {
                    throw std::runtime_error("--prefix-tokens must be smaller than the tokenized prompt");
                }
                if (prefix_tokens) {
                    if (llama_prefill_layered(ctx.get(), tokens.data(), prefix_tokens, pp) != 0) {
                        throw std::runtime_error("failed to seed the resident prefix");
                    }
                    pp.reuse_prefix = true;
                }
                const int64_t prefill_start = ggml_time_us();
                if (llama_prefill_layered(ctx.get(), tokens.data() + prefix_tokens, tokens.size() - prefix_tokens, pp, &stats) != 0) {
                    LOG_ERR("resident model=%zu state=%zu free snapshot=%zu extra requirement=%zu capacity gap=%zu policy gap=%zu prefix invalidated=%d\n",
                            stats.resident_model_bytes, stats.resident_state_bytes, stats.device_free_bytes,
                            stats.minimum_additional_bytes, stats.capacity_shortfall_bytes, stats.budget_shortfall_bytes,
                            stats.prefix_invalidated);
                    throw std::runtime_error("layered prefill failed; no fallback was used");
                }
                const double prefill_ms = (ggml_time_us() - prefill_start) * 1e-3;
                LOG_INF("prefill wall=%.3f ms, tokens=%u, throughput=%.3f tokens/s\n",
                        prefill_ms, stats.evaluated_tokens, stats.evaluated_tokens * 1000.0 / prefill_ms);
                LOG_INF("phase ms: weight-load=%.3f, graph-setup=%.3f, compute-wait=%.3f, activation-copy=%.3f\n",
                        stats.weight_load_ms, stats.graph_setup_ms, stats.compute_wait_ms, stats.activation_copy_ms);
                LOG_INF("reused prefix=%u tokens, evaluated suffix=%u tokens, capacity-only plan=%d\n",
                        stats.reused_prefix_tokens, stats.evaluated_tokens, stats.capacity_only);
                LOG_INF("single in-place activation buffer: host bytes=%zu, pinned bytes=%zu\n",
                        stats.activation_host_bytes, stats.activation_pinned_bytes);
                LOG_INF("expert layer loads=%u, copied bytes=%" PRIu64 ", activation read/write=%" PRIu64 "/%" PRIu64
                        ", peak measured compute bytes=%zu\n", stats.expert_layer_loads, stats.expert_copy_bytes,
                        stats.activation_read_bytes, stats.activation_write_bytes, stats.peak_compute_bytes);
                LOG_INF("expert slots=%u, hits=%" PRIu64 ", misses=%" PRIu64 ", peak bank bytes=%zu\n",
                        pp.expert_slots, stats.expert_cache_hits, stats.expert_cache_misses, stats.peak_expert_bytes);
                LOG_INF("expert device allocations=%u, reuses=%u\n", stats.expert_allocations, stats.expert_reuses);
                LOG_INF("merged expert layers=%u, deferred static upload=%d\n",
                        stats.merged_expert_layers, pp.defer_static_upload);
                LOG_INF("weight staging allocations=%u, reuses=%u, peak pinned bytes=%zu, worker bytes=%" PRIu64 "\n",
                        stats.weight_staging_allocations, stats.weight_staging_reuses,
                        stats.peak_weight_staging_bytes, stats.weight_worker_bytes);
                LOG_INF("direct expert upload bytes=%" PRIu64 "\n", stats.direct_expert_bytes);
                LOG_INF("device state bytes=%zu, peak accounted device bytes=%zu\n",
                        stats.resident_state_bytes, stats.peak_device_bytes);
                LOG_INF("streamed weights=%d, static layer loads=%u, static bytes=%" PRIu64
                        ", static prefetches=%u, skipped=%u, peak static bank bytes=%zu\n",
                        pp.stream_weights, stats.static_layer_loads, stats.static_copy_bytes,
                        stats.static_prefetches, stats.static_prefetch_skips, stats.peak_static_bytes);
                LOG_INF("demand asynchronous bytes=%" PRIu64 ", shared-compute overlap batches=%u, pinned bytes=%zu\n",
                        stats.demand_async_bytes, stats.demand_overlap_batches, stats.demand_pinned_bytes);
                LOG_INF("pipeline=%d, transfer buffer bytes=%zu, input prefetches=%u, asynchronous output batches=%u\n",
                        stats.pipeline_active, stats.transfer_buffer_bytes, stats.input_prefetches, stats.async_output_batches);
                LOG_INF("weight prefetches=%u, skipped=%u, allocation attempts=%u, prefetched bytes=%" PRIu64
                        ", submitted during compute=%" PRIu64 ", peak device bytes=%zu, pinned bytes=%zu\n",
                        stats.weight_prefetches, stats.weight_prefetch_skips, stats.weight_prefetch_allocation_attempts,
                        stats.prefetched_expert_bytes, stats.weight_prefetch_submit_bytes,
                        stats.peak_prefetch_bytes, stats.prefetch_pinned_bytes);
                LOG_INF("selected batch=%u, groups=%zu, planner alternatives=%u, probes=%u, profile reused=%d, planning=%.2f ms\n",
                        stats.selected_ubatch, stats.groups.size(), stats.planner_candidates,
                        stats.calibration_probes, stats.profile_reused, stats.planning_ms);
                for (const auto & group : stats.groups) {
                    LOG_INF("layers [%u,%u): prefetch=%u, experts=%zu, scheduler=%zu\n",
                            group.first_layer, group.first_layer + group.n_layers, group.n_prefetch_layers,
                            group.expert_bytes, group.compute_bytes);
                }
                llama_perf_context_print(ctx.get());
                const auto * vocab = llama_model_get_vocab(model.get());
                const size_t nv = llama_vocab_n_tokens(vocab);
                if (!logits_file.empty()) {
                    const auto path = logits_file + (compare_optimizations ? ".case" + std::to_string(point) : "") +
                        (sweeping ? ".ub" + std::to_string(benchmark_ubatch) : "");
                    auto * file = std::fopen(path.c_str(), "wb");
                    if (!file) {
                        throw std::runtime_error("failed to open logits output file");
                    }
                    const size_t written = std::fwrite(llama_get_logits_ith(ctx.get(), -1), sizeof(float), nv, file);
                    const int closed = std::fclose(file);
                    if (written != nv || closed != 0) {
                        throw std::runtime_error("failed to write logits output file");
                    }
                }
                common_batch batch(ctx.get());
                for (int32_t i = 0; i < params.n_predict; ++i) {
                    const auto * logits = llama_get_logits_ith(ctx.get(), -1);
                    const llama_token token = std::max_element(logits, logits + nv) - logits;
                    if (llama_vocab_is_eog(vocab, token)) {
                        break;
                    }
                    const auto piece = common_token_to_piece(vocab, token);
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    std::fflush(stdout);
                    batch.clear();
                    batch.add(token, tokens.size() + i, 0, true);
                    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        throw std::runtime_error("ordinary decode failed after layered prefill");
                    }
                }
                std::putchar('\n');
                if (sweeping) {
                    LOG_INF("sweep ubatch=%u end status=0\n", benchmark_ubatch);
                }
            } catch (const std::exception & e) {
                if (!sweeping) {
                    throw;
                }
                LOG_ERR("sweep ubatch=%u failed: %s\n", benchmark_ubatch, e.what());
                LOG_INF("sweep ubatch=%u end status=1\n", benchmark_ubatch);
                exit_status = 2;
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        return 1;
    }
    llama_backend_free();
    return exit_status;
}
