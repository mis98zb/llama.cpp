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
#include <stdexcept>

int main(int argc, char ** argv) {
    llama_layered_prefill_params pp;
    std::vector<char *> common_args = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--staging-type") == 0) {
            if (++i == argc || (std::strcmp(argv[i], "bf16") != 0 && std::strcmp(argv[i], "f32") != 0)) {
                std::fprintf(stderr, "--staging-type requires bf16 or f32\n");
                return 1;
            }
            pp.staging_type = std::strcmp(argv[i], "bf16") == 0 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        } else if (std::strcmp(argv[i], "--no-pipeline") == 0) {
            pp.pipeline = false;
        } else if (std::strcmp(argv[i], "--no-weight-prefetch") == 0) {
            pp.weight_prefetch = false;
        } else if (std::strcmp(argv[i], "--compute-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--prefetch-budget-mib") == 0 ||
                   std::strcmp(argv[i], "--prefetch-headroom-mib") == 0) {
            const char * option = argv[i];
            if (++i == argc) {
                std::fprintf(stderr, "%s requires a nonnegative MiB count\n", option);
                return 1;
            }
            try {
                size_t end = 0;
                const auto value = std::stoull(argv[i], &end);
                if (argv[i][0] == '-' || end != std::strlen(argv[i]) ||
                    value > (std::numeric_limits<size_t>::max() >> 20)) {
                    throw std::invalid_argument("invalid MiB count");
                }
                const size_t bytes = size_t(value) << 20;
                if (std::strcmp(option, "--compute-budget-mib") == 0) {
                    pp.max_compute_bytes = bytes;
                } else if (std::strcmp(option, "--prefetch-budget-mib") == 0) {
                    pp.max_prefetch_bytes = bytes;
                } else {
                    pp.prefetch_headroom_bytes = bytes;
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
                            })) {
        return 1;
    }
    if (params.n_parallel != 1 || params.n_predict < 0 || params.moe_cache_size != 0 ||
        params.embedding || !params.lora_adapters.empty()) {
        LOG_ERR("layered prefill requires one sequence, finite generation, no adapters and no decode expert cache\n");
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    try {
        auto mp = common_model_params_to_llama(params);
        std::vector<llama_model_tensor_buft_override> overrides = {
            { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        };
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
        if (tokens.empty() || tokens.size() > (size_t) INT32_MAX - params.n_predict) {
            throw std::runtime_error("invalid prompt length");
        }
        auto cp = common_context_params_to_llama(params);
        cp.n_batch = std::max<uint32_t>(cp.n_batch, tokens.size());
        if (cp.n_ctx == 0) {
            cp.n_ctx = tokens.size() + params.n_predict;
        }
        if (tokens.size() + params.n_predict > cp.n_ctx) {
            throw std::runtime_error("context cannot hold prompt and requested generation");
        }
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        if (!ctx) {
            throw std::runtime_error("failed to create context");
        }
        pp.n_ubatch = cp.n_ubatch;
        if (pp.staging_type == GGML_TYPE_BF16) {
            LOG_WRN("BF16 staging rounds residuals and may change sparse selection; use --staging-type f32 as a reference\n");
        }
        llama_layered_prefill_stats stats;
        if (llama_prefill_layered(ctx.get(), tokens.data(), tokens.size(), pp, &stats) != 0) {
            throw std::runtime_error("layered prefill failed; no fallback was used");
        }
        LOG_INF("expert layer loads=%u, copied bytes=%" PRIu64 ", activation read/write=%" PRIu64 "/%" PRIu64
                ", peak measured compute bytes=%zu\n", stats.expert_layer_loads, stats.expert_copy_bytes,
                stats.activation_read_bytes, stats.activation_write_bytes, stats.peak_compute_bytes);
        LOG_INF("pipeline=%d, transfer buffer bytes=%zu, input prefetches=%u, asynchronous output batches=%u\n",
                stats.pipeline_active, stats.transfer_buffer_bytes, stats.input_prefetches, stats.async_output_batches);
        LOG_INF("weight prefetches=%u, skipped=%u, allocation attempts=%u, prefetched bytes=%" PRIu64
                ", submitted during compute=%" PRIu64 ", peak device bytes=%zu, pinned bytes=%zu\n",
                stats.weight_prefetches, stats.weight_prefetch_skips, stats.weight_prefetch_allocation_attempts,
                stats.prefetched_expert_bytes, stats.weight_prefetch_submit_bytes,
                stats.peak_prefetch_bytes, stats.prefetch_pinned_bytes);
        llama_perf_context_print(ctx.get());
        const auto * vocab = llama_model_get_vocab(model.get());
        const size_t nv = llama_vocab_n_tokens(vocab);
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
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
