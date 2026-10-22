#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"
#include "../src/llama-ext.h"
#include "../src/llama-prefill-plan.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llm_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;
    size_t seed = params.seed;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, params.stdev);

    // TODO: refactor per-tensor initialization logic in a cleaner way

    // note: Mamba A must be negative (state decay)
    const bool is_ssm_a = strstr(tensor->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            float val = dis(gen);
            tmp[i] = is_ssm_a ? -fabsf(val) : val;
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            float val = dis(gen);
            tmp[i] = ggml_fp32_to_fp16(is_ssm_a ? -fabsf(val) : val);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_Q4_K) {
        std::vector<float> tmp(ne);
        for (auto & v : tmp) {
            v = dis(gen);
        }
        std::vector<uint8_t> quantized(ggml_nbytes(tensor));
        const size_t bytes = ggml_quantize_chunk(tensor->type, tmp.data(), quantized.data(),
                                                0, ne / tensor->ne[0], tensor->ne[0], nullptr);
        GGML_ASSERT(bytes == quantized.size());
        ggml_backend_tensor_set(tensor, quantized.data(), 0, bytes);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --layered-prefill        Test experimental GLM5-Next layer-major prefill\n");
    LOG("  --qwen-prefill           Test Qwen3.5 layer-major and route-driven prefill\n");
    LOG("  --layered-planner        Test grouped execution and joint planning\n");
    LOG("  --layered-prefix         Test resident hybrid prefix reuse\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_GLM5_NEXT) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_MAPLE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(host_experts_test(arch) ? 64 : 2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0,
        uint32_t ubatch_size = 64, ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_data = nullptr,
        ggml_type cache_k = GGML_TYPE_F16, int32_t gpu_layers = -1, bool defer_compute = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;
    if (gpu_layers >= 0) {
        model_params.n_gpu_layers = gpu_layers;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.cb_eval = cb_eval;
    ctx_params.cb_eval_user_data = cb_data;
    ctx_params.type_k = cache_k;
    ctx_params.type_v = cache_k;
    if (!encode) {
        ctx_params.n_ubatch = ubatch_size;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(defer_compute ?
        llama_init_from_model_layered(model.get(), ctx_params) : llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

static bool round_prefill_boundary(ggml_tensor * tensor, bool ask, void * data) {
    const bool selected = *(const bool *) data && strncmp(tensor->name, "l_out-", 6) == 0 &&
                          strcmp(tensor->name, "l_out-3") != 0;
    if (ask) {
        return selected;
    }
    if (!selected) {
        return true;
    }
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    for (auto & value : values) {
        value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    return true;
}

static gguf_context_ptr layered_metadata(bool quantized) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    llama_model_saver ms(LLM_ARCH_GLM5_NEXT, metadata.get());
    ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, std::vector<uint32_t>{0, 1, 0, 1});
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(32));
    if (quantized) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(256));
        ms.add_kv(LLM_KV_FEATURES_LENGTH, uint32_t(256));
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(256));
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(256));
        for (uint32_t il = 1; il < 4; ++il) {
            for (const char * part : {"gate", "up", "down"}) {
                ggml_tensor tensor = {};
                tensor.type = GGML_TYPE_Q4_K;
                tensor.ne[0] = tensor.ne[1] = 256;
                tensor.ne[2] = 2;
                tensor.ne[3] = 1;
                ggml_format_name(&tensor, "blk.%u.ffn_%s_exps.weight", il, part);
                gguf_add_tensor(metadata.get(), &tensor);
            }
        }
    }
    return metadata;
}

static int test_layered_prefill(size_t seed, float stdev, const char * target_backend,
                                bool quantized, bool pipeline, int prefetch_mode = 0, uint32_t group_layers = 1,
                                bool qwen = false, uint32_t slots = 0, bool cpu_non_experts = true,
                                bool stream_weights = false, ggml_type cache_type = GGML_TYPE_F16,
                                bool merge_gate_up = false, bool defer_static_upload = false) {
    const auto tokens = get_tokens(137, 128, seed);
    uint64_t expected_copies = 0;
    uint32_t n_cases = 0;
    std::vector<ggml_backend_dev_t> devices;
    if (target_backend) {
        devices.push_back(ggml_backend_dev_by_name(target_backend));
    }
    const llama_model_tensor_buft_override overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() }, { nullptr, nullptr },
    };
    const llama_model_tensor_buft_override streamed[] = {
        { ".*", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr },
    };
    for (uint32_t ubatch : qwen ? std::vector<uint32_t>{1, 3, 7} : std::vector<uint32_t>{7, 32, 64}) {
        for (ggml_type staging : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
            auto metadata = qwen ? get_gguf_ctx(LLM_ARCH_QWEN35MOE, true) : layered_metadata(quantized);
            if (qwen) {
                llama_model_saver ms(LLM_ARCH_QWEN35MOE, metadata.get());
                ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(4));
                ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(merge_gate_up ? 128 : 16));
                ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(256));
                if (quantized) {
                    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(256));
                    ms.add_kv(LLM_KV_FEATURES_LENGTH, uint32_t(256));
                    for (uint32_t il = 0; il < 4; ++il) {
                        for (const char * part : {"gate", "up", "down"}) {
                            ggml_tensor tensor = {};
                            tensor.type = GGML_TYPE_Q4_K;
                            tensor.ne[0] = tensor.ne[1] = 256;
                            tensor.ne[2] = merge_gate_up ? 128 : 16;
                            tensor.ne[3] = 1;
                            ggml_format_name(&tensor, "blk.%u.ffn_%s_exps.weight", il, part);
                            gguf_add_tensor(metadata.get(), &tensor);
                        }
                        if (merge_gate_up) {
                            ggml_tensor tensor = {};
                            tensor.type = GGML_TYPE_Q4_K;
                            tensor.ne[0] = 256;
                            tensor.ne[1] = 512;
                            tensor.ne[2] = 128;
                            tensor.ne[3] = 1;
                            ggml_format_name(&tensor, "blk.%u.ffn_gate_up_exps.weight", il);
                            gguf_add_tensor(metadata.get(), &tensor);
                        }
                    }
                }
            }
            bool rounding = staging == GGML_TYPE_BF16;
            ggml_context_ptr split_context;
            ggml_backend_buffer_ptr split_buffer;
            auto reference = get_model_and_ctx(metadata.get(), nullptr, seed, stdev, devices,
                                               LLAMA_SPLIT_MODE_LAYER, false, devices.empty() ? overrides : nullptr,
                                               0, ubatch, rounding ? round_prefill_boundary : nullptr, &rounding, cache_type);
            auto candidate = get_model_and_ctx(metadata.get(), nullptr, seed, stdev, devices,
                                               LLAMA_SPLIT_MODE_LAYER, false, stream_weights ? streamed : overrides, 0, ubatch,
                                               nullptr, nullptr, cache_type,
                                               stream_weights ? INT32_MAX : (qwen && cpu_non_experts ? 0 : -1),
                                               stream_weights);
            auto & layered = candidate.second;
            if (merge_gate_up) {
                split_context.reset(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
                if (!split_context) {
                    throw std::runtime_error("failed to allocate split gate/up fixture metadata");
                }
                std::vector<std::pair<ggml_tensor *, ggml_tensor *>> splits;
                for (auto & layer : candidate.first->layers) {
                    auto * source = layer.ffn_gate_up_exps;
                    if (!source || !ggml_backend_buffer_is_host(source->buffer)) {
                        throw std::runtime_error("split gate/up fixture requires host-resident merged weights");
                    }
                    auto * gate = ggml_new_tensor_3d(split_context.get(), source->type,
                        source->ne[0], source->ne[1] / 2, source->ne[2]);
                    auto * up = ggml_dup_tensor(split_context.get(), gate);
                    splits.emplace_back(gate, up);
                }
                split_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(split_context.get(), ggml_backend_cpu_buffer_type()));
                if (!split_buffer) {
                    throw std::runtime_error("failed to allocate split gate/up fixture weights");
                }
                for (size_t il = 0; il < splits.size(); ++il) {
                    auto & layer = candidate.first->layers[il];
                    const auto * source = layer.ffn_gate_up_exps;
                    const auto pair = splits[il];
                    const size_t bytes = pair.first->nb[2];
                    for (int64_t e = 0; e < source->ne[2]; ++e) {
                        const auto * data = static_cast<const uint8_t *>(source->data) + e * source->nb[2];
                        ggml_backend_tensor_set(pair.first, data, e * bytes, bytes);
                        ggml_backend_tensor_set(pair.second, data + bytes, e * bytes, bytes);
                    }
                    layer.ffn_gate_up_exps = nullptr;
                    layer.ffn_gate_exps = pair.first;
                    layer.ffn_up_exps = pair.second;
                }
            }
            const auto logits = get_logits(reference.first.get(), reference.second.get(), tokens);
            rounding = false;
            llama_layered_prefill_params pp;
            pp.n_ubatch = ubatch;
            pp.require_gpu = false;
            pp.staging_type = staging;
            pp.pipeline = pipeline;
            pp.group_layers = group_layers;
            pp.expert_slots = slots;
            pp.stream_weights = stream_weights;
            pp.merge_gate_up = merge_gate_up;
            pp.defer_static_upload = defer_static_upload;
            pp.weight_prefetch = prefetch_mode != 0;
            if (prefetch_mode) {
                pp.max_compute_bytes = size_t(64) << 20;
                pp.prefetch_headroom_bytes = prefetch_mode == 3 ? SIZE_MAX : size_t(64) << 20;
                pp.max_prefetch_bytes = prefetch_mode == 2 ? 1 : size_t(64) << 20;
            }
            llama_layered_prefill_stats stats;
            if (devices.empty()) {
                pp.profile_gpu = true;
                if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -1 ||
                    llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                    throw std::runtime_error("GPU profiling accepted CPU execution or changed its state");
                }
                pp.profile_gpu = false;
                for (bool merged : {false, true}) {
                    pp.merge_gate_up = merged;
                    pp.defer_static_upload = !merged;
                    if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -1 ||
                        llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                        throw std::runtime_error("streamed GPU optimization accepted CPU execution or changed state");
                    }
                }
                pp.merge_gate_up = merge_gate_up;
                pp.defer_static_upload = defer_static_upload;
            }
            const llama_token invalid = 128;
            if (llama_prefill_layered(layered.get(), &invalid, 1, pp) != -1) {
                throw std::runtime_error("layered prefill accepted an invalid token");
            }
            if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp, &stats) != 0) {
                throw std::runtime_error("layered prefill failed");
            }
            const size_t nv = llama_vocab_n_tokens(llama_model_get_vocab(reference.first.get()));
            std::vector<float> expected(logits.end() - nv, logits.end());
            std::vector<float> actual(llama_get_logits(layered.get()), llama_get_logits(layered.get()) + nv);
            const double tolerance = devices.empty() && staging == GGML_TYPE_F32 ? 1e-10 : 1e-5;
            if (!std::isfinite(nmse(expected, actual)) || nmse(expected, actual) > tolerance) {
                LOG_ERR("prefill mismatch: ubatch=%u staging=%s quantized=%d NMSE=%g tolerance=%g\n",
                        ubatch, ggml_type_name(staging), quantized, nmse(expected, actual), tolerance);
                throw std::runtime_error("layered prefill logits mismatch");
            }
            if (stats.expert_layer_loads != (qwen ? 4u : 3u) || stats.expert_copy_bytes == 0 ||
                (!slots && expected_copies && expected_copies != stats.expert_copy_bytes) ||
                stats.groups.size() != (4 + group_layers - 1) / group_layers ||
                stats.microbatches != stats.groups.size() * ((tokens.size() + ubatch - 1) / ubatch)) {
                throw std::runtime_error("experts were not loaded once per layer");
            }
            expected_copies = stats.expert_copy_bytes;
            if (stats.merged_expert_layers != (merge_gate_up ? 4u : 0u)) {
                throw std::runtime_error("streamed gate/up packing did not use the merged expert graph");
            }
            if (stream_weights && (stats.static_layer_loads != 4 || stats.static_copy_bytes == 0 ||
                                   stats.peak_static_bytes == 0)) {
                throw std::runtime_error("non-expert layer weights were not streamed once per layer");
            }
            if (slots && (stats.expert_cache_misses == 0 || stats.expert_cache_hits == 0 ||
                          stats.peak_expert_bytes == 0 ||
                          stats.expert_cache_hits + stats.expert_cache_misses > tokens.size() * 2 * 4)) {
                throw std::runtime_error("route-driven expert cache was not used");
            }
            const bool gpu = !devices.empty() && ggml_backend_dev_type(devices[0]) != GGML_BACKEND_DEVICE_TYPE_CPU;
            const size_t activation_bytes = tokens.size() * llama_model_n_embd(candidate.first.get()) *
                (qwen ? 1 : 4) * ggml_type_size(staging);
            if (stats.activation_host_bytes != activation_bytes ||
                stats.activation_pinned_bytes != (gpu ? activation_bytes : 0)) {
                throw std::runtime_error("activations did not use one suffix-sized host buffer");
            }
            if (stream_weights && gpu && (stats.expert_allocations > 2 || stats.expert_reuses < 2)) {
                throw std::runtime_error("streamed expert device banks were not reused");
            }
            if (stats.pipeline_active != (pipeline && gpu)) {
                throw std::runtime_error("layered prefill pipeline was not activated as requested");
            }
            if (slots && gpu && pipeline &&
                (stats.demand_async_bytes != stats.expert_copy_bytes || stats.demand_overlap_batches == 0 ||
                 stats.demand_pinned_bytes == 0)) {
                throw std::runtime_error("expert uploads did not use the shared-compute overlap path");
            }
            if (stats.pipeline_active &&
                (stats.transfer_buffer_bytes == 0 || stats.async_output_batches != stats.microbatches ||
                 stats.input_prefetches != (stats.groups.size() - 1) * ((tokens.size() + ubatch - 1) / ubatch - 1))) {
                throw std::runtime_error("layered prefill did not pipeline all activation transfers");
            }
            const uint64_t first_layer_bytes = qwen ? stats.expert_copy_bytes / 4 : 0;
            if (stream_weights && gpu && prefetch_mode == 1 &&
                (stats.static_prefetches != 3 || stats.static_prefetch_skips != 0)) {
                throw std::runtime_error("known next-layer static weights were not prefetched");
            }
            if (!stream_weights && gpu && prefetch_mode == 1 &&
                (stats.weight_prefetches != 3 || stats.weight_prefetch_skips != 0 ||
                 stats.weight_prefetch_allocation_attempts != 3 ||
                 stats.prefetched_expert_bytes != stats.expert_copy_bytes - first_layer_bytes ||
                 stats.weight_prefetch_submit_bytes != stats.prefetched_expert_bytes ||
                 stats.peak_prefetch_bytes == 0 || stats.prefetch_pinned_bytes == 0)) {
                throw std::runtime_error("next-layer expert weights were not prefetched during compute submission");
            }
            if (!stream_weights && gpu && prefetch_mode >= 2 &&
                (stats.weight_prefetches != 0 || stats.weight_prefetch_skips != 3 ||
                 stats.weight_prefetch_allocation_attempts != 0 ||
                 stats.prefetched_expert_bytes != 0 || stats.peak_prefetch_bytes != 0)) {
                throw std::runtime_error("next-layer expert prefetch did not respect memory limits");
            }
            if (staging == GGML_TYPE_F32 && devices.empty()) {
                std::vector<uint8_t> a(llama_state_seq_get_size(reference.second.get(), 0));
                std::vector<uint8_t> b(llama_state_seq_get_size(layered.get(), 0));
                llama_state_seq_get_data(reference.second.get(), a.data(), a.size(), 0);
                llama_state_seq_get_data(layered.get(), b.data(), b.size(), 0);
                if (a != b) {
                    throw std::runtime_error("layered prefill final sequence state mismatch");
                }
            }
            // Decode both prefixes with the same weight placement and backend kernels.
            llama_context_params decode_params = llama_context_default_params();
            decode_params.n_ctx = 256;
            decode_params.n_ubatch = ubatch;
            decode_params.n_threads = decode_params.n_threads_batch = 4;
            decode_params.type_k = decode_params.type_v = cache_type;
            llama_context_ptr replay(llama_init_from_model(candidate.first.get(), decode_params));
            std::vector<uint8_t> prefix(llama_state_seq_get_size(reference.second.get(), 0));
            if (!replay || llama_state_seq_get_data(reference.second.get(), prefix.data(), prefix.size(), 0) != prefix.size() ||
                llama_state_seq_set_data(replay.get(), prefix.data(), prefix.size(), 0) != prefix.size()) {
                throw std::runtime_error("failed to restore reference prefix for decode comparison");
            }
            common_batch next_ref(replay.get()), next_layer(layered.get());
            next_ref.add(17, tokens.size(), 0, true);
            next_layer.add(17, tokens.size(), 0, true);
            if (llama_process(replay.get(), LLAMA_PROCESS_TYPE_DECODE, next_ref.get()) ||
                llama_process(layered.get(), LLAMA_PROCESS_TYPE_DECODE, next_layer.get())) {
                throw std::runtime_error("decode after layered prefill failed");
            }
            expected.assign(llama_get_logits(replay.get()), llama_get_logits(replay.get()) + nv);
            actual.assign(llama_get_logits(layered.get()), llama_get_logits(layered.get()) + nv);
            if (!std::isfinite(nmse(expected, actual)) || nmse(expected, actual) > tolerance) {
                throw std::runtime_error("decode after layered prefill logits mismatch");
            }
            if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -1) {
                throw std::runtime_error("layered prefill accepted a nonempty context");
            }
            llama_memory_clear(llama_get_memory(layered.get()), true);
            if (stats.pipeline_active) {
                pp.max_transfer_bytes = 1;
                if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2 ||
                    llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                    throw std::runtime_error("layered prefill ignored transfer buffer budget");
                }
                pp.max_transfer_bytes = size_t(512) << 20;
            }
            pp.max_host_bytes = 1;
            if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2) {
                throw std::runtime_error("layered prefill ignored host budget");
            }
            pp.max_host_bytes = size_t(16) << 30;
            pp.max_compute_bytes = 1;
            if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2 ||
                llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                throw std::runtime_error("layered prefill workspace failure retained a partial prefix");
            }
            pp.max_compute_bytes = size_t(8) << 30;
            pp.max_expert_bytes = 1;
            if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2 ||
                llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                throw std::runtime_error("layered prefill expert budget failure retained a partial prefix");
            }
            pp.max_expert_bytes = size_t(6) << 30;
            if (stream_weights) {
                pp.max_static_bytes = 1;
                if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2 ||
                    llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                    throw std::runtime_error("static weight budget failure retained a partial prefix");
                }
                pp.max_static_bytes = size_t(256) << 20;
                if (gpu) {
                    pp.max_device_bytes = 1;
                    if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2 ||
                        llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                        throw std::runtime_error("device working-set budget failure retained a partial prefix");
                    }
                    pp.max_device_bytes = 0;
                }
            }
            if (slots) {
                pp.expert_slots = 2;
                pp.n_ubatch = ubatch;
                if (ubatch > 1 &&
                    (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2 ||
                     llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0)) {
                    throw std::runtime_error("expert slot exhaustion did not fail and clear partial state");
                }
                pp.expert_slots = slots;
            }
            if (gpu && prefetch_mode == 1) {
                pp.max_compute_bytes = stats.peak_compute_bytes - 1;
                if (llama_prefill_layered(layered.get(), tokens.data(), tokens.size(), pp) != -2 ||
                    llama_memory_seq_pos_max(llama_get_memory(layered.get()), 0) >= 0) {
                    throw std::runtime_error("prefetch execution did not clean up a reduced workspace budget");
                }
                pp.max_compute_bytes = size_t(8) << 30;
            }
            pp.staging_type = GGML_TYPE_F32;
            if (llama_prefill_layered(layered.get(), tokens.data(), 1, pp) != 0) {
                throw std::runtime_error("layered prefill did not recover after budget failure");
            }
            llama_memory_clear(llama_get_memory(reference.second.get()), true);
            expected = get_logits(reference.first.get(), reference.second.get(), {tokens[0]});
            actual.assign(llama_get_logits(layered.get()), llama_get_logits(layered.get()) + nv);
            if (!std::isfinite(nmse(expected, actual)) || nmse(expected, actual) > tolerance) {
                throw std::runtime_error("single-token prefill after failure differs from ordinary decode");
            }
            LOG("layered prefill: ubatch=%u group=%u staging=%s quantized=%d pipeline=%d prefetch=%u skip=%u, logits/state/decode/budgets OK, expert bytes=%" PRIu64 "\n",
                ubatch, group_layers, ggml_type_name(staging), quantized, stats.pipeline_active,
                stats.weight_prefetches, stats.weight_prefetch_skips, stats.expert_copy_bytes);
            ++n_cases;
        }
    }
    if (n_cases != 6) {
        throw std::runtime_error("not all layered prefill test cases executed");
    }
    return 0;
}

static void test_prefill_cost_model() {
    llama_prefill_cost cost;
    cost.measured[0] = true;
    cost.compute_ms[0] = 2;
    cost.read_bytes_per_ms = cost.write_bytes_per_ms = 1000000;
    cost.weight_bytes_per_ms = 1000;
    const std::vector<uint32_t> families = {0, 0};
    const std::vector<size_t> weights = {200000, 200000};
    const std::vector<llama_layered_prefill_group> prefetch = {{0, 1, 1}, {1, 1, 0}};
    const std::vector<llama_layered_prefill_group> grouped = {{0, 2, 0}};
    const auto estimate = [&](const std::vector<llama_layered_prefill_group> & groups, size_t row) {
        return llama_prefill_estimate(cost, groups, families, weights, 1024, 64, row, true);
    };
    if (!(estimate(prefetch, 1) < estimate(grouped, 1))) {
        throw std::runtime_error("cost model failed to prefer prefetch for weight-bound input");
    }
    cost.weight_bytes_per_ms = 1000000;
    if (!(estimate(grouped, 65536) < estimate(prefetch, 65536))) {
        throw std::runtime_error("cost model failed to prefer grouping for activation-bound input");
    }
    const std::vector<llama_layered_prefill_group> plain = {{0, 1, 0}, {1, 1, 0}};
    const double serial = llama_prefill_estimate(cost, plain, families, weights, 1024, 64, 65536, false);
    const double piped = estimate(plain, 65536);
    if (!(piped < serial)) {
        throw std::runtime_error("cost model did not account for activation overlap");
    }
    bool rejected = false;
    try {
        llama_prefill_estimate(cost, {{0, 1, 2}, {1, 1, 0}}, families, weights, 1024, 64, 1, true);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("cost model accepted an out-of-range prefetch");
    }
}

static std::vector<float> native_suffix(llama_model * model, llama_context * ctx,
                                        const std::vector<llama_token> & tokens, llama_pos first);
static std::vector<uint8_t> prefix_state(llama_context * ctx);

static int test_joint_prefill(size_t seed, float stdev, const char * target_backend) {
    LOG("joint planner: testing cost model\n");
    test_prefill_cost_model();
    LOG("joint planner: testing executable plans\n");
    const auto tokens = get_tokens(97, 128, seed);
    std::vector<ggml_backend_dev_t> devices;
    if (target_backend) {
        devices.push_back(ggml_backend_dev_by_name(target_backend));
    }
    const bool gpu = !devices.empty() && ggml_backend_dev_type(devices[0]) == GGML_BACKEND_DEVICE_TYPE_GPU;
    const llama_model_tensor_buft_override overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() }, { nullptr, nullptr },
    };
    for (ggml_type staging : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
        auto metadata = layered_metadata(true);
        auto candidate = get_model_and_ctx(metadata.get(), nullptr, seed, stdev, devices,
                                           LLAMA_SPLIT_MODE_LAYER, false, overrides, 0, 32);
        auto reference_model = get_model_and_ctx(metadata.get(), nullptr, seed, stdev, devices,
                                                 LLAMA_SPLIT_MODE_LAYER, false, gpu ? nullptr : overrides, 0, 32);
        reference_model.second.reset();
        auto * ctx = candidate.second.get();
        llama_layered_prefill_params params;
        params.require_gpu = false;
        params.n_ubatch = 32;
        params.staging_type = staging;
        params.max_compute_bytes = size_t(64) << 20;
        params.max_prefetch_bytes = size_t(64) << 20;
        params.prefetch_headroom_bytes = size_t(64) << 20;
        params.weight_prefetch = false;
        llama_layered_prefill_stats single;
        if (llama_prefill_layered(ctx, tokens.data(), tokens.size(), params, &single)) {
            throw std::runtime_error("failed to obtain reference expert allocation size");
        }
        const size_t one_layer = single.groups[1].expert_bytes;
        const auto compare = [&](const llama_layered_prefill_stats & stats) {
            if (!stats.selected_ubatch || stats.selected_ubatch > params.n_ubatch ||
                !stats.planner_candidates || !std::isfinite(stats.estimated_ms) || stats.estimated_ms <= 0) {
                throw std::runtime_error("joint planner did not return a measured executable plan");
            }
            uint32_t next_layer = 0;
            for (size_t g = 0; g < stats.groups.size(); ++g) {
                const auto & group = stats.groups[g];
                if (group.first_layer != next_layer || group.n_layers == 0 ||
                    group.n_layers > params.max_group_layers || group.compute_bytes > params.max_compute_bytes ||
                    (params.max_group_expert_bytes && group.expert_bytes > params.max_group_expert_bytes) ||
                    group.n_prefetch_layers > (g + 1 < stats.groups.size() ? stats.groups[g + 1].n_layers : 0)) {
                    throw std::runtime_error("joint planner returned an invalid group or exceeded a budget");
                }
                next_layer += group.n_layers;
            }
            const size_t row = ggml_row_size(staging, 256*4);
            if (next_layer != 4 ||
                stats.microbatches != stats.groups.size() * ((tokens.size() + stats.selected_ubatch - 1) / stats.selected_ubatch) ||
                stats.activation_read_bytes != (stats.groups.size() - 1) * row * tokens.size() ||
                stats.activation_write_bytes != stats.groups.size() * row * tokens.size() ||
                llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != (llama_pos) tokens.size() - 1) {
                throw std::runtime_error("selected plan was not executed or calibration leaked into the prefix");
            }
            auto cp = llama_context_default_params();
            cp.n_ctx = 256;
            cp.n_ubatch = stats.selected_ubatch;
            cp.n_threads = cp.n_threads_batch = 4;
            bool rounding = staging == GGML_TYPE_BF16;
            cp.cb_eval = rounding ? round_prefill_boundary : nullptr;
            cp.cb_eval_user_data = &rounding;
            llama_context_ptr ref(llama_init_from_model(reference_model.first.get(), cp));
            if (!ref) {
                throw std::runtime_error("failed to create selected-batch reference");
            }
            const auto all = get_logits(reference_model.first.get(), ref.get(), tokens);
            rounding = false;
            const size_t nv = 128;
            std::vector<float> expected(all.end() - nv, all.end());
            std::vector<float> actual(llama_get_logits(ctx), llama_get_logits(ctx) + nv);
            const double tolerance = !gpu && staging == GGML_TYPE_F32 ? 1e-10 : 1e-5;
            if (!std::isfinite(nmse(expected, actual)) || nmse(expected, actual) > tolerance) {
                throw std::runtime_error("joint planner logits mismatch at its selected batch size");
            }
            std::vector<uint8_t> prefix(llama_state_seq_get_size(ref.get(), 0));
            llama_state_seq_get_data(ref.get(), prefix.data(), prefix.size(), 0);
            if (!gpu && staging == GGML_TYPE_F32) {
                std::vector<uint8_t> actual_state(llama_state_seq_get_size(ctx, 0));
                llama_state_seq_get_data(ctx, actual_state.data(), actual_state.size(), 0);
                if (prefix != actual_state) {
                    throw std::runtime_error("joint planner final state differs from normal execution");
                }
            }
            cp.cb_eval = nullptr;
            llama_context_ptr replay(llama_init_from_model(candidate.first.get(), cp));
            if (!replay || llama_state_seq_set_data(replay.get(), prefix.data(), prefix.size(), 0) != prefix.size()) {
                throw std::runtime_error("failed to restore selected-plan reference state");
            }
            common_batch a(replay.get()), b(ctx);
            a.add(17, tokens.size(), 0, true);
            b.add(17, tokens.size(), 0, true);
            if (llama_process(replay.get(), LLAMA_PROCESS_TYPE_DECODE, a.get()) ||
                llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get())) {
                throw std::runtime_error("decode after joint planning failed");
            }
            expected.assign(llama_get_logits(replay.get()), llama_get_logits(replay.get()) + nv);
            actual.assign(llama_get_logits(ctx), llama_get_logits(ctx) + nv);
            if (!std::isfinite(nmse(expected, actual)) || nmse(expected, actual) > tolerance) {
                throw std::runtime_error("decode after joint planning logits mismatch");
            }
        };
        params.auto_plan = true;
        for (size_t limit : {size_t(0), one_layer}) {
            params.max_group_expert_bytes = limit;
            params.weight_prefetch = true;
            llama_memory_clear(llama_get_memory(ctx), true);
            params.max_calibration_ms = 2000;
            llama_layered_prefill_stats stats;
            if (llama_prefill_layered(ctx, tokens.data(), tokens.size(), params, &stats)) {
                throw std::runtime_error("joint prefill planning failed");
            }
            compare(stats);
            const auto retained = prefix_state(ctx);
            const uint32_t retained_tokens = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            const auto appended = get_tokens(32, 128, seed + 5);
            params.reuse_prefix = true;
            params.max_calibration_ms = 0;
            llama_layered_prefill_stats reused;
            if (llama_prefill_layered(ctx, appended.data(), appended.size(), params, &reused) ||
                !reused.profile_reused || reused.capacity_only || reused.calibration_probes ||
                reused.reused_prefix_tokens != retained_tokens || reused.evaluated_tokens != appended.size()) {
                throw std::runtime_error("live prefix did not reuse timing calibration without destructive probing");
            }
            auto oracle_params = llama_context_default_params();
            oracle_params.n_ctx = 256;
            oracle_params.n_ubatch = reused.selected_ubatch;
            oracle_params.n_threads = oracle_params.n_threads_batch = 4;
            bool suffix_rounding = staging == GGML_TYPE_BF16;
            oracle_params.cb_eval = suffix_rounding ? round_prefill_boundary : nullptr;
            oracle_params.cb_eval_user_data = &suffix_rounding;
            llama_context_ptr oracle(llama_init_from_model(reference_model.first.get(), oracle_params));
            if (!oracle || llama_state_seq_set_data(oracle.get(), retained.data(), retained.size(), 0) != retained.size()) {
                throw std::runtime_error("failed to restore the timed-prefix oracle");
            }
            const auto suffix_logits = native_suffix(reference_model.first.get(), oracle.get(), appended, retained_tokens);
            std::vector<float> reused_logits(llama_get_logits(ctx), llama_get_logits(ctx) + 128);
            if (!std::isfinite(nmse(suffix_logits, reused_logits)) ||
                nmse(suffix_logits, reused_logits) > (!gpu && staging == GGML_TYPE_F32 ? 1e-10 : 1e-5)) {
                throw std::runtime_error("timed resident-prefix logits mismatch");
            }
            params.reuse_prefix = false;
            llama_memory_clear(llama_get_memory(ctx), true);
            params.max_calibration_ms = 0;
            llama_layered_prefill_stats cached;
            if (llama_prefill_layered(ctx, tokens.data(), tokens.size(), params, &cached) ||
                !cached.profile_reused || cached.calibration_probes != 0) {
                throw std::runtime_error("joint planner did not reuse its context-local calibration");
            }
            compare(cached);
            LOG("joint planner: staging=%s weight limit=%zu batch=%u groups=%zu alternatives=%u, state/decode/cache OK\n",
                ggml_type_name(staging), limit, stats.selected_ubatch, stats.groups.size(), stats.planner_candidates);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        params.max_group_expert_bytes = 1;
        if (llama_prefill_layered(ctx, tokens.data(), tokens.size(), params) != -2 ||
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) >= 0) {
            throw std::runtime_error("infeasible joint plan retained a partial prefix");
        }
    }
    return 0;
}

static std::vector<float> native_suffix(llama_model * model, llama_context * ctx,
                                        const std::vector<llama_token> & tokens, llama_pos first) {
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.add(tokens[i], first + i, 0, i + 1 == tokens.size());
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("native suffix decode failed");
    }
    const size_t nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto * logits = llama_get_logits_ith(ctx, -1);
    return {logits, logits + nv};
}

static std::vector<uint8_t> prefix_state(llama_context * ctx) {
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    if (llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != state.size()) {
        throw std::runtime_error("failed to save test prefix state");
    }
    return state;
}

static int test_prefix_prefill(size_t seed, float stdev, const char * target_backend, ggml_type cache_k, bool qwen = false) {
    std::vector<ggml_backend_dev_t> devices;
    if (target_backend) {
        devices.push_back(ggml_backend_dev_by_name(target_backend));
    }
    const bool gpu = !devices.empty() && ggml_backend_dev_type(devices[0]) == GGML_BACKEND_DEVICE_TYPE_GPU;
    const llama_model_tensor_buft_override overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() }, { nullptr, nullptr },
    };
    const llama_model_tensor_buft_override streamed[] = {
        { ".*", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr },
    };
    uint32_t cases = 0;
    for (ggml_type staging : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
        for (uint32_t prefix_tokens : {3u, 33u}) {
            for (uint32_t group : qwen ? std::vector<uint32_t>{1, 2, 3, 4} : std::vector<uint32_t>{1, 2, 3, 0}) {
                auto metadata = qwen ? get_gguf_ctx(LLM_ARCH_QWEN35MOE, true) : layered_metadata(true);
                if (qwen) {
                    llama_model_saver ms(LLM_ARCH_QWEN35MOE, metadata.get());
                    ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(4));
                    ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(16));
                    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(256));
                    ms.add_kv(LLM_KV_FEATURES_LENGTH, uint32_t(256));
                    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(256));
                    for (uint32_t il = 0; il < 4; ++il) {
                        for (const char * part : {"gate", "up", "down"}) {
                            ggml_tensor tensor = {};
                            tensor.type = GGML_TYPE_Q4_K;
                            tensor.ne[0] = tensor.ne[1] = 256;
                            tensor.ne[2] = 16;
                            tensor.ne[3] = 1;
                            ggml_format_name(&tensor, "blk.%u.ffn_%s_exps.weight", il, part);
                            gguf_add_tensor(metadata.get(), &tensor);
                        }
                    }
                }
                const bool demand = qwen && group == 4;
                const bool stream_weights = demand && gpu;
                auto candidate = get_model_and_ctx(metadata.get(), nullptr, seed, stdev, devices,
                                                    LLAMA_SPLIT_MODE_LAYER, false, stream_weights ? streamed : overrides,
                                                    0, 7, nullptr, nullptr, cache_k, stream_weights ? INT32_MAX : -1,
                                                    stream_weights);
                auto reference = get_model_and_ctx(metadata.get(), nullptr, seed, stdev, devices,
                                                    LLAMA_SPLIT_MODE_LAYER, false, gpu ? nullptr : overrides, 0, 7, nullptr, nullptr, cache_k);
                const auto prefix = get_tokens(prefix_tokens, 128, seed);
                const auto suffix = get_tokens(19, 128, seed + 1);
                native_suffix(candidate.first.get(), candidate.second.get(), prefix, 0);
                const auto saved = prefix_state(candidate.second.get());
                auto cp = llama_context_default_params();
                cp.n_ctx = 256;
                cp.n_ubatch = 7;
                cp.n_threads = cp.n_threads_batch = 4;
                cp.type_k = cache_k;
                cp.type_v = cache_k;
                bool rounding = staging == GGML_TYPE_BF16;
                cp.cb_eval = rounding ? round_prefill_boundary : nullptr;
                cp.cb_eval_user_data = &rounding;
                reference.second.reset(llama_init_from_model(reference.first.get(), cp));
                if (!reference.second ||
                    llama_state_seq_set_data(reference.second.get(), saved.data(), saved.size(), 0) != saved.size()) {
                    throw std::runtime_error("failed to restore prefix with the reference precision");
                }
                const auto expected = native_suffix(reference.first.get(), reference.second.get(), suffix, prefix_tokens);
                rounding = false;
                llama_layered_prefill_params params;
                params.n_ubatch = 7;
                params.require_gpu = false;
                params.reuse_prefix = true;
                params.pipeline = group != 3;
                params.group_layers = group && !demand ? group : 1;
                params.auto_plan = group == 0;
                params.expert_slots = demand ? 16 : 0;
                params.stream_weights = stream_weights;
                params.max_group_expert_bytes = group == 0 ? 300*1024 : 0;
                params.max_calibration_ms = 0;
                params.max_compute_bytes = size_t(64) << 20;
                params.prefetch_headroom_bytes = size_t(64) << 20;
                params.max_prefetch_layers = prefix_tokens == 3 ? 1 : 2;
                params.weight_prefetch = group == 2;
                params.staging_type = staging;
                auto * ctx = candidate.second.get();
                if (group == 0 || (qwen && group == 2)) {
                    if (llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0) != saved.size()) {
                        throw std::runtime_error("failed to exercise restored resident prefix");
                    }
                }
                const auto before = prefix_state(ctx);
                params.reuse_prefix = false;
                if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params) != -1 || prefix_state(ctx) != before) {
                    throw std::runtime_error("nonempty prefix was accepted without explicit reuse");
                }
                params.reuse_prefix = true;
                const llama_token invalid = 128;
                if (llama_prefill_layered(ctx, &invalid, 1, params) != -1 || prefix_state(ctx) != before) {
                    throw std::runtime_error("invalid suffix token changed the prefix");
                }
                params.max_host_bytes = 1;
                if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params) != -2 || prefix_state(ctx) != before) {
                    throw std::runtime_error("admission failure changed the resident prefix");
                }
                const size_t suffix_bytes = suffix.size() * llama_model_n_embd(candidate.first.get()) *
                    (qwen ? 1 : 4) * ggml_type_size(staging);
                params.max_host_bytes = group == 1 ? suffix_bytes : size_t(16) << 30;
                llama_layered_prefill_stats stats;
                if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params, &stats)) {
                    throw std::runtime_error("resident prefix suffix prefill failed");
                }
                if (stats.reused_prefix_tokens != prefix_tokens || stats.evaluated_tokens != suffix.size() ||
                    stats.calibration_probes != 0 ||
                    stats.expert_layer_loads != (qwen ? 4u : 3u) ||
                    (qwen ? stats.expert_copy_bytes == 0 : stats.expert_copy_bytes != 663552) ||
                    stats.microbatches != stats.groups.size() * ((suffix.size() + stats.selected_ubatch - 1) / stats.selected_ubatch) ||
                    llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != (llama_pos) (prefix_tokens + suffix.size() - 1) ||
                    (group == 0 && (!stats.capacity_only || stats.groups.size() < 3))) {
                    throw std::runtime_error("prefix was recomputed or resident capacity planning was not honored");
                }
                if (stats.activation_host_bytes != suffix_bytes ||
                    stats.activation_pinned_bytes != (gpu ? suffix_bytes : 0)) {
                    throw std::runtime_error("prefix activation storage includes cached tokens or a second full buffer");
                }
                if (gpu && group == 2 && stats.weight_prefetches != params.max_prefetch_layers) {
                    throw std::runtime_error("resident prefix did not consume the partial/full next-group prefetch");
                }
                const double tolerance = !gpu && staging == GGML_TYPE_F32 ? 1e-10 : 1e-5;
                std::vector<float> actual(llama_get_logits(ctx), llama_get_logits(ctx) + 128);
                if (!std::isfinite(nmse(expected, actual)) || nmse(expected, actual) > tolerance) {
                    LOG_ERR("prefix mismatch: prefix=%u group=%u staging=%s NMSE=%g\n",
                            prefix_tokens, group, ggml_type_name(staging), nmse(expected, actual));
                    throw std::runtime_error("suffix logits differ from native complete-prefix execution");
                }
                const auto oracle_state = prefix_state(reference.second.get());
                if (!gpu && staging == GGML_TYPE_F32 && prefix_state(ctx) != oracle_state) {
                    throw std::runtime_error("complete prefix/suffix state differs from native execution");
                }
                cp.cb_eval = nullptr;
                llama_context_ptr replay(llama_init_from_model(candidate.first.get(), cp));
                if (!replay || llama_state_seq_set_data(replay.get(), oracle_state.data(), oracle_state.size(), 0) != oracle_state.size()) {
                    throw std::runtime_error("failed to restore prefix/suffix decode oracle");
                }
                const auto next = native_suffix(candidate.first.get(), replay.get(), {17}, prefix_tokens + suffix.size());
                const auto got = native_suffix(candidate.first.get(), ctx, {17}, prefix_tokens + suffix.size());
                if (!std::isfinite(nmse(next, got)) || nmse(next, got) > tolerance) {
                    throw std::runtime_error("decode after resident-prefix prefill differs");
                }
                const auto tail = get_tokens(9, 128, seed + 2);
                rounding = staging == GGML_TYPE_BF16;
                cp.cb_eval = rounding ? round_prefill_boundary : nullptr;
                llama_context_ptr tail_reference(llama_init_from_model(reference.first.get(), cp));
                const auto tail_state = prefix_state(ctx);
                if (!tail_reference ||
                    llama_state_seq_set_data(tail_reference.get(), tail_state.data(), tail_state.size(), 0) != tail_state.size()) {
                    throw std::runtime_error("failed to restore repeated-suffix oracle");
                }
                const auto tail_expected = native_suffix(reference.first.get(), tail_reference.get(), tail,
                    prefix_tokens + suffix.size() + 1);
                rounding = false;
                llama_layered_prefill_stats tail_stats;
                if (llama_prefill_layered(ctx, tail.data(), tail.size(), params, &tail_stats) != 0 ||
                    tail_stats.reused_prefix_tokens != prefix_tokens + suffix.size() + 1 ||
                    tail_stats.activation_host_bytes != suffix_bytes / suffix.size() * tail.size()) {
                    throw std::runtime_error("repeated suffix did not reuse the layered and decoded prefix");
                }
                actual.assign(llama_get_logits(ctx), llama_get_logits(ctx) + 128);
                if (!std::isfinite(nmse(tail_expected, actual)) || nmse(tail_expected, actual) > tolerance) {
                    throw std::runtime_error("repeated suffix logits differ from native execution");
                }
                if (llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0) != saved.size()) {
                    throw std::runtime_error("failed to restore prefix for rejection tests");
                }
                if (group == 0) {
                    const auto admission_state = prefix_state(ctx);
                    params.max_group_expert_bytes = 1;
                    llama_layered_prefill_stats rejected;
                    if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params, &rejected) != -2 ||
                        rejected.budget_shortfall_bytes == 0 || prefix_state(ctx) != admission_state) {
                        throw std::runtime_error("resident budget failure did not report its gap or preserve the prefix");
                    }
                    params.max_group_expert_bytes = 300*1024;
                    if (gpu) {
                        params.prefetch_headroom_bytes = SIZE_MAX;
                        if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params, &rejected) != -2 ||
                            rejected.capacity_shortfall_bytes == 0 || rejected.resident_state_bytes == 0 ||
                            rejected.resident_model_bytes == 0 || prefix_state(ctx) != admission_state) {
                            throw std::runtime_error("resident VRAM failure did not report capacity or preserve state");
                        }
                        params.prefetch_headroom_bytes = size_t(64) << 20;
                    }
                }
                if (((!qwen && group == 1) || demand) && prefix_tokens == 3 && staging == GGML_TYPE_F32) {
                    auto failure = params;
                    if (demand) {
                        failure.expert_slots = 2;
                    } else {
                        failure.max_expert_bytes = 1;
                    }
                    llama_layered_prefill_stats failed;
                    if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), failure, &failed) != -2 ||
                        !failed.prefix_invalidated || llama_memory_seq_pos_max(llama_get_memory(ctx), 0) >= 0) {
                        throw std::runtime_error("partial execution failure was not marked as invalidating the prefix");
                    }
                    if (llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0) != saved.size()) {
                        throw std::runtime_error("failed to restore the invalidated test prefix");
                    }
                }
                auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
                hybrid->get_mem_recr()->seq_add(0, 0, -1, 1);
                if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params) != -1) {
                    throw std::runtime_error("incomplete recurrent prefix state was accepted");
                }
                if (group == 0 || demand) {
                    if (llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0) != saved.size()) {
                        throw std::runtime_error("failed to restore prefix for shared-cell rejection");
                    }
                    auto shared = hybrid->get_mem_attn()->snapshot_metadata();
                    const auto cell = shared.cells[0].seq_pos_get(0).begin()->second;
                    shared.cells[0].seq_add(cell, 1);
                    hybrid->get_mem_attn()->restore_metadata(shared);
                    if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params) != -1) {
                        throw std::runtime_error("single-sequence prefix reuse accepted shared sequence cells");
                    }
                    hybrid->clear(true);
                    auto orphan = hybrid->get_mem_recr()->snapshot_metadata();
                    orphan.cells[0].seq_id.insert(1);
                    hybrid->get_mem_recr()->restore_metadata(orphan);
                    if (llama_prefill_layered(ctx, suffix.data(), suffix.size(), params) != -1) {
                        throw std::runtime_error("empty attention cache concealed another sequence's recurrent state");
                    }
                }
                LOG("resident prefix: model=%s staging=%s cache=%s prefix=%u group=%u pipeline=%d, suffix/state/decode/admission OK\n",
                    qwen ? "qwen" : "glm", ggml_type_name(staging), ggml_type_name(cache_k), prefix_tokens, group, stats.pipeline_active);
                ++cases;
            }
        }
    }
    return cases == 16 ? 0 : 1;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llm_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                test_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    const char * target_backend = nullptr;
    bool layered_prefill = false;
    bool layered_planner = false;
    bool joint_only = false;
    bool layered_prefix = false;
    bool qwen_prefill = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "--layered-prefill") == 0) {
            layered_prefill = true;
        } else if (strcmp(argv[i], "--qwen-prefill") == 0) {
            qwen_prefill = true;
        } else if (strcmp(argv[i], "--layered-planner") == 0) {
            layered_planner = true;
        } else if (strcmp(argv[i], "--joint-only") == 0) {
            layered_planner = true;
            joint_only = true;
        } else if (strcmp(argv[i], "--layered-prefix") == 0) {
            layered_prefix = true;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llm_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    try {
        if (qwen_prefill) {
            common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
            for (bool quantized : {false, true}) {
                for (bool pipeline : {false, true}) {
                    for (uint32_t slots : {0u, 14u}) {
                        if (test_layered_prefill(seed, stdev, target_backend, quantized, pipeline, 0, 1, true, slots)) {
                            return 1;
                        }
                    }
                }
            }
            if (target_backend) {
                if (test_layered_prefill(seed, stdev, target_backend, true, true, 1, 1, true)) {
                    return 1;
                }
                for (uint32_t slots : {0u, 14u}) {
                    if (test_layered_prefill(seed, stdev, target_backend, true, true, 1, 1, true, slots, false, true)) {
                        return 1;
                    }
                }
                if (test_layered_prefill(seed, stdev, target_backend, true, true, 0, 1, true, 14, false, true, GGML_TYPE_Q8_0)) {
                    return 1;
                }
                for (bool merge : {false, true}) {
                    if (test_layered_prefill(seed, stdev, target_backend, true, true, 1, 1, true, 0,
                                            false, true, GGML_TYPE_Q8_0, merge, true)) {
                        return 1;
                    }
                }
                if (test_layered_prefill(seed, stdev, target_backend, true, false, 0, 1, true, 0,
                                        false, true, GGML_TYPE_F16, true, false)) {
                    return 1;
                }
                return test_layered_prefill(seed, stdev, target_backend, true, true, 0, 1, true, 14, false);
            }
            return test_layered_prefill(seed, stdev, target_backend, true, false, 0, 1, true, 14, true, true);
        }
        if (layered_prefix) {
            common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
            for (bool qwen : {false, true}) {
                for (ggml_type cache : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
                    if (test_prefix_prefill(seed, stdev, target_backend, cache, qwen)) {
                        return 1;
                    }
                }
            }
            return 0;
        }
        if (layered_planner) {
            common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
            if (!joint_only) {
                for (uint32_t group : {2u, 3u, 4u}) {
                    for (bool pipeline : {false, true}) {
                        if (test_layered_prefill(seed, stdev, target_backend, true, pipeline, 0, group)) {
                            return 1;
                        }
                    }
                }
            }
            return test_joint_prefill(seed, stdev, target_backend);
        }
        if (layered_prefill) {
            common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
            for (bool pipeline : {false, true}) {
                for (bool quantized : {false, true}) {
                    const int result = test_layered_prefill(seed, stdev, target_backend, quantized, pipeline);
                    if (result != 0) {
                        return result;
                    }
                }
            }
            if (target_backend && ggml_backend_dev_type(ggml_backend_dev_by_name(target_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                for (int mode : {1, 2, 3}) {
                    const int result = test_layered_prefill(seed, stdev, target_backend, true, true, mode);
                    if (result != 0) {
                        return result;
                    }
                }
                return test_layered_prefill(seed, stdev, target_backend, true, false, 1);
            }
            return 0;
        }
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
