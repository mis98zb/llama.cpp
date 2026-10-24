#include "../src/llama-kv-mask.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

template<typename T>
static T mask_drop() {
    if constexpr (std::is_same_v<T, ggml_fp16_t>) {
        return ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    } else {
        return -std::numeric_limits<T>::infinity();
    }
}

template<typename T>
static void check(
        int offset, int used, int n_kv, bool spatial, int variant,
        llama_kv_mask_threadpool * pool = nullptr, uint32_t n = 17) {
    llama_kv_cells cells;
    cells.resize(n_kv);
    for (int j = 0; j < used; ++j) {
        cells.pos_set(j, offset + j);
        cells.seq_add(j, 2);
        cells.ext_set(j, {offset + j + 1, offset + j, LLAMA_TOKEN_NULL});
    }
    if (variant == 1) {
        cells.rm(used / 2);
    } else if (variant == 2) {
        cells.pos_add(used / 2, 1);
    } else if (variant == 3) {
        cells.seq_add(used / 2, 3);
        cells.seq_rm(used / 2, 2);
    } else if (variant == 4) {
        cells.seq_add(used / 2, 3);
    }
    std::vector<llama_pos> pos(n * 3);
    std::vector<int32_t> n_seq(n, 1);
    llama_seq_id seq[] = {2, 3};
    std::vector<llama_seq_id *> seq_ids(n, seq);
    for (uint32_t i = 0; i < n; ++i) {
        pos[i] = offset - 2 + i;
        pos[i + n] = pos[i];
        pos[i + 2 * n] = pos[i] + (i % 2);
    }
    if (variant == 5) {
        n_seq[n - 1] = 2;
    } else if (variant == 6) {
        seq_ids[n - 1] = seq + 1;
    }
    llama_ubatch batch = {};
    batch.n_tokens = n;
    batch.n_pos = spatial ? 3 : 1;
    batch.pos = pos.data();
    batch.n_seq_id = n_seq.data();
    batch.seq_id = seq_ids.data();
    const T keep = T(0), drop = mask_drop<T>(), sentinel = T(23);
    std::vector<T> actual(n_kv * n, sentinel);
    const bool accepted = llama_kv_mask_contiguous(cells, batch, n_kv, actual.data(), keep, drop, pool);
    const bool expected = variant == 0 || variant == 4;
    GGML_ASSERT(accepted == expected);
    if (pool && (!accepted || uint64_t(n_kv) * n * sizeof(T) < (8u << 20))) {
        GGML_ASSERT(pool->n_threads_used == 1);
    } else if (pool && pool->n_threads_max > 1) {
        GGML_ASSERT(pool->n_threads_used > 1);
    }
    for (uint32_t i = 0; i < n; ++i) {
        for (int j = 0; j < n_kv; ++j) {
            bool visible = !cells.is_empty(j) && cells.seq_has(j, seq[0]) && cells.pos_get(j) <= pos[i];
            if (visible && spatial && cells.pos_get(j) == pos[i]) {
                visible = !cells.ext_get(j).is_2d_gt(pos[i + 2 * n], pos[i + n]);
            }
            GGML_ASSERT(actual[i * n_kv + j] == (accepted ? (visible ? keep : drop) : sentinel));
        }
    }
}

static void check_workers() {
    llama_kv_mask_threadpool pool(24);
    for (uint32_t rows : {1u, 3u, 257u, 24u, 2u, 1025u, 1u}) {
        std::vector<uint32_t> visits(rows, 0);
        const auto fill = [&](uint32_t begin, uint32_t end) {
            GGML_ASSERT(begin <= end && end <= rows);
            for (uint32_t i = begin; i < end; ++i) {
                ++visits[i];
            }
        };
        // A large byte estimate exercises all worker counts without a large allocation.
        GGML_ASSERT(pool.run(rows, 4u << 20, fill) == std::min(rows, 24u));
        for (auto visits_row : visits) {
            GGML_ASSERT(visits_row == 1);
        }
    }
}

template<typename T>
static void bench() {
    const uint32_t n = 2048;
    const int n_ctx = 64000;
    llama_kv_cells cells;
    cells.resize(n_ctx);
    for (int j = 0; j < n_ctx; ++j) {
        cells.pos_set(j, j);
        cells.seq_add(j, 0);
    }
    llama_seq_id seq = 0;
    std::vector<llama_pos> pos(n);
    std::vector<int32_t> n_seq(n, 1);
    std::vector<llama_seq_id *> seq_ids(n, &seq);
    llama_ubatch batch = {};
    batch.n_tokens = n;
    batch.n_pos = 1;
    batch.pos = pos.data();
    batch.n_seq_id = n_seq.data();
    batch.seq_id = seq_ids.data();
    std::vector<T> data(size_t(n_ctx) * n);
    std::puts(sizeof(T) == 2 ? "F16 contiguous masks, 64000 tokens, ubatch 2048" :
                              "F32 contiguous masks, 64000 tokens, ubatch 2048");
    for (uint32_t threads : {1u, 2u, 4u, 8u, 12u, 24u}) {
        llama_kv_mask_threadpool pool(threads);
        double samples[3];
        for (int rep = 0; rep < 4; ++rep) {
            double elapsed_us = 0;
            for (uint32_t offset = 0; offset < n_ctx; offset += n) {
                batch.n_tokens = std::min(n, n_ctx - offset);
                const uint32_t end = offset + batch.n_tokens;
                for (uint32_t i = 0; i < batch.n_tokens; ++i) {
                    pos[i] = offset + i;
                }
                // Only the live prefix is addressed, as in layered prefill.
                const int64_t start = ggml_time_us();
                GGML_ASSERT(llama_kv_mask_contiguous(cells, batch, end, data.data(), T(0), mask_drop<T>(), &pool));
                elapsed_us += ggml_time_us() - start;
            }
            if (rep) {
                samples[rep - 1] = elapsed_us * 1e-3;
            }
        }
        std::sort(samples, samples + 3);
        std::printf("threads=%u median=%.3f ms samples=%.3f,%.3f,%.3f\n",
            threads, samples[1], samples[0], samples[1], samples[2]);
    }
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_time_init();
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        bench<ggml_fp16_t>();
        bench<float>();
        return 0;
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    check_workers();
    std::puts("mask worker row partitions match");
    check<float>(100, 513, 520, true, 0);
    check<ggml_fp16_t>(100, 513, 520, true, 0);
    llama_kv_mask_threadpool pool(24);
    for (int offset : {0, 100, std::numeric_limits<int32_t>::max() - 2048}) {
        for (int used : {1, 9, 32, 513}) {
            for (int padding : {0, 7}) {
                for (bool spatial : {false, true}) {
                    for (int variant = 0; variant <= 6; ++variant) {
                        // A single live cell cannot have an internal hole or a position gap.
                        if (used == 1 && (variant == 1 || variant == 2)) {
                            continue;
                        }
                        check<float>(offset, used, used + padding, spatial, variant, &pool);
                        check<ggml_fp16_t>(offset, used, used + padding, spatial, variant, &pool);
                    }
                }
            }
        }
    }
    for (uint32_t threads : {1u, 2u, 4u, 8u, 12u, 24u}) {
        llama_kv_mask_threadpool workers(threads);
        for (bool spatial : {false, true}) {
            for (int variant : {0, 1, 4, 6}) {
                check<float>(100, 8191, 8199, spatial, variant, &workers, 513);
                check<ggml_fp16_t>(100, 8191, 8199, spatial, variant, &workers, 513);
            }
        }
    }
    std::puts("serial and parallel contiguous causal masks match scalar reference");
    return 0;
}
