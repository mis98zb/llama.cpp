#pragma once

#include "llama-batch.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class llama_kv_mask_threadpool {
public:
    explicit llama_kv_mask_threadpool(uint32_t n_threads) : n_threads_max(std::max(1u, n_threads)) {}

    ~llama_kv_mask_threadpool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
    }

    uint32_t run(uint32_t rows, uint64_t row_bytes, const std::function<void(uint32_t, uint32_t)> & fill) {
        const uint64_t bytes = row_bytes * rows;
        const uint32_t n_threads = bytes < (8u << 20) ? 1 :
            uint32_t(std::min<uint64_t>(std::min(n_threads_max, rows), bytes / (4u << 20)));
        if (n_threads <= 1) {
            fill(0, rows);
            return 1;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            while (workers.size() + 1 < n_threads) {
                const uint32_t id = workers.size() + 1;
                workers.emplace_back([this, id, seen = generation]() mutable {
                    std::unique_lock<std::mutex> lock(mutex);
                    for (;;) {
                        ready.wait(lock, [&]() { return stopping || generation != seen; });
                        if (stopping) {
                            return;
                        }
                        seen = generation;
                        if (id >= active) {
                            continue;
                        }
                        const uint32_t begin = uint64_t(n_rows) * id / active;
                        const uint32_t end = uint64_t(n_rows) * (id + 1) / active;
                        lock.unlock();
                        task(begin, end);
                        lock.lock();
                        if (--pending == 0) {
                            done.notify_one();
                        }
                    }
                });
            }
            task = fill;
            n_rows = rows;
            active = n_threads;
            pending = n_threads - 1;
            ++generation;
        }
        ready.notify_all();
        fill(0, rows / n_threads);
        {
            std::unique_lock<std::mutex> lock(mutex);
            done.wait(lock, [&]() { return pending == 0; });
            task = {};
        }
        return n_threads;
    }

    const uint32_t n_threads_max;
    uint32_t n_threads_used = 1;

private:
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable done;
    std::vector<std::thread> workers;
    std::function<void(uint32_t, uint32_t)> task;
    uint64_t generation = 0;
    uint32_t n_rows = 0;
    uint32_t active = 0;
    uint32_t pending = 0;
    bool stopping = false;
};

template<typename T>
bool llama_kv_mask_contiguous(
        const llama_kv_cells & cells, const llama_ubatch & ubatch,
        int64_t n_kv, T * data, T keep, T drop, llama_kv_mask_threadpool * pool = nullptr) {
    if (pool) {
        pool->n_threads_used = 1;
    }
    if (ubatch.n_tokens == 0 || n_kv <= 0 || n_kv > cells.size() || cells.is_empty(0)) {
        return false;
    }
    const llama_seq_id seq = ubatch.seq_id[0][0];
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != seq) {
            return false;
        }
    }
    const int64_t first = cells.pos_get(0);
    int64_t used = 0;
    while (used < n_kv && !cells.is_empty(used)) {
        if (!cells.seq_has(used, seq) || cells.pos_get(used) != first + used) {
            return false;
        }
        ++used;
    }
    for (int64_t j = used; j < n_kv; ++j) {
        if (!cells.is_empty(j)) {
            return false;
        }
    }
    const auto fill = [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            const int64_t pos = ubatch.pos[i];
            int64_t visible = std::max<int64_t>(0, std::min(used, pos - first + 1));
            if (ubatch.is_pos_2d() && visible > 0 && first + visible - 1 == pos &&
                cells.ext_get(visible - 1).is_2d_gt(
                    ubatch.pos[i + ubatch.n_tokens * 2], ubatch.pos[i + ubatch.n_tokens])) {
                --visible;
            }
            T * row = data + n_kv * i;
            std::fill(row, row + visible, keep);
            std::fill(row + visible, row + n_kv, drop);
        }
    };
    if (pool) {
        pool->n_threads_used = pool->run(ubatch.n_tokens, uint64_t(n_kv) * sizeof(T), fill);
    } else {
        fill(0, ubatch.n_tokens);
    }
    return true;
}
