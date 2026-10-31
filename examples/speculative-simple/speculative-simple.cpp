#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    bool layered_prefill = false;
    std::vector<char *> args = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--layered-prefill") == 0) {
            layered_prefill = true;
        } else {
            args.push_back(argv[i]);
        }
    }
    const int nargs = args.size();
    args.push_back(nullptr);
    if (!common_params_parse(nargs, args.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }
    if (layered_prefill) {
        if (params.n_parallel != 1 || params.n_predict <= 0 || params.speculative.draft.n_max <= 0) {
            LOG_ERR("layered MTP requires one sequence and positive generation/draft limits\n");
            return 1;
        }
        params.n_gpu_layers = 0;
        params.no_kv_offload = true;
        params.no_op_offload = true;
        params.no_extra_bufts = true;
        params.warmup = false;
        params.display_prompt = false;
        params.fit_params = false;
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.speculative.draft.mparams = params.model;
        params.speculative.draft.n_gpu_layers = 2;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params, false, layered_prefill);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("failed to initialize the target model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);
        if (layered_prefill) {
            params_dft.no_kv_offload = false;
            params_dft.no_op_offload = false;
            params_dft.n_gpu_layers = 2;
            params_dft.n_outputs_max = 1;
            params_dft.n_outputs_max_per_seq = 1;
        }

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    if (layered_prefill && !ctx_dft) {
        LOG_ERR("failed to initialize the GPU MTP context\n");
        return 1;
    }

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    if (params.display_prompt) {
        for (auto id : inp) {
            LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
        }
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        common_batch batch_prompt(ctx_tgt);
        const size_t prefill_tokens = inp.size() - (layered_prefill ? 0 : 1);
        for (size_t i = 0; i < prefill_tokens; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }

        const int64_t prefill_start = ggml_time_us();
        if (layered_prefill) {
            llama_layered_prefill_params pp;
            pp.full_layers = true;
            pp.auto_plan = true;
            pp.weight_prefetch = false;
            pp.n_ubatch = params.n_ubatch;
            pp.staging_type = GGML_TYPE_F32;
            pp.max_host_bytes = size_t(32) << 30;
            pp.max_transfer_bytes = 0;
            llama_layered_prefill_stats stats;
            if (llama_prefill_layered(ctx_tgt, inp.data(), inp.size(), pp, &stats)) {
                LOG_ERR("full-layer target prefill failed\n");
                common_speculative_free(spec);
                return 1;
            }
            LOG_INF("full-layer prefill: tokens=%zu seconds=%.3f weights=%" PRIu64
                    " state upload/download=%" PRIu64 "/%" PRIu64 " peak bank=%zu\n",
                    inp.size(), (ggml_time_us() - prefill_start) / 1e6, stats.layer_weight_bytes,
                    stats.state_upload_bytes, stats.state_download_bytes, stats.peak_layer_bytes);
            LOG_INF("full-layer plan: ubatch=%u groups=%zu candidates=%u dry-graphs=%u capacity-only=%d probes=%u planning=%.3f seconds\n",
                    stats.selected_ubatch, stats.groups.size(), stats.planner_candidates, stats.planner_graphs, stats.capacity_only,
                    stats.calibration_probes, stats.planning_ms / 1e3);
            LOG_INF("prefill transfers: pipeline=%d buffers=%zu input-prefetches=%u async-outputs=%u activation-up/down=%" PRIu64 "/%" PRIu64 "\n",
                    stats.pipeline_active, stats.transfer_buffer_bytes, stats.input_prefetches, stats.async_output_batches,
                    stats.activation_read_bytes, stats.activation_write_bytes);
            const auto & t = stats.timing;
            LOG_INF("prefill wall ms: execution=%.3f setup=%.3f memory=%.3f weights=%.3f state-up=%.3f state-down=%.3f\n",
                    t.total_ms, t.setup_ms, t.memory_ms, t.weight_ms, t.state_upload_ms, t.state_download_ms);
            LOG_INF("prefill wall ms: graph=%.3f inputs=%.3f activation-up=%.3f compute/wait=%.3f activation-down=%.3f output=%.3f cleanup=%.3f other=%.3f\n",
                    t.graph_ms, t.inputs_ms, t.activation_upload_ms, t.compute_ms, t.activation_download_ms,
                    t.output_ms, t.cleanup_ms, t.total_ms - t.accounted_ms());
            LOG_INF("prefill graphs: builds=%u reused=%u measurements=%u build/measure/alloc=%.3f/%.3f/%.3f ms\n",
                    stats.graph_builds, stats.graph_reuses, stats.graph_measurements,
                    t.graph_build_ms, t.graph_measure_ms, t.graph_alloc_ms);
            LOG_INF("prefill kernels: f16-indexer=%d compact-attention=%d\n",
                    stats.f16_indexer_active, stats.compact_attention_active);
            LOG_INF("prefill workspace: reserved=%zu reserve=%.3f ms backend-segments=%u indexer-window=%u\n",
                    stats.reserved_workspace_bytes, t.workspace_reserve_ms, stats.max_backend_segments, stats.max_indexer_queries);
            LOG_INF("prefill visibility: causal-positions=%d pool-ranges=%d\n",
                    stats.compact_causal_input, stats.compact_pool_input);
            for (const auto & group : stats.groups) {
                LOG_INF("full-layer plan [%u,%u): weights=%zu state=%zu workspace=%zu\n",
                        group.first_layer, group.first_layer + group.n_layers,
                        group.expert_bytes, group.state_bytes, group.compute_bytes);
                const auto & g = group.timing;
                LOG_INF("prefill wall ms [%u,%u): total=%.3f memory=%.3f weights=%.3f state-up=%.3f state-down=%.3f graph=%.3f inputs=%.3f activation-up=%.3f compute/wait=%.3f activation-down=%.3f output=%.3f cleanup=%.3f other=%.3f\n",
                        group.first_layer, group.first_layer + group.n_layers, g.total_ms, g.memory_ms,
                        g.weight_ms, g.state_upload_ms, g.state_download_ms, g.graph_ms, g.inputs_ms,
                        g.activation_upload_ms, g.compute_ms, g.activation_download_ms, g.output_ms,
                        g.cleanup_ms, g.total_ms - g.accounted_ms());
            }
        } else if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get())) {
            LOG_ERR("target prompt processing failed\n");
            common_speculative_free(spec);
            return 1;
        }

        const int64_t mtp_start = ggml_time_us();
        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
        if (layered_prefill) {
            LOG_INF("MTP prompt initialization: %.3f seconds\n", (ggml_time_us() - mtp_start) / 1e6);
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;
    const auto t_enc_end = ggml_time_us();
    const auto t_dec_start = ggml_time_us();
    if (layered_prefill) {
        prompt_tgt = inp;
        n_past = inp.size();
        id_last = common_sampler_sample(smpl.get(), ctx_tgt, -1);
        common_sampler_accept(smpl.get(), id_last, true);
        if (llama_vocab_is_eog(vocab, id_last) && !params.sampling.ignore_eos) {
            has_eos = true;
        } else {
            LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
            n_predict = 1;
        }
    }

    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);

    llama_tokens draft;

    common_prompt_checkpoint ckpt;

    while (!has_eos && (params.n_predict < 0 || n_predict < params.n_predict)) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                batch_tgt.add(draft[i], n_past + i, seq_id, true);
            }


            if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get())) {
                LOG_ERR("target verification failed\n");
                common_speculative_free(spec);
                return 1;
            }
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            common_speculative_free(spec);
            return 1;
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            draft = std::move(ids);

            {
                GGML_ASSERT(ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last) && !params.sampling.ignore_eos) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
            ++n_predict;

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted ? 100.0f * n_accept / n_drafted : 0.0f);
    LOG_INF("stop      = %s\n", has_eos ? "eos" : "generation budget");

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());


    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
