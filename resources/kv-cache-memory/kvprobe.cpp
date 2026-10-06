// SPDX-License-Identifier: GPL-2.0
// kvprobe: put llama.cpp's KV cache into exact states for the video captures.
//
// It runs a list of operations, separated by ';', through the public llama.h API
// (the calls tools/server makes), and logs each one with a KVPROBE prefix:
//
//   text:S:<string>     tokenize <string> (BOS if S is empty) and decode it into sequence S
//   gen:S,T,...:N       N decode steps, one greedy token per listed sequence in each step
//   cp:A:B[:P0:P1]      llama_memory_seq_cp   (A -> B, positions [P0, P1), default all)
//   rm:S:P0:P1          llama_memory_seq_rm   (-1 means open-ended)
//   keep:S              llama_memory_seq_keep
//   add:S:P0:P1:D       llama_memory_seq_add  (shift positions [P0, P1) by D)
//   save:S:FILE         llama_state_seq_get_data_ext(S) to FILE
//   load:S:FILE         llama_state_seq_set_data_ext(FILE) into S
//   report              pos_min and pos_max of every sequence that has cells
//   timed:S,T,...:N     like gen, but times the llama_decode call and the llama_synchronize wait apart
//   perf                graph reuse count and token counts from llama_perf_context
//
//   LLAMA_KV_CACHE_DEBUG=3 kvprobe -m model.gguf -c 256 -np 2 -kvu -v --ops 'text:0:The cat sat;report'
//
// With LLAMA_KV_CACHE_DEBUG=3 and -v, find_slot() prints the cell map before each micro-batch.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// sequences the probe tracks (LLAMA_MAX_SEQ itself is internal to libllama)
static constexpr int LLAMA_MAX_SEQ_PROBE = 64;

static std::vector<std::string> split(const std::string & s, char sep, int max_parts = -1) {
    std::vector<std::string> out;
    size_t start = 0;
    while (max_parts < 0 || (int) out.size() < max_parts - 1) {
        const size_t i = s.find(sep, start);
        if (i == std::string::npos) {
            break;
        }
        out.push_back(s.substr(start, i - start));
        start = i + 1;
    }
    out.push_back(s.substr(start));
    return out;
}

static void batch_add(llama_batch & b, llama_token tok, llama_pos pos, llama_seq_id s, bool output) {
    b.token[b.n_tokens]     = tok;
    b.pos[b.n_tokens]       = pos;
    b.n_seq_id[b.n_tokens]  = 1;
    b.seq_id[b.n_tokens][0] = s;
    b.logits[b.n_tokens]    = output;
    b.n_tokens++;
}

static llama_pos next_pos(llama_context * ctx, llama_seq_id s) {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), s) + 1;
}

static llama_token argmax(llama_context * ctx, int32_t i) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const float * logits = llama_get_logits_ith(ctx, i);
    llama_token best = 0;
    for (llama_token t = 1; t < llama_vocab_n_tokens(vocab); ++t) {
        if (logits[t] > logits[best]) {
            best = t;
        }
    }
    return best;
}

static llama_token last_tok[LLAMA_MAX_SEQ_PROBE] = {};

static bool op_text(llama_context * ctx, llama_seq_id s, const std::string & text) {
    const llama_pos p0 = next_pos(ctx, s);
    std::vector<llama_token> toks = common_tokenize(ctx, text, p0 == 0, false);
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch_add(batch, toks[i], p0 + (llama_pos) i, s, i + 1 == toks.size());
    }
    const int32_t ret = llama_decode(ctx, batch);
    LOG_INF("KVPROBE text seq %d: %zu tokens at positions [%d, %d], llama_decode = %d\n",
            s, toks.size(), p0, p0 + (int) toks.size() - 1, ret);
    if (ret == 0) {
        last_tok[s] = argmax(ctx, batch.n_tokens - 1);
    }
    llama_batch_free(batch);
    return ret == 0;
}

static bool op_gen(llama_context * ctx, const std::vector<llama_seq_id> & seqs, int n) {
    llama_batch batch = llama_batch_init((int32_t) seqs.size(), 0, 1);
    for (int step = 0; step < n; ++step) {
        batch.n_tokens = 0;
        for (llama_seq_id s : seqs) {
            batch_add(batch, last_tok[s], next_pos(ctx, s), s, true);
        }
        const int32_t ret = llama_decode(ctx, batch);
        LOG_INF("KVPROBE gen step %d: %zu sequences, llama_decode = %d\n", step, seqs.size(), ret);
        if (ret != 0) {
            llama_batch_free(batch);
            return false;
        }
        for (size_t i = 0; i < seqs.size(); ++i) {
            last_tok[seqs[i]] = argmax(ctx, (int32_t) i);
        }
    }
    llama_batch_free(batch);
    return true;
}

static bool op_timed(llama_context * ctx, const std::vector<llama_seq_id> & seqs, int n) {
    llama_batch batch = llama_batch_init((int32_t) seqs.size(), 0, 1);
    for (int step = 0; step < n; ++step) {
        batch.n_tokens = 0;
        for (llama_seq_id s : seqs) {
            batch_add(batch, last_tok[s], next_pos(ctx, s), s, true);
        }
        const int64_t t0 = ggml_time_us();
        const int32_t ret = llama_decode(ctx, batch);
        const int64_t t1 = ggml_time_us();
        llama_synchronize(ctx);
        const int64_t t2 = ggml_time_us();
        LOG_INF("KVPROBE timed step %d: llama_decode = %d returned after %.3f ms, llama_synchronize waited %.3f ms\n",
                step, ret, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0);
        if (ret != 0) {
            llama_batch_free(batch);
            return false;
        }
        for (size_t i = 0; i < seqs.size(); ++i) {
            last_tok[seqs[i]] = argmax(ctx, (int32_t) i);
        }
    }
    llama_batch_free(batch);
    return true;
}

static void op_perf(llama_context * ctx) {
    const llama_perf_context_data d = llama_perf_context(ctx);
    LOG_INF("KVPROBE perf: graphs reused = %d, prompt tokens = %d, generated tokens = %d\n",
            d.n_reused, d.n_p_eval, d.n_eval);
}

static void op_report(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    // a cache with one stream per sequence only knows ids below n_seq_max
    const llama_seq_id n = std::min<llama_seq_id>(LLAMA_MAX_SEQ_PROBE, (llama_seq_id) llama_n_seq_max(ctx));
    for (llama_seq_id s = 0; s < n; ++s) {
        const llama_pos lo = llama_memory_seq_pos_min(mem, s);
        if (lo < 0) {
            continue;
        }
        LOG_INF("KVPROBE report seq %d: pos_min = %d, pos_max = %d\n", s, lo, llama_memory_seq_pos_max(mem, s));
    }
}

static bool op_save(llama_context * ctx, llama_seq_id s, const std::string & path) {
    const size_t n = llama_state_seq_get_size_ext(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> buf(n);
    const size_t got = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), s, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::ofstream(path, std::ios::binary).write((const char *) buf.data(), (std::streamsize) got);
    LOG_INF("KVPROBE save seq %d: %zu bytes to %s\n", s, got, path.c_str());
    return got == n && got > 0;
}

static bool op_load(llama_context * ctx, llama_seq_id s, const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t got = llama_state_seq_set_data_ext(ctx, buf.data(), buf.size(), s, LLAMA_STATE_SEQ_FLAGS_NONE);
    LOG_INF("KVPROBE load seq %d: %zu of %zu bytes from %s\n", s, got, buf.size(), path.c_str());
    return got == buf.size();
}

static bool run(llama_context * ctx, const std::string & op) {
    const std::vector<std::string> a = split(op, ':', op.rfind("text:", 0) == 0 ? 3 : -1);
    llama_memory_t mem = llama_get_memory(ctx);
    const auto i = [&](size_t k) { return std::atoi(a.at(k).c_str()); };

    if (a[0] == "text") {
        return op_text(ctx, i(1), a.at(2));
    }
    if (a[0] == "gen") {
        std::vector<llama_seq_id> seqs;
        for (const auto & x : split(a.at(1), ',')) {
            seqs.push_back(std::atoi(x.c_str()));
        }
        return op_gen(ctx, seqs, i(2));
    }
    if (a[0] == "timed") {
        std::vector<llama_seq_id> seqs;
        for (const auto & x : split(a.at(1), ',')) {
            seqs.push_back(std::atoi(x.c_str()));
        }
        return op_timed(ctx, seqs, i(2));
    }
    if (a[0] == "perf") {
        op_perf(ctx);
        return true;
    }
    if (a[0] == "cp") {
        const llama_pos p0 = a.size() > 3 ? i(3) : -1;
        const llama_pos p1 = a.size() > 4 ? i(4) : -1;
        llama_memory_seq_cp(mem, i(1), i(2), p0, p1);
        LOG_INF("KVPROBE seq_cp %d -> %d [%d, %d)\n", i(1), i(2), p0, p1);
        return true;
    }
    if (a[0] == "rm") {
        const bool ok = llama_memory_seq_rm(mem, i(1), i(2), i(3));
        LOG_INF("KVPROBE seq_rm seq %d [%d, %d) = %d\n", i(1), i(2), i(3), ok);
        return ok;
    }
    if (a[0] == "keep") {
        llama_memory_seq_keep(mem, i(1));
        LOG_INF("KVPROBE seq_keep seq %d\n", i(1));
        return true;
    }
    if (a[0] == "add") {
        llama_memory_seq_add(mem, i(1), i(2), i(3), i(4));
        LOG_INF("KVPROBE seq_add seq %d [%d, %d) by %d\n", i(1), i(2), i(3), i(4));
        return true;
    }
    if (a[0] == "save") {
        return op_save(ctx, i(1), a.at(2));
    }
    if (a[0] == "load") {
        return op_load(ctx, i(1), a.at(2));
    }
    if (a[0] == "report") {
        op_report(ctx);
        return true;
    }
    LOG_ERR("KVPROBE unknown op '%s'\n", op.c_str());
    return false;
}

int main(int argc, char ** argv) {
    common_params params;
    std::string ops = "report";

    // -kvu is not a LLAMA_EXAMPLE_COMMON option, so the probe takes it itself
    bool kv_unified = false;
    std::vector<char *> args_fwd = { argv[0] };
    for (int k = 1; k < argc; k++) {
        if (std::string(argv[k]) == "--ops" && k + 1 < argc) {
            ops = argv[++k];
            continue;
        }
        if (std::string(argv[k]) == "-kvu") {
            kv_unified = true;
            continue;
        }
        args_fwd.push_back(argv[k]);
    }

    params.n_ctx      = 256;
    params.n_parallel = 1;

    common_init();
    if (!common_params_parse((int) args_fwd.size(), args_fwd.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.kv_unified = kv_unified;
    ggml_backend_load_all();

    common_init_result_ptr init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (init->model() == nullptr || ctx == nullptr) {
        LOG_ERR("kvprobe: failed to init\n");
        return 1;
    }

    for (const auto & op : split(ops, ';')) {
        if (op.empty()) {
            continue;
        }
        LOG_INF("KVPROBE ===== %s\n", op.c_str());
        if (!run(ctx, op)) {
            LOG_INF("KVPROBE op failed: %s\n", op.c_str());
        }
    }
    LOG_INF("KVPROBE done\n");
    return 0;
}
