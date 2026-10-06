#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Decode a sequence state written by llama_state_seq_get_data_ext (llama.cpp 99b95488c).

    parse_state.py FILE [--cells N] [--values N] [--ext] [--seq-file]

Layout (llama-context.cpp state_seq_get_data, llama-kv-cache.cpp state_write*):
  u32 magic 0xaf143cd8, i32 seq_id, u32 n_stream,
  per stream: u32 cell_count; if > 0:
    per cell: i32 pos, u32 n_seq_id, [ext: i32 x, i32 y, i32 tok, only for M-RoPE/PLE models], i32 seq_id * n_seq_id
    u32 v_trans, u32 n_layer
    per layer: i32 k_type, u64 k_size_row, cell_count rows of K
    v_trans == 0: per layer: i32 v_type, u64 v_size_row, cell_count rows of V
    v_trans == 1: per layer: i32 v_type, u32 v_size_el, u32 n_embd_v_gqa, then for each element j the
                  cell_count values of element j (one value per cell)
Pass --ext for a model whose cells carry the extra struct (the file does not say).

--seq-file: a file from llama_state_seq_save_file (the server's /slots?action=save). It starts
  u32 magic 'ggsq' 0x67677371, u32 version 3, u32 n_token_count, n_token_count u32 words (the
  server packs them, server-common.cpp server_tokens::serialize: i32 -1, u32 version 1, u32 N, N
  tokens, u32 n_media, n_media u32 media keys, then per media chunk u32 size and its bytes; the
  whole blob is padded once to 4 bytes, server_tokens_state_writer::take), then u32 n_stream
  directly: no 0xaf143cd8 and no seq_id (llama-context.cpp state_seq_save_file). The magic must
  match the mode. A file that ends early, or has bytes left over, is reported and exits with 1.
"""
import argparse
import struct

import numpy as np

TYPES = {0: ('f32', 1, 4), 1: ('f16', 1, 2), 8: ('q8_0', 32, 34), 30: ('bf16', 1, 2)}


class Reader:
    def __init__(self, data):
        self.d, self.o = data, 0

    def take(self, fmt):
        self.need(struct.calcsize('<' + fmt))
        v = struct.unpack_from('<' + fmt, self.d, self.o)
        self.o += struct.calcsize('<' + fmt)
        return v if len(v) > 1 else v[0]

    def need(self, n):
        if n < 0 or n > len(self.d) - self.o:
            raise struct.error(f'{n:,} bytes wanted, {len(self.d) - self.o:,} left')

    def raw(self, n):
        self.need(n)
        b = self.d[self.o:self.o + n]
        self.o += n
        return b


def decode_row(t, row):
    """A row's bytes as floats; q8_0 is 34-byte blocks: an f16 scale d, then 32 int8 quants, value = d * q."""
    if t in TYPES and len(row) % TYPES[t][2]:
        raise SystemExit(f'a row of {len(row)} bytes is not whole {TYPES[t][0]} blocks of {TYPES[t][2]} bytes')
    if t == 1:
        return np.frombuffer(row, dtype='<f2').astype(np.float32)
    if t == 0:
        return np.frombuffer(row, dtype='<f4')
    if t == 8:
        out = []
        for b in range(len(row) // 34):
            blk = row[34 * b:34 * b + 34]
            d = np.frombuffer(blk[:2], dtype='<f2').astype(np.float32)[0]
            out.append(d * np.frombuffer(blk[2:], dtype=np.int8).astype(np.float32))
        return np.concatenate(out)
    raise SystemExit(f'type {t}: not handled')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--cells', type=int, default=3, help='cells whose K row to print')
    ap.add_argument('--values', type=int, default=8, help='values per row to print')
    ap.add_argument('--ext', action='store_true')
    ap.add_argument('--seq-file', action='store_true', help='llama_state_seq_save_file format')
    a = ap.parse_args()
    r = Reader(open(a.file, 'rb').read())
    print(f'file: {a.file}, {len(r.d):,} bytes')
    want = 0x67677371 if a.seq_file else 0xaf143cd8
    if len(r.d) < 4 or struct.unpack_from('<I', r.d)[0] != want:
        raise SystemExit(f'not a {"slot file" if a.seq_file else "sequence state"}: magic is not 0x{want:08x}'
                         f' (try {"without" if a.seq_file else "with"} --seq-file)')
    try:
        dump(r, a)
    except struct.error as e:
        raise SystemExit(f'@{r.o:>8}  truncated: the file ends inside a field ({e})')
    if r.o != len(r.d):
        raise SystemExit(f'{len(r.d) - r.o:,} bytes left after the last field')


def dump(r, a):
    if a.seq_file:
        magic, version, n_words = r.take('I'), r.take('I'), r.take('I')
        print(f'@{0:>8}  magic = 0x{magic:08x}, version = {version}, n_token_count = {n_words}')
        if version != 3:
            raise SystemExit(f'version {version}: this parser reads version 3 (LLAMA_STATE_SEQ_VERSION)')
        r.need(4 * n_words)
        at = r.o
        words = [r.take('i') for _ in range(n_words)]
        n = words[2] if len(words) >= 3 and words[0] == -1 and words[1] == 1 else -1
        if 0 <= n and 3 + n < len(words):
            rest = len(words) - 4 - n
            print(f'@{at:>8}  packed tokens: marker = -1, version = {words[1]}, n_tokens = {n}, '
                  f'tokens = {words[3:3 + n]}, n_media = {words[3 + n]}'
                  + (f', then {rest} words of media keys and chunks' if rest else ''))
        else:
            print(f'@{at:>8}  tokens = {words}')
        at = r.o
        n_stream = r.take('I')
        print(f'@{at:>8}  n_stream = {n_stream}')
    else:
        magic, seq, n_stream = r.take('I'), r.take('i'), r.take('I')
        print(f'@{0:>8}  magic = 0x{magic:08x}, seq_id = {seq}, n_stream = {n_stream}')
    for s in range(n_stream):
        at = r.o
        n = r.take('I')
        print(f'@{at:>8}  stream {s}: cell_count = {n}')
        if n == 0:
            continue
        cells = []
        for c in range(n):
            at = r.o
            pos, nseq = r.take('i'), r.take('I')
            ext = r.take('iii') if a.ext else None
            seqs = [r.take('i') for _ in range(nseq)]
            cells.append(pos)
            if c < a.cells or c == n - 1:
                print(f'@{at:>8}    cell {c}: pos = {pos}, n_seq_id = {nseq}, seq_ids = {seqs}' + (f', ext = {ext}' if ext else ''))
        at = r.o
        v_trans, n_layer = r.take('I'), r.take('I')
        print(f'@{at:>8}  v_trans = {v_trans}, n_layer = {n_layer}')
        for il in range(n_layer):
            at = r.o
            t, row = r.take('i'), r.take('Q')
            rows = r.raw(n * row)
            if il < 2 or il == n_layer - 1:
                name = TYPES.get(t, (str(t),))[0]
                print(f'@{at:>8}  K layer {il}: type {t} ({name}), row = {row} bytes, {n} rows = {n * row:,} bytes')
                for c in range(min(a.cells, n)):
                    vals = decode_row(t, rows[c * row:(c + 1) * row])
                    print(f'            cell {c} (pos {cells[c]}): first bytes {rows[c * row:c * row + 8].hex(" ")}'
                          f' -> {np.array2string(vals[:a.values], precision=4, suppress_small=True)}')
        for il in range(n_layer):
            at = r.o
            t = r.take('i')
            if not v_trans:
                row = r.take('Q')
                r.raw(n * row)
                if il < 1 or il == n_layer - 1:
                    print(f'@{at:>8}  V layer {il}: type {t}, row = {row} bytes, {n} rows')
            else:
                el, width = r.take('I'), r.take('I')
                r.raw(width * n * el)
                if il < 1 or il == n_layer - 1:
                    print(f'@{at:>8}  V layer {il}: type {t}, element = {el} bytes, {width} elements x {n} cells')
    if r.o == len(r.d):
        print(f'@{r.o:>8}  end (exact)')


if __name__ == '__main__':
    main()
