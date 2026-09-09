#!/usr/bin/env python3
"""Diagrams for the llama.cpp KV cache post.

d1  one token's contribution: rows in every layer's K and V tensor
d2  the anatomy of a cell: metadata columns beside the tensor rows
d3  the mask loop: the only consumer of pos and seq
d4  a shared system prompt: why seq is a set and not a single id
d5  find_slot placement and the SWA reuse condition
d6  unified vs non-unified cell arrays and the resulting n_kv
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "kvcache")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c5f8a"
GREEN = "#2e7d52"
ORANGE = "#b8681c"
RED = "#c0392b"
PURPLE = "#6b4c9a"
GREY = "#888"
DARK = "#333"

FONT = 'font-family="Helvetica, Arial, sans-serif"'
MONO = 'font-family="Menlo, Consolas, monospace"'


def head(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" {FONT}>\n'
            f'  <defs>\n'
            f'    <marker id="ar" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{BLUE}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arGrey" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{GREY}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arRed" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{RED}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arGreen" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{GREEN}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arPurple" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{PURPLE}"/>\n'
            f'    </marker>\n'
            f'  </defs>\n'
            f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n')


def title(x, y, t, size=16):
    return (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
            f'font-weight="bold" fill="{DARK}">{t}</text>\n')


def txt(x, y, t, size=12, fill="#555", anchor="middle", weight="normal", mono=False):
    f = MONO if mono else ""
    return (f'  <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" {f}>{t}</text>\n')


def box(x, y, w, h, fill="#ffffff", stroke=GREY, rx=4, sw=1.4, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def line(x1, y1, x2, y2, stroke=GREY, sw=1.4, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n'


# ---------------------------------------------------------------------------
# d1: what one token puts into the cache
# ---------------------------------------------------------------------------
def d1():
    W, H = 880, 430
    s = head(W, H)
    s += title(W / 2, 26, "One token writes one row into every layer, at the same index")

    # the token
    s += box(28, 84, 132, 54, "#f2f6fa", BLUE, 5, 1.6)
    s += txt(94, 106, '"sat"', 14, BLUE, "middle", "bold", mono=True)
    s += txt(94, 126, "position 7, chat A", 10.5, "#666")

    s += line(166, 111, 202, 111, BLUE, 1.6, marker="ar")
    s += txt(184, 100, "forward", 9.5, "#777")
    s += txt(28, 176, "The model computes a K", 11, "#666", "start")
    s += txt(28, 192, "vector and a V vector for", 11, "#666", "start")
    s += txt(28, 208, "this token in each of its", 11, "#666", "start")
    s += txt(28, 224, "36 layers.", 11, "#666", "start")

    # layer stack
    lx, ly = 216, 62
    lw, lh = 300, 40
    shown = [("layer 0", 0), ("layer 1", 1), ("layer 2", 2)]
    for i, (nm, k) in enumerate(shown):
        y = ly + i * (lh + 8)
        s += box(lx, y, lw, lh, "#ffffff", GREY, 4, 1.2)
        s += txt(lx + 42, y + 24, nm, 10.5, "#777", "middle", mono=True)
        # K and V mini rows
        for j, (lab, col) in enumerate([("K", BLUE), ("V", GREEN)]):
            rx = lx + 86 + j * 106
            s += box(rx, y + 8, 98, 24, "#eef4fa" if j == 0 else "#eef7f1", col, 3, 1.2)
            s += txt(rx + 49, y + 24, f"{lab} row 7", 10.5, col, "middle", "bold", mono=True)

    yd = ly + 3 * (lh + 8)
    s += txt(lx + lw / 2, yd + 14, "\u22ee", 18, "#aaa")
    y35 = yd + 26
    s += box(lx, y35, lw, lh, "#ffffff", GREY, 4, 1.2)
    s += txt(lx + 42, y35 + 24, "layer 35", 10.5, "#777", "middle", mono=True)
    for j, (lab, col) in enumerate([("K", BLUE), ("V", GREEN)]):
        rx = lx + 86 + j * 106
        s += box(rx, y35 + 8, 98, 24, "#eef4fa" if j == 0 else "#eef7f1", col, 3, 1.2)
        s += txt(rx + 49, y35 + 24, f"{lab} row 7", 10.5, col, "middle", "bold", mono=True)

    # the vertical "same index" bracket
    brx = lx + lw + 16
    s += line(brx, ly + 6, brx, y35 + lh - 6, PURPLE, 1.6)
    s += line(brx - 5, ly + 6, brx + 5, ly + 6, PURPLE, 1.6)
    s += line(brx - 5, y35 + lh - 6, brx + 5, y35 + lh - 6, PURPLE, 1.6)
    s += txt(brx + 12, ly + 76, "row index 7", 12, PURPLE, "start", "bold", mono=True)
    s += txt(brx + 12, ly + 94, "in all 72 tensors.", 11, PURPLE, "start")
    s += txt(brx + 12, ly + 112, "That one shared index", 11, "#666", "start")
    s += txt(brx + 12, ly + 128, "is what the code calls", 11, "#666", "start")
    s += txt(brx + 12, ly + 146, "a cell.", 12, PURPLE, "start", "bold")

    # arithmetic strip
    ay = H - 78
    s += box(40, ay, W - 80, 58, "#fbfcfd", GREY, 5, 1.2)
    s += txt(56, ay + 22, "Qwen3-4B, f16 cache:", 11.5, DARK, "start", "bold")
    s += txt(56, ay + 42,
             "1024 values per row  \u00d7  2 bytes  \u00d7  2 rows (K and V)  \u00d7  36 layers  "
             "=  147,456 bytes  =  144 KiB for this one token",
             11.5, "#555", "start", mono=True)
    s += "</svg>\n"
    open(os.path.join(OUT, "one-token.svg"), "w").write(s)


# ---------------------------------------------------------------------------
# d2: anatomy of a cell
# ---------------------------------------------------------------------------
def d2():
    W, H = 880, 470
    s = head(W, H)
    s += title(W / 2, 26, "A cell is a row index plus four columns of bookkeeping")
    s += txt(W / 2, 48,
             "Server state: chat A and chat B opened with the same 3-token system prompt. "
             "A is 3 turns in, B has just started.", 11.5, "#666")

    cols = [("cell", 54), ("pos", 54), ("seq", 84), ("shift", 54), ("K row", 116), ("V row", 116)]
    rows = [
        ("0", "0", "{A,B}", "0", "shared", True),
        ("1", "1", "{A,B}", "0", "shared", True),
        ("2", "2", "{A,B}", "0", "shared", True),
        ("3", "3", "{A}", "0", "chat A", False),
        ("4", "4", "{A}", "0", "chat A", False),
        ("5", "-1", "-", "0", "", None),
        ("6", "3", "{B}", "0", "chat B", False),
        ("7", "-1", "-", "0", "", None),
    ]

    x0, y0 = 96, 92
    rh = 34
    # header
    cx = x0
    for nm, cw in cols:
        s += txt(cx + cw / 2, y0 - 10, nm, 11, "#777", "middle", "bold", mono=True)
        cx += cw + 6

    for r, (ci, pos, seq, sh, tag, shared) in enumerate(rows):
        y = y0 + r * (rh + 4)
        empty = pos == "-1"
        cx = x0
        vals = [ci, pos, seq, sh]
        colors = ["#777", BLUE, PURPLE, ORANGE]
        for k, (nm, cw) in enumerate(cols[:4]):
            v = vals[k]
            fill = "#f7f7f7" if empty else "#ffffff"
            stroke = GREY if empty else colors[k]
            s += box(cx, y, cw, rh, fill, stroke, 3, 1.2)
            s += txt(cx + cw / 2, y + 22, v, 11.5, "#aaa" if empty else colors[k],
                     "middle", "bold", mono=True)
            cx += cw + 6
        # K and V rows
        for k, (nm, cw) in enumerate(cols[4:]):
            col = BLUE if k == 0 else GREEN
            if empty:
                s += box(cx, y, cw, rh, "#f7f7f7", GREY, 3, 1.0, dash="3,3")
                s += txt(cx + cw / 2, y + 22, "unused", 10.5, "#bbb", "middle", mono=True)
            else:
                fill = "#e6eef7" if shared else ("#eef4fa" if k == 0 else "#eef7f1")
                s += box(cx, y, cw, rh, fill, col, 3, 1.2)
                s += txt(cx + cw / 2, y + 22, "1024 values", 10.5, col, "middle", mono=True)
            cx += cw + 6

    # side annotations
    ax = x0 + sum(cw for _, cw in cols) + 6 * 6 + 16
    s += txt(ax, y0 + 40, "cells 0-2 carry two", 11, PURPLE, "start", "bold")
    s += txt(ax, y0 + 56, "bits, so the system", 11, PURPLE, "start")
    s += txt(ax, y0 + 72, "prompt is stored once", 11, PURPLE, "start")
    s += txt(ax, y0 + 88, "and read by both chats.", 11, PURPLE, "start")

    s += txt(ax, y0 + 5 * (rh + 4) + 22, "pos = -1 means free.", 11, GREY, "start", "bold")
    s += txt(ax, y0 + 5 * (rh + 4) + 38, "The vectors are still", 11, "#888", "start")
    s += txt(ax, y0 + 5 * (rh + 4) + 54, "there; nothing reads them.", 11, "#888", "start")

    s += txt(W / 2, H - 46,
             "The four left columns are separate std::vectors in llama_kv_cells. "
             "The two right columns are slices of the ggml tensors cache_k_l* and cache_v_l*.",
             11.5, "#555")
    s += txt(W / 2, H - 26,
             "Nothing in the codebase is a struct named cell. The index is the cell.",
             11.5, DARK, "middle", "bold")
    s += "</svg>\n"
    open(os.path.join(OUT, "cell-and-tensors.svg"), "w").write(s)


# ---------------------------------------------------------------------------
# d3: the mask loop is the only consumer of pos and seq
# ---------------------------------------------------------------------------
def d3():
    W, H = 880, 520
    s = head(W, H)
    s += title(W / 2, 26, "Building one row of the attention mask: every field earns its place here")
    s += txt(W / 2, 48,
             "Chat A submits three more tokens, positions 5, 6 and 7. This is the mask row "
             "for the query at position 5.", 11.5, "#666")

    # query box
    s += box(40, 74, 150, 46, "#f2f6fa", BLUE, 5, 1.5)
    s += txt(115, 94, "query token", 11, "#666")
    s += txt(115, 112, "seq A, pos 5", 11.5, BLUE, "middle", "bold", mono=True)

    cells = [
        (0, "0", "{A,B}", "keep"),
        (1, "1", "{A,B}", "keep"),
        (2, "2", "{A,B}", "keep"),
        (3, "3", "{A}", "keep"),
        (4, "4", "{A}", "keep"),
        (5, "5", "{A}", "keep"),
        (6, "3", "{B}", "seq"),
        (7, "6", "{A}", "future"),
        (8, "-1", "-", "empty"),
    ]

    x0, y0 = 62, 158
    cw, ch = 78, 30
    gap = 4

    labels = [("cell", 0), ("pos[j]", 1), ("seq[j]", 2)]
    for r, (lab, k) in enumerate(labels):
        y = y0 + r * (ch + gap)
        s += txt(x0 - 6, y + 20, lab, 11, DARK, "end", "bold", mono=True)
        for i, (ci, pos, seq, verdict) in enumerate(cells):
            x = x0 + 8 + i * (cw + gap)
            empty = verdict == "empty"
            v = [str(ci), pos, seq][k]
            col = ["#777", BLUE, PURPLE][k]
            s += box(x, y, cw, ch, "#f7f7f7" if empty else "#ffffff",
                     GREY if empty else col, 3, 1.2)
            s += txt(x + cw / 2, y + 20, v, 11.5, "#aaa" if empty else col,
                     "middle", "bold", mono=True)

    # verdict row
    yv = y0 + 3 * (ch + gap) + 16
    s += txt(x0 - 6, yv + 22, "mask", 11, DARK, "end", "bold", mono=True)
    reasons = {
        "keep": ("0", GREEN, "#eaf5ee", "read it"),
        "seq":  ("-inf", RED, "#fbeceb", "other chat"),
        "future": ("-inf", ORANGE, "#fdf3e8", "not yet"),
        "empty": ("-inf", GREY, "#f4f4f4", "nothing here"),
    }
    for i, (ci, pos, seq, verdict) in enumerate(cells):
        x = x0 + 8 + i * (cw + gap)
        val, col, fill, why = reasons[verdict]
        s += line(x + cw / 2, yv - 14, x + cw / 2, yv - 2, col, 1.3, marker=None)
        s += box(x, yv, cw, 34, fill, col, 3, 1.5)
        s += txt(x + cw / 2, yv + 22, val, 12, col, "middle", "bold", mono=True)
        s += txt(x + cw / 2, yv + 50, why, 10, col)

    # the three tests, spelled out
    ty = yv + 78
    s += box(40, ty, W - 80, 116, "#fbfcfd", GREY, 5, 1.2)
    s += txt(58, ty + 24, "The loop over j runs these three tests, in this order:",
             11.5, DARK, "start", "bold")
    tests = [
        ("cells.is_empty(j)", GREY,
         "nothing was ever written here, so there is no key to compare against"),
        ("!cells.seq_has(j, seq_id)", RED,
         "this key belongs to another conversation; mixing them would corrupt both"),
        ("cells.pos_get(j) > p1", ORANGE,
         "this key is from a later position, and a token may not attend to its own future"),
    ]
    for i, (code, col, why) in enumerate(tests):
        yy = ty + 48 + i * 22
        s += txt(58, yy, code, 11, col, "start", "bold", mono=True)
        s += txt(272, yy, why, 11, "#555", "start")

    s += txt(W / 2, H - 14,
             "pos exists to answer the third question and seq exists to answer the second. "
             "Remove either and this row cannot be built.", 11.5, DARK, "middle", "bold")
    s += "</svg>\n"
    open(os.path.join(OUT, "mask-row.svg"), "w").write(s)


# ---------------------------------------------------------------------------
# d4: the life of a cell
# ---------------------------------------------------------------------------
def d4():
    W, H = 880, 380
    s = head(W, H)
    s += title(W / 2, 26, "The life of one cell, and the only two ways it comes back")

    states = [
        (40, "free", "pos = -1\nseq = {}", GREY, "#f7f7f7"),
        (250, "written", "pos = 12\nseq = {A}", BLUE, "#eef4fa"),
        (460, "shared", "pos = 12\nseq = {A,B}", PURPLE, "#f2eefa"),
        (670, "free again", "pos = -1\nseq = {}", GREY, "#f7f7f7"),
    ]
    bw, bh = 150, 76
    y = 104
    for x, nm, body, col, fill in states:
        s += box(x, y, bw, bh, fill, col, 6, 1.6)
        s += txt(x + bw / 2, y + 26, nm, 12.5, col, "middle", "bold")
        for k, ln in enumerate(body.split("\n")):
            s += txt(x + bw / 2, y + 46 + k * 16, ln, 10.5, "#666", "middle", mono=True)

    arrows = [
        ("find_slot()", "then pos_set()", BLUE),
        ("seq_cp(A, B)", "one more bit", PURPLE),
        ("seq_rm(B),", "then seq_rm(A)", GREEN),
    ]
    for k, (a, b, col) in enumerate(arrows):
        xa = states[k][0] + bw + 8
        xb = states[k + 1][0] - 8
        s += line(xa, y + bh / 2, xb, y + bh / 2, col, 1.6, marker="ar" if col == BLUE else
                  ("arPurple" if col == PURPLE else "arGreen"))
        xm = (xa + xb) / 2
        s += txt(xm, y - 22, a, 10, col, "middle", "bold", mono=True)
        s += txt(xm, y - 8, b, 10, col, "middle", mono=True)

    # the refcount note
    ry = 226
    s += box(60, ry, W - 120, 76, "#fbfcfd", GREY, 5, 1.2)
    s += txt(80, ry + 24, "Dropping chat B does not free the cell:", 11.5, DARK, "start", "bold")
    s += txt(80, ry + 44,
             "seq[i].reset(B) clears one bit. The cell empties only when seq[i].none() is true, "
             "which is the whole", 11.5, "#555", "start")
    s += txt(80, ry + 62,
             "reclamation rule. There is no reference count, no timestamp, and no eviction score anywhere in it.",
             11.5, "#555", "start")

    s += txt(W / 2, ry + 106,
             "The second route back to free is the sliding-window test inside find_slot(), "
             "which reuses a cell whose position", 11.5, "#555")
    s += txt(W / 2, ry + 124,
             "has fallen out of its sequence's window. Those are the only two.", 11.5, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "cell-life.svg"), "w").write(s)


# ---------------------------------------------------------------------------
# d5: find_slot placement
# ---------------------------------------------------------------------------
def d5():
    W, H = 880, 400
    s = head(W, H)
    s += title(W / 2, 26, "find_slot() walks the array from a cursor; it never searches by content")

    cw, chh = 46, 30
    n = 14
    x0 = 70
    y0 = 84

    s += txt(W / 2, 52, "before: the cursor sits at cell 5 and three tokens of chat A need homes",
             12, "#666")

    used = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 9: "B*"}
    for i in range(n):
        x = x0 + i * (cw + 4)
        if i in used:
            swa = used[i].endswith("*")
            s += box(x, y0, cw, chh, "#fdf3e8" if swa else "#eef4fa",
                     ORANGE if swa else BLUE, 3, 1.3)
            s += txt(x + cw / 2, y0 + 20, used[i].rstrip("*"), 11, ORANGE if swa else BLUE,
                     "middle", "bold", mono=True)
        else:
            s += box(x, y0, cw, chh, "#fbfbfb", GREY, 3, 1.0, dash="3,3")
        s += txt(x + cw / 2, y0 - 6, str(i), 10, "#888")

    hx = x0 + 5 * (cw + 4) + cw / 2
    s += line(hx, y0 + chh + 26, hx, y0 + chh + 6, RED, 1.6, marker="arRed")
    s += txt(hx, y0 + chh + 40, "head", 11, RED, "middle", "bold", mono=True)

    s += txt(x0 + 9 * (cw + 4) + cw / 2 + 30, y0 + chh + 40,
             "chat B, but outside its sliding window", 10, ORANGE)
    s += line(x0 + 9 * (cw + 4) + cw / 2, y0 + chh + 26,
              x0 + 9 * (cw + 4) + cw / 2, y0 + chh + 6, ORANGE, 1.4, "3,2", "arGrey")

    y1 = y0 + 150
    s += txt(W / 2, y1 - 26,
             "after: they land in 5, 6, 7. Cell 9 was equally eligible; the scan simply reached 5 first.",
             12, "#666")
    newly = {5: "A", 6: "A", 7: "A"}
    for i in range(n):
        x = x0 + i * (cw + 4)
        if i in newly:
            s += box(x, y1, cw, chh, "#eaf5ee", GREEN, 3, 1.6)
            s += txt(x + cw / 2, y1 + 20, newly[i], 11, GREEN, "middle", "bold", mono=True)
        elif i in used:
            swa = used[i].endswith("*")
            s += box(x, y1, cw, chh, "#fdf3e8" if swa else "#eef4fa",
                     ORANGE if swa else BLUE, 3, 1.3)
            s += txt(x + cw / 2, y1 + 20, used[i].rstrip("*"), 11, ORANGE if swa else BLUE,
                     "middle", "bold", mono=True)
        else:
            s += box(x, y1, cw, chh, "#fbfbfb", GREY, 3, 1.0, dash="3,3")
        s += txt(x + cw / 2, y1 - 6, str(i), 10, "#888")

    s += txt(W / 2, H - 44,
             "A cell is available when it is empty, or when it holds exactly one sequence whose "
             "position has fallen", 11.5, "#555")
    s += txt(W / 2, H - 26,
             "outside that sequence's sliding window. The three tokens need not be adjacent, "
             "which is why writes", 11.5, "#555")
    s += txt(W / 2, H - 8,
             "go through an index tensor rather than a memcpy at the cursor.", 11.5, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "find-slot.svg"), "w").write(s)


# ---------------------------------------------------------------------------
# d6: unified vs non-unified, and n_kv
# ---------------------------------------------------------------------------
def d6():
    W, H = 880, 440
    s = head(W, H)
    s += title(W / 2, 26, "Same bytes, different partitioning, different attention width")

    chh = 34
    y0 = 76
    s += txt(60, y0 - 22, "unified: one array of 40960 cells, shared by all four slots",
             12.5, DARK, "start", "bold")
    x0 = 60
    total = 40960
    scale = 740.0 / total
    s += box(x0, y0, 740, chh, "#fbfbfb", GREY, 3, 1.2)
    s += box(x0, y0, 31111 * scale, chh, "#eef4fa", BLUE, 3, 1.3)
    s += txt(x0 + 31111 * scale / 2, y0 + 22, "slot 0, 31111 tokens", 11.5, BLUE, "middle", "bold")
    s += box(x0 + 31111 * scale, y0, 300 * scale, chh, "#eaf5ee", GREEN, 3, 1.3)

    nkv_u = 31232
    bx = x0 + nkv_u * scale
    s += line(x0, y0 + chh + 14, bx, y0 + chh + 14, RED, 1.6)
    s += line(x0, y0 + chh + 9, x0, y0 + chh + 19, RED, 1.6)
    s += line(bx, y0 + chh + 9, bx, y0 + chh + 19, RED, 1.6)
    s += txt((x0 + bx) / 2, y0 + chh + 32,
             "n_kv = 31232 for every slot, including slot 1", 11.5, RED, "middle", "bold", mono=True)
    s += txt(bx + 24, y0 + 16, "slot 1, 30 tokens", 10.5, GREEN, "start", "bold")
    s += txt(bx + 24, y0 + 32, "pays the full width", 10.5, GREEN, "start")

    y1 = 230
    s += txt(60, y1 - 22, "non-unified: four arrays of 40960 cells, one per slot",
             12.5, DARK, "start", "bold")
    aw = 176
    for k in range(4):
        x = 60 + k * (aw + 8)
        s += box(x, y1, aw, chh, "#fbfbfb", GREY, 3, 1.2)
        if k == 0:
            s += box(x, y1, aw * (31111 / 40960.0), chh, "#eef4fa", BLUE, 3, 1.3)
            s += txt(x + aw * 0.38, y1 + 22, "slot 0", 11, BLUE, "middle", "bold")
        elif k == 1:
            s += box(x, y1, max(4, aw * (256 / 40960.0)), chh, "#eaf5ee", GREEN, 3, 1.3)
            s += txt(x + aw / 2, y1 + 22, "slot 1", 11, GREEN, "middle", "bold")
        else:
            s += txt(x + aw / 2, y1 + 22, f"slot {k}", 11, "#aaa", "middle")
        s += txt(x + aw / 2, y1 - 6, f"stream {k}", 10, "#888")

    x_s0 = 60
    s += line(x_s0, y1 + chh + 12, x_s0 + aw * (31111 / 40960.0), y1 + chh + 12, RED, 1.5)
    s += txt(x_s0 + aw * 0.38, y1 + chh + 28, "n_kv = 31232", 10.5, RED, "middle", "bold", mono=True)
    x_s1 = 60 + (aw + 8)
    s += line(x_s1, y1 + chh + 12, x_s1 + 8, y1 + chh + 12, GREEN, 1.5)
    s += txt(x_s1 + 20, y1 + chh + 28, "n_kv = 256", 10.5, GREEN, "start", "bold", mono=True)

    ry = 340
    s += box(60, ry, 760, 78, "#fbfcfd", GREY, 5, 1.2)
    s += txt(80, ry + 22,
             "Measured: RTX 4090, Qwen3-4B Q4_K_M, q4_0 KV, 40960-token slots, idle-slot purge disabled.",
             11.5, DARK, "start", "bold")
    s += txt(80, ry + 42,
             "Slot 1 decoding alone, then decoding again with the 31111-token prompt parked on slot 0:",
             11.5, "#555", "start")
    s += txt(80, ry + 60,
             "unified  228 -> 202 tok/s (11.5%)        non-unified  235 -> 231 tok/s (1.6%)",
             11.5, RED, "start", "bold", mono=True)

    s += "</svg>\n"
    open(os.path.join(OUT, "unified-vs-split.svg"), "w").write(s)


if __name__ == "__main__":
    d1()
    d2()
    d3()
    d4()
    d5()
    d6()
    print("wrote", os.path.abspath(OUT))
