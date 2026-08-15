#!/usr/bin/env python3
"""Diagrams 4-7 for the ubatch / continuous batching post: the unified batch,
the mask, chunked prefill, and the MLP."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "ubatch")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c5f8a"
GREEN = "#2e7d52"
ORANGE = "#b8681c"
RED = "#c0392b"
GREY = "#888"
DARK = "#333"
FONT = 'font-family="Helvetica, Arial, sans-serif"'

TINT = {BLUE: "#e8eef4", GREEN: "#e9f2ec", ORANGE: "#fdf3e8", RED: "#fbeeea"}


def head(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" {FONT}>\n'
            f'  <defs>\n'
            f'    <marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">\n'
            f'      <polygon points="0 0, 9 3, 0 6" fill="{BLUE}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arGrey" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">\n'
            f'      <polygon points="0 0, 9 3, 0 6" fill="{GREY}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arRed" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">\n'
            f'      <polygon points="0 0, 9 3, 0 6" fill="{RED}"/>\n'
            f'    </marker>\n'
            f'  </defs>\n'
            f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n')


def title(x, y, t, size=16):
    return (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
            f'font-weight="bold" fill="{DARK}">{t}</text>\n')


def txt(x, y, t, size=12, fill="#555", anchor="middle", weight="normal", family=None):
    fam = f' font-family="{family}"' if family else ""
    return (f'  <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}"{fam}>{t}</text>\n')


def mono(x, y, t, size=11.5, fill="#444", anchor="middle", weight="normal"):
    return txt(x, y, t, size, fill, anchor, weight, family="Menlo, Consolas, monospace")


def box(x, y, w, h, fill="#ffffff", stroke=GREY, rx=4, sw=1.4):
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>\n')


def line(x1, y1, x2, y2, stroke=GREY, sw=1.4, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n'


# ----------------------------------------------------------------------------
# Diagram 4: one ubatch, three conversations
# ----------------------------------------------------------------------------
def d4():
    W, H = 800, 460
    s = head(W, H)
    s += title(W / 2, 26, "One physical batch, drawn from three different conversations")
    s += txt(W / 2, 48, "llama.cpp fills the batch in arrival order and does not group by sequence",
             11.5, "#777")

    rows = [("the", 0, 4021, BLUE),
            ("import", 1, 17, GREEN),
            ("numpy", 1, 18, GREEN),
            ("recipe", 2, 903, ORANGE),
            ("cat", 0, 4022, BLUE),
            ("as", 1, 19, GREEN)]

    top = 78
    rh = 42
    x = 150
    tokw = 130
    for i, (tok, sq, pos, col) in enumerate(rows):
        y = top + i * (rh + 6)
        s += txt(x - 16, y + 27, f"row {i}", 11, "#888", "end")
        s += box(x, y, tokw, rh, TINT[col], col, 4, 1.5)
        s += mono(x + tokw / 2, y + 26, f'"{tok}"', 12.5, col, "middle", "bold")
        s += box(x + tokw + 12, y, 92, rh, "#ffffff", col, 4, 1.2)
        s += mono(x + tokw + 58, y + 26, f"seq {sq}", 11.5, col)
        s += box(x + tokw + 112, y, 108, rh, "#ffffff", "#bbb", 4, 1.2)
        s += mono(x + tokw + 166, y + 26, f"pos {pos}", 11.5, "#666")

    # legend
    ly = top + 6 * (rh + 6) + 16
    for j, (col, name) in enumerate([(BLUE, "conversation 0"), (GREEN, "conversation 1"),
                                     (ORANGE, "conversation 2")]):
        lx = 150 + j * 175
        s += box(lx, ly, 14, 14, TINT[col], col, 3, 1.3)
        s += txt(lx + 22, ly + 12, name, 11.5, col, "start")

    s += txt(W / 2, H - 40,
             "n_tokens is 6, so the tensors have 6 rows. There are no filler rows,",
             12, "#555")
    s += txt(W / 2, H - 22,
             "and the rows do not have to be sorted or of equal length.", 12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "unified-batch.svg"), "w").write(s)


# ----------------------------------------------------------------------------
# Diagram 5: the mask
# ----------------------------------------------------------------------------
def d5():
    W, H = 860, 560
    s = head(W, H)
    s += title(W / 2, 26, "The attention mask is what keeps the conversations apart")

    # 6 queries x 6 keys, same rows as diagram 4
    rows = [("seq 0", 4021, BLUE), ("seq 1", 17, GREEN), ("seq 1", 18, GREEN),
            ("seq 2", 903, ORANGE), ("seq 0", 4022, BLUE), ("seq 1", 19, GREEN)]
    n = len(rows)
    cell = 46
    gx = 250
    gy = 110

    s += txt(gx + n * cell / 2, gy - 52, "keys  (what a query is allowed to look at)", 12, "#666")
    # column headers
    for j, (sq, pos, col) in enumerate(rows):
        cx = gx + j * cell + cell / 2
        s += mono(cx, gy - 30, sq.replace("seq ", "s"), 11, col, "middle", "bold")
        s += mono(cx, gy - 15, str(pos), 9.5, "#999")
    # row headers
    s += (f'  <text x="{gx - 96}" y="{gy + n * cell / 2}" text-anchor="middle" font-size="12" '
          f'fill="#666" transform="rotate(-90 {gx - 96} {gy + n * cell / 2})">queries</text>\n')
    for i, (sq, pos, col) in enumerate(rows):
        cy = gy + i * cell + cell / 2
        s += mono(gx - 16, cy + 4, f"{sq.replace('seq ', 's')} pos {pos}", 11, col, "end", "bold")

    for i, (sqi, posi, coli) in enumerate(rows):
        for j, (sqj, posj, colj) in enumerate(rows):
            x = gx + j * cell
            y = gy + i * cell
            same_seq = sqi == sqj
            causal_ok = posj <= posi
            allowed = same_seq and causal_ok
            if allowed:
                s += box(x, y, cell, cell, TINT[coli], coli, 3, 1.2)
                s += txt(x + cell / 2, y + cell / 2 + 6, "&#10003;", 17, coli, "middle", "bold")
            elif not same_seq:
                s += box(x, y, cell, cell, "#f7f7f7", "#ddd", 3, 1.0)
                s += txt(x + cell / 2, y + cell / 2 + 5, "&#215;", 15, "#bbb", "middle", "bold")
            else:
                s += box(x, y, cell, cell, "#fdf0ee", RED, 3, 1.2)
                s += txt(x + cell / 2, y + cell / 2 + 5, "&#215;", 15, RED, "middle", "bold")

    bot = gy + n * cell

    # grey callout: cross-sequence cell (row 3 = seq 2 query, col 5 = seq 1 key)
    s += line(gx + n * cell + 3, gy + 3 * cell + cell * 0.7, W - 214, 292, "#bbb", 1.2)
    s += txt(W - 208, 288, "a different conversation,", 11, "#999", "start")
    s += txt(W - 208, 303, "never visible", 11, "#999", "start")

    # red callout: same seq, future token (row 1 seq1 pos17, col 2 seq1 pos18)
    ci, cj = 1, 2
    s += line(gx + cj * cell + cell, gy + ci * cell + 10, W - 214, 150, RED, 1.2)
    s += txt(W - 208, 146, "same conversation,", 11, RED, "start")
    s += txt(W - 208, 161, "but a future position", 11, RED, "start")

    # predicate box
    py = bot + 34
    s += box(160, py, 540, 66, "#fafafa", "#ddd", 6, 1.2)
    s += mono(430, py + 27, "keep  =  same sequence   AND   key position &#8804; query position", 12.5, DARK, "middle", "bold")
    s += txt(430, py + 50, "the first half separates the conversations, the second half keeps things causal",
             11, "#777")

    s += txt(W / 2, H - 16,
             "Every cell outside the coloured triangles is set to -inf before the softmax.",
             12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "mask.svg"), "w").write(s)


# ----------------------------------------------------------------------------
# Diagram 6: chunked prefill, peak vs total
# ----------------------------------------------------------------------------
def d6():
    W, H = 800, 430
    s = head(W, H)
    s += title(W / 2, 26, "Chunked prefill: the same total work, a much smaller live piece")

    size = 250
    ly = 90

    # left: whole triangle at once
    lx = 90
    s += txt(lx + size / 2, ly - 22, "all at once", 12.5, DARK, "middle", "bold")
    s += box(lx, ly, size, size, "#ffffff", "#ccc", 3, 1.2)
    s += (f'  <path d="M {lx} {ly} L {lx + size} {ly + size} L {lx} {ly + size} Z" '
          f'fill="{TINT[BLUE]}" stroke="{BLUE}" stroke-width="1.4"/>\n')
    s += txt(lx + size * 0.32, ly + size * 0.72, "live", 13, BLUE, "middle", "bold")
    s += txt(lx + size / 2, ly + size + 24, "peak working set: the whole triangle", 11.5, "#777")

    # right: strips
    rx = 450
    s += txt(rx + size / 2, ly - 22, "in chunks of U", 12.5, DARK, "middle", "bold")
    s += box(rx, ly, size, size, "#ffffff", "#ccc", 3, 1.2)
    nstrip = 5
    sh = size / nstrip
    live = 3
    for k in range(nstrip):
        y = ly + k * sh
        w = size * (k + 1) / nstrip
        if k < live:
            s += (f'  <path d="M {rx} {y} L {rx + w} {y + sh} L {rx} {y + sh} Z" '
                  f'fill="#eeeeee" stroke="#ccc" stroke-width="1"/>\n')
            s += (f'  <rect x="{rx}" y="{y}" width="{w - sh}" height="{sh}" '
                  f'fill="#eeeeee" stroke="#ccc" stroke-width="1"/>\n')
        elif k == live:
            s += (f'  <path d="M {rx} {y} L {rx + w} {y + sh} L {rx} {y + sh} Z" '
                  f'fill="{TINT[RED]}" stroke="{RED}" stroke-width="1.4"/>\n')
            s += (f'  <rect x="{rx}" y="{y}" width="{w - sh}" height="{sh}" '
                  f'fill="{TINT[RED]}" stroke="{RED}" stroke-width="1.4"/>\n')
        else:
            s += (f'  <path d="M {rx} {y} L {rx + w} {y + sh} L {rx} {y + sh} Z" '
                  f'fill="#ffffff" stroke="#ddd" stroke-width="1" stroke-dasharray="3 3"/>\n')
            s += (f'  <rect x="{rx}" y="{y}" width="{w - sh}" height="{sh}" '
                  f'fill="#ffffff" stroke="#ddd" stroke-width="1" stroke-dasharray="3 3"/>\n')
    s += line(rx + size + 14, ly + live * sh, rx + size + 14, ly + (live + 1) * sh, RED, 2)
    s += txt(rx + size + 22, ly + live * sh + sh / 2 - 6, "U rows,", 11, RED, "start", "bold")
    s += txt(rx + size + 22, ly + live * sh + sh / 2 + 9, "live now", 11, RED, "start", "bold")
    s += txt(rx + 46, ly + 30, "done, now in", 10.5, "#999", "start")
    s += txt(rx + 46, ly + 44, "the KV cache", 10.5, "#999", "start")
    s += txt(rx + size / 2, ly + size + 24, "peak working set: one strip", 11.5, "#777")

    s += txt(W / 2, H - 40,
             "The shaded area is the same in both pictures, so the arithmetic is roughly the same.",
             12, "#555")
    s += txt(W / 2, H - 22,
             "What changes is how much of it has to be in flight at any one moment.", 12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "chunked-prefill.svg"), "w").write(s)


# ----------------------------------------------------------------------------
# Diagram 7: the MLP, shared weights, independent rows
# ----------------------------------------------------------------------------
def d7():
    W, H = 800, 420
    s = head(W, H)
    s += title(W / 2, 26, "The feed-forward block: one set of weights, rows that never meet")

    cols = [BLUE, GREEN, GREEN, ORANGE]
    labels = ["seq 0", "seq 1", "seq 1", "seq 2"]
    rh = 40
    top = 100
    inx = 70
    inw = 150

    s += txt(inx + inw / 2, top - 24, "input rows", 12.5, DARK, "middle", "bold")
    s += mono(inx + inw / 2, top - 8, "[4, d]", 11, "#888")
    for i, c in enumerate(cols):
        y = top + i * (rh + 8)
        s += box(inx, y, inw, rh, TINT[c], c, 4, 1.5)
        s += mono(inx + inw / 2, y + 25, labels[i], 11.5, c, "middle", "bold")

    # weight block
    wx = 340
    wy = top - 4
    wh = 4 * (rh + 8) - 4
    outx = 580
    outw = 150

    # straight horizontal paths, drawn first so the W block sits on top
    for i, c in enumerate(cols):
        y = top + i * (rh + 8) + rh / 2
        s += line(inx + inw + 4, y, outx - 8, y, c, 1.3, dash="4 3", marker="ar" if c == BLUE else None)

    s += box(wx, wy, 110, wh, "#f2f2f2", DARK, 4, 1.6)
    s += txt(wx + 55, wy + wh / 2 - 6, "W", 20, DARK, "middle", "bold")
    s += mono(wx + 55, wy + wh / 2 + 16, "[d, f]", 11, "#777")
    s += txt(wx + 55, wy + wh + 22, "one weight matrix,", 11, "#777")
    s += txt(wx + 55, wy + wh + 37, "applied to every row", 11, "#777")

    # output rows
    s += txt(outx + outw / 2, top - 24, "output rows", 12.5, DARK, "middle", "bold")
    s += mono(outx + outw / 2, top - 8, "[4, f]", 11, "#888")
    for i, c in enumerate(cols):
        y = top + i * (rh + 8)
        s += box(outx, y, outw, rh, TINT[c], c, 4, 1.5)
        s += mono(outx + outw / 2, y + 25, labels[i], 11.5, c, "middle", "bold")

    s += txt(W / 2, H - 58,
             "Row 2 of the output is built only from row 2 of the input. Nothing here can carry a",
             12, "#555")
    s += txt(W / 2, H - 40,
             "value from one conversation into another, no matter how the rows are arranged.",
             12, "#555")
    s += txt(W / 2, H - 16,
             "This is also where batching pays off: the weights are fetched once and shared by all four rows.",
             11.5, "#888")
    s += "</svg>\n"
    open(os.path.join(OUT, "mlp-rows.svg"), "w").write(s)


d4()
d5()
d6()
d7()
print("wrote diagrams 4-7")
