#!/usr/bin/env python3
"""Diagrams 8-9: the output projection, and the per-layer audit."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "ubatch")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c5f8a"
GREEN = "#2e7d52"
ORANGE = "#b8681c"
RED = "#c0392b"
PURPLE = "#6b4c9a"
GREY = "#888"
DARK = "#333"
FONT = 'font-family="Helvetica, Arial, sans-serif"'
TINT = {BLUE: "#e8eef4", GREEN: "#e9f2ec", ORANGE: "#fdf3e8", RED: "#fbeeea", PURPLE: "#efe9f6"}


def head(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" {FONT}>\n'
            f'  <defs>\n'
            f'    <marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">\n'
            f'      <polygon points="0 0, 9 3, 0 6" fill="{DARK}"/>\n'
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


def box(x, y, w, h, fill="#ffffff", stroke=GREY, rx=4, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def line(x1, y1, x2, y2, stroke=GREY, sw=1.4, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n'


# ----------------------------------------------------------------------------
# Diagram 8: W_O mixes heads, not tokens
# ----------------------------------------------------------------------------
def d8():
    W, H = 820, 660
    s = head(W, H)
    s += title(W / 2, 26, "The output projection mixes heads, not tokens")

    # ---------------- panel A: one row ----------------
    s += txt(W / 2, 58, "A.  what happens inside a single row", 13, DARK, "middle", "bold")

    hw = 74
    hh = 40
    hx = 150
    hy = 82
    heads = ["head 0", "head 1", "head 2", "head 3"]
    for i, h in enumerate(heads):
        x = hx + i * (hw + 6)
        s += box(x, hy, hw, hh, TINT[BLUE], BLUE, 4, 1.4)
        s += txt(x + hw / 2, hy + 18, h, 10.5, BLUE, "middle", "bold")
        s += mono(x + hw / 2, hy + 32, "d_head", 9, "#8aa")
    span = 4 * (hw + 6) - 6
    s += txt(hx - 12, hy + 24, "token i", 11.5, BLUE, "end", "bold")
    # bracket
    by = hy + hh + 8
    s += (f'  <path d="M {hx} {by} L {hx} {by + 7} L {hx + span} {by + 7} L {hx + span} {by}" '
          f'fill="none" stroke="{GREY}" stroke-width="1.3"/>\n')
    s += txt(hx + span / 2, by + 24, "concatenated along the feature axis", 11, "#777")

    s += line(hx + span / 2, by + 32, hx + span / 2, by + 54, DARK, 1.4, marker="ar")
    wy = by + 58
    s += box(hx + span / 2 - 55, wy, 110, 34, "#f2f2f2", DARK, 4, 1.5)
    s += txt(hx + span / 2, wy + 22, "W_O", 14, DARK, "middle", "bold")
    s += line(hx + span / 2, wy + 40, hx + span / 2, wy + 60, DARK, 1.4, marker="ar")
    oy = wy + 64
    s += box(hx + span / 2 - 55, oy, 110, 34, TINT[BLUE], BLUE, 4, 1.5)
    s += txt(hx + span / 2, oy + 22, "token i out", 11, BLUE, "middle", "bold")

    s += txt(hx + span + 40, wy + 12, "the four heads for token i are", 11, "#777", "start")
    s += txt(hx + span + 40, wy + 27, "blended together here, which is", 11, "#777", "start")
    s += txt(hx + span + 40, wy + 42, "the whole point of this layer", 11, "#777", "start")

    # ---------------- panel B: full matrix ----------------
    py = 330
    s += txt(W / 2, py - 12, "B.  what happens across rows", 13, DARK, "middle", "bold")

    cols = [BLUE, GREEN, ORANGE, GREEN]
    labels = ["seq 0", "seq 1", "seq 2", "seq 1"]
    rh = 34
    rx0 = 110
    inw = 220
    top = py + 18
    s += txt(rx0 + inw / 2, top - 6, "concatenated heads", 11, "#777")
    s += mono(rx0 + inw / 2, top + 8, "[rows, n_head &#215; d_head]", 10, "#999")
    for i, c in enumerate(cols):
        y = top + 18 + i * (rh + 6)
        s += box(rx0, y, inw, rh, TINT[c], c, 4, 1.4)
        # sub-blocks for heads
        for k in range(4):
            s += line(rx0 + (k + 1) * inw / 4, y, rx0 + (k + 1) * inw / 4, y + rh, c, 0.7, dash="2 2")
        s += mono(rx0 + inw / 2, y + 22, labels[i], 11, c, "middle", "bold")

    wox = 400
    woy = top + 22
    woh = 4 * (rh + 6) - 14
    s += box(wox, woy, 76, woh, "#f2f2f2", DARK, 4, 1.5)
    s += txt(wox + 38, woy + woh / 2 + 5, "W_O", 13, DARK, "middle", "bold")

    outx = 560
    outw = 150
    s += txt(outx + outw / 2, top - 6, "layer output", 11, "#777")
    s += mono(outx + outw / 2, top + 8, "[rows, d]", 10, "#999")
    for i, c in enumerate(cols):
        y = top + 18 + i * (rh + 6)
        s += line(rx0 + inw + 4, y + rh / 2, outx - 8, y + rh / 2, c, 1.2, dash="4 3")
        s += box(outx, y, outw, rh, TINT[c], c, 4, 1.4)
        s += mono(outx + outw / 2, y + 22, labels[i], 11, c, "middle", "bold")

    # forbidden arrow: row 0 -> row 2 output
    y0 = top + 18 + 0 * (rh + 6) + rh / 2
    y2 = top + 18 + 2 * (rh + 6) + rh / 2
    s += (f'  <path d="M {rx0 + inw + 10} {y0} C {480} {y0}, {480} {y2}, {outx - 10} {y2}" '
          f'fill="none" stroke="{RED}" stroke-width="1.6" stroke-dasharray="5 4"/>\n')
    cxx, cyy = 512, (y0 + y2) / 2
    s += line(cxx - 12, cyy - 12, cxx + 12, cyy + 12, RED, 3)
    s += line(cxx + 12, cyy - 12, cxx - 12, cyy + 12, RED, 3)

    rowbot = top + 18 + 4 * (rh + 6)
    s += line(cxx, cyy + 16, cxx, rowbot + 12, RED, 1.1, dash="3 3")
    s += txt(cxx - 8, rowbot + 26, "there is no path from one row into another row's output:",
             11, RED, "end", "bold")
    s += txt(cxx - 8, rowbot + 42, "the matmul contracts the feature axis, and leaves rows alone",
             11, RED, "end")

    s += txt(W / 2, H - 40,
             "So heads do get combined, but only within a row. The row index is carried straight",
             12, "#555")
    s += txt(W / 2, H - 22,
             "through, which is why a token from one conversation cannot land in another's output.",
             12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "output-projection.svg"), "w").write(s)


# ----------------------------------------------------------------------------
# Diagram 9: layer audit
# ----------------------------------------------------------------------------
def d9():
    W, H = 780, 640
    s = head(W, H)
    s += title(W / 2, 26, "One transformer layer, with the row-crossing step marked")

    steps = [
        ("input hidden states", "[rows, d]", "row-wise", GREEN),
        ("RMSNorm", "normalises within each row", "row-wise", GREEN),
        ("Q, K, V projections", "one matmul, applied per row", "row-wise", GREEN),
        ("RoPE", "uses each token's own position", "row-wise", GREEN),
        ("softmax(QK&#7488;) V", "rows read other rows, under the mask", "crosses rows", RED),
        ("output projection W_O", "mixes heads within a row", "row-wise", GREEN),
        ("residual add", "elementwise", "row-wise", GREEN),
        ("RMSNorm", "within each row again", "row-wise", GREEN),
        ("feed-forward, up / gate / down", "shared weights, per row", "row-wise", GREEN),
        ("residual add", "elementwise", "row-wise", GREEN),
    ]

    bw = 420
    bh = 40
    bx = 130
    top = 62
    gap = 10
    for i, (name, sub, tag, col) in enumerate(steps):
        y = top + i * (bh + gap)
        sw = 2.0 if col == RED else 1.3
        s += box(bx, y, bw, bh, TINT[col], col, 5, sw)
        s += txt(bx + 14, y + 18, name, 12, col, "start", "bold")
        s += txt(bx + 14, y + 32, sub, 10.5, "#777", "start")
        s += txt(bx + bw - 12, y + 25, tag, 10.5, col, "end", "bold")
        if i < len(steps) - 1:
            s += line(bx + bw / 2, y + bh, bx + bw / 2, y + bh + gap, "#ccc", 1.2)

    # mask callout on the attention row
    ay = top + 4 * (bh + gap)
    s += line(bx + bw + 8, ay + bh / 2, bx + bw + 62, ay + bh / 2, RED, 1.4, marker="arRed")
    s += box(bx + bw + 66, ay - 12, 118, bh + 24, "#fdf0ee", RED, 5, 1.4)
    s += txt(bx + bw + 125, ay + 6, "the mask", 11.5, RED, "middle", "bold")
    s += txt(bx + bw + 125, ay + 22, "sits here", 11.5, RED)
    s += txt(bx + bw + 125, ay + 40, "and only here", 10.5, "#a55")

    bot = top + len(steps) * (bh + gap)
    s += txt(W / 2, bot + 24,
             "Nine of the ten steps treat the rows as independent. One does not, and that one",
             12, "#555")
    s += txt(W / 2, bot + 42,
             "is the step that carries the mask.", 12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "layer-audit.svg"), "w").write(s)


d8()
d9()
print("wrote diagrams 8-9")
