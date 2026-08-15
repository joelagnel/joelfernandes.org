#!/usr/bin/env python3
"""Diagrams 1-3 for the ubatch / continuous batching post."""
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
            f'  </defs>\n'
            f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n')


def title(x, y, t, size=16):
    return (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
            f'font-weight="bold" fill="{DARK}">{t}</text>\n')


def txt(x, y, t, size=12, fill="#555", anchor="middle", weight="normal", style="normal"):
    return (f'  <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" font-style="{style}">{t}</text>\n')


def box(x, y, w, h, fill="#ffffff", stroke=GREY, rx=4, sw=1.4, op=1.0):
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"/>\n')


def line(x1, y1, x2, y2, stroke=GREY, sw=1.4, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n'


# ----------------------------------------------------------------------------
# Diagram 1: the word "batch" in two settings
# ----------------------------------------------------------------------------
def d1():
    W, H = 800, 400
    s = head(W, H)
    s += title(W / 2, 26, "The same tensor shape, two different meanings of &#8220;batch&#8221;")

    # ---- left panel: training ----
    lx = 40
    s += txt(lx + 155, 60, "Training", 14, DARK, "middle", "bold")
    s += txt(lx + 155, 78, "rows are independent examples", 11.5, "#666")

    cell = 44
    ch = 30
    toks = [["A", "B", "C", "D"], ["F", "G", "H", "I"], ["K", "L", "M", "N"], ["P", "Q", "R", "S"]]
    top = 96
    for r, row in enumerate(toks):
        y = top + r * (ch + 8)
        for c, t in enumerate(row):
            x = lx + 40 + c * (cell + 4)
            s += box(x, y, cell, ch, "#f4f7fa", BLUE, 3, 1.2)
            s += txt(x + cell / 2, y + 20, t, 13, BLUE, "middle", "bold")
        s += txt(lx + 32, y + 20, f"ex {r}", 10.5, "#777", "end")

    bot = top + 4 * (ch + 8)
    # batch bracket (vertical, left)
    bx = lx + 18
    s += (f'  <path d="M {bx} {top} L {bx - 8} {top} L {bx - 8} {bot - 8} L {bx} {bot - 8}" '
          f'fill="none" stroke="{BLUE}" stroke-width="1.6"/>\n')
    s += (f'  <text x="{bx - 14}" y="{(top + bot) / 2}" text-anchor="middle" font-size="11.5" '
          f'fill="{BLUE}" font-weight="bold" transform="rotate(-90 {bx - 14} {(top + bot) / 2})">'
          f'batch = 4</text>\n')
    # sequence arrow
    s += line(lx + 40, bot + 12, lx + 40 + 4 * (cell + 4) - 4, bot + 12, "#aaa", 1.2, marker="arGrey")
    s += txt(lx + 40 + 2 * (cell + 4), bot + 30, "sequence length", 11, "#777")

    s += txt(lx + 155, bot + 58, "shape [4, T, d]", 12, "#555", "middle", "bold")

    # divider
    s += line(W / 2, 55, W / 2, H - 30, "#ddd", 1.2, dash="4 4")

    # ---- right panel: llama.cpp ----
    rx = W / 2 + 40
    s += txt(rx + 150, 60, "llama.cpp inference", 14, DARK, "middle", "bold")
    s += txt(rx + 150, 78, "rows are token slots, tagged with a sequence", 11.5, "#666")

    slots = [("A", "seq 0", "pos 0", BLUE),
             ("B", "seq 0", "pos 1", BLUE),
             ("F", "seq 1", "pos 40", GREEN),
             ("K", "seq 2", "pos 7", ORANGE)]
    for r, (t, sq, ps, col) in enumerate(slots):
        y = top + r * (ch + 8)
        s += box(rx + 40, y, cell, ch, "#ffffff", col, 3, 1.4)
        s += txt(rx + 40 + cell / 2, y + 20, t, 13, col, "middle", "bold")
        s += txt(rx + 40 + cell + 14, y + 20, f"{sq}", 11, col, "start")
        s += txt(rx + 40 + cell + 74, y + 20, f"{ps}", 11, "#777", "start")
        s += txt(rx + 32, y + 20, f"slot {r}", 10.5, "#777", "end")

    bx2 = rx + 18
    s += (f'  <path d="M {bx2} {top} L {bx2 - 8} {top} L {bx2 - 8} {bot - 8} L {bx2} {bot - 8}" '
          f'fill="none" stroke="{DARK}" stroke-width="1.6"/>\n')
    s += (f'  <text x="{bx2 - 14}" y="{(top + bot) / 2}" text-anchor="middle" font-size="11.5" '
          f'fill="{DARK}" font-weight="bold" transform="rotate(-90 {bx2 - 14} {(top + bot) / 2})">'
          f'n_tokens = 4</text>\n')

    s += txt(rx + 150, bot + 30, "four token positions, three different conversations", 11, "#777")
    s += txt(rx + 150, bot + 58, "shape [4, d]", 12, "#555", "middle", "bold")

    s += txt(W / 2, H - 12, "Both are 4 rows. Only the left one means &#8220;4 independent examples&#8221;.",
             12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "batch-two-meanings.svg"), "w").write(s)


# ----------------------------------------------------------------------------
# Diagram 2: which knob moves which slab of VRAM
# ----------------------------------------------------------------------------
def d2():
    W, H = 800, 540
    s = head(W, H)
    s += title(W / 2, 26, "Which knob moves which part of VRAM")

    barw = 170
    base = 400

    def bar(x, label, sub, kv_h, ws_h, wt_h=120):
        out = ""
        y = base
        y -= wt_h
        out += box(x, y, barw, wt_h, "#e8eef4", BLUE, 4, 1.4)
        out += txt(x + barw / 2, y + wt_h / 2 + 5, "model weights", 12.5, BLUE, "middle", "bold")
        y -= kv_h
        out += box(x, y, barw, kv_h, "#e9f2ec", GREEN, 4, 1.4)
        out += txt(x + barw / 2, y + kv_h / 2 + 5, "KV cache", 12.5, GREEN, "middle", "bold")
        y -= ws_h
        out += box(x, y, barw, ws_h, "#fbeeea", RED, 4, 1.4)
        if ws_h > 26:
            out += txt(x + barw / 2, y + ws_h / 2 + 5, "compute workspace", 11.5, RED, "middle", "bold")
        else:
            out += txt(x + barw / 2, y + ws_h / 2 + 4, "workspace", 10.5, RED, "middle", "bold")
        out += txt(x + barw / 2, base + 26, label, 12.5, DARK, "middle", "bold")
        out += txt(x + barw / 2, base + 44, sub, 11, "#777")
        return out, y

    x1 = 220
    x2 = 470
    b1, top1 = bar(x1, "-ub 512", "one active sequence", kv_h=110, ws_h=118)
    b2, top2 = bar(x2, "-ub 128", "same -c, smaller chunk", kv_h=110, ws_h=34)
    s += b1 + b2

    s += line(x1 + barw + 8, top1 + 10, x2 - 10, top2 + 6, RED, 1.6, dash="5 4", marker="arRed")
    s += txt((x1 + barw + x2) / 2, top1 - 6, "-ub moves this", 11.5, RED, "middle", "bold")

    # left-hand knob callouts, well inside the canvas
    cx = x1 - 22
    s += txt(cx, base - 120 + 60, "unchanged by any of", 11, BLUE, "end")
    s += txt(cx, base - 120 + 76, "these three knobs", 11, BLUE, "end")
    s += line(cx + 6, base - 120 + 66, x1 - 4, base - 120 + 60, BLUE, 1.2)

    s += txt(cx, base - 120 - 55 + 8, "-c, and the KV", 11, GREEN, "end")
    s += txt(cx, base - 120 - 55 + 24, "cache precision", 11, GREEN, "end")
    s += line(cx + 6, base - 120 - 55 + 14, x1 - 4, base - 120 - 55 + 10, GREEN, 1.2)

    s += txt(W / 2, H - 46,
             "Lowering -ub shrinks the reserved workspace. It leaves the KV cache alone,",
             12, "#555")
    s += txt(W / 2, H - 28,
             "so it is not the knob to reach for when the context itself is what does not fit.",
             12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "vram-slabs.svg"), "w").write(s)


# ----------------------------------------------------------------------------
# Diagram 3: prefill vs decode
# ----------------------------------------------------------------------------
def d3():
    W, H = 800, 350
    s = head(W, H)
    s += title(W / 2, 26, "Prefill and decode do very different amounts of work per pass")

    y0 = 80
    s += txt(60, y0 - 16, "Prefill: the tokens are already known", 12.5, BLUE, "start", "bold")
    x = 60
    for i in range(4):
        s += box(x, y0, 88, 52, "#e8eef4", BLUE, 4, 1.4)
        s += txt(x + 44, y0 + 24, f"{i * 128}&#8211;{i * 128 + 127}", 11.5, BLUE)
        s += txt(x + 44, y0 + 41, "128 tokens", 10, "#777")
        x += 96
    s += txt(x + 22, y0 + 30, "&#8230;", 16, "#999")
    s += txt(60, y0 + 78, "one forward pass per chunk, up to n_ubatch token positions at a time",
             11.5, "#777", "start")

    y1 = 205
    s += txt(60, y1 - 16, "Decode: each next token has to be sampled before the one after it exists",
             12.5, ORANGE, "start", "bold")
    x = 60
    for i in range(14):
        s += box(x, y1, 18, 52, "#fdf3e8", ORANGE, 3, 1.3)
        x += 26
    s += txt(x + 18, y1 + 32, "&#8230;", 16, "#999")
    s += txt(60, y1 + 78, "one forward pass per token, and with a single sequence that pass has one row",
             11.5, "#777", "start")

    s += txt(W / 2, H - 14,
             "n_ubatch is the ceiling on the left. On the right it is mostly unused capacity.",
             12, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "prefill-vs-decode.svg"), "w").write(s)


d1()
d2()
d3()
print("wrote diagrams 1-3 to", os.path.normpath(OUT))
