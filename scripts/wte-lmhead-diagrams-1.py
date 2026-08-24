#!/usr/bin/env python3
"""Diagrams 1-6 for the wte / lm_head weight tying post."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
c = SourceFileLoader("c", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "wte-lmhead-common.py")).load_module()
from c import (head, title, txt, box, line, path, circle, save,
               BLUE, GREEN, ORANGE, RED, PURPLE, GREY, LGREY, DARK)


# ---------------------------------------------------------------------------
# Diagram 1: the two doors. what goes in and what comes out of each layer
# ---------------------------------------------------------------------------
def d1():
    W, H = 820, 340
    s = head(W, H)
    s += title(W / 2, 26, "The two ends of the model")

    # left panel: wte
    lx = 40
    s += box(lx, 50, 340, 250, "#f7fafc", BLUE, sw=1.6)
    s += txt(lx + 170, 76, "wte", 17, BLUE, weight="bold", mono=True)
    s += txt(lx + 170, 96, "nn.Embedding(50257, 768)", 11.5, "#666", mono=True)

    s += box(lx + 95, 116, 150, 34, "#ffffff", GREY)
    s += txt(lx + 170, 138, "token id: 6766", 12.5, DARK, mono=True)
    s += txt(lx + 170, 166, "one integer", 11, "#777", style="italic")

    s += line(lx + 170, 176, lx + 170, 208, BLUE, 1.8, marker="ar")
    s += txt(lx + 250, 196, "row lookup", 11, BLUE)

    s += box(lx + 60, 216, 220, 34, "#ffffff", GREEN)
    s += txt(lx + 170, 238, "768 numbers", 12.5, DARK, mono=True)
    s += txt(lx + 170, 268, "a vector in embedding space", 11, "#777", style="italic")
    s += txt(lx + 170, 288, "shape (768,)", 11, "#777", mono=True)

    # right panel: lm_head
    rx = 440
    s += box(rx, 50, 340, 250, "#fffaf5", ORANGE, sw=1.6)
    s += txt(rx + 170, 76, "lm_head", 17, ORANGE, weight="bold", mono=True)
    s += txt(rx + 170, 96, "nn.Linear(768, 50257, bias=False)", 11.5, "#666", mono=True)

    s += box(rx + 60, 116, 220, 34, "#ffffff", GREEN)
    s += txt(rx + 170, 138, "768 numbers", 12.5, DARK, mono=True)
    s += txt(rx + 170, 166, "a vector in embedding space", 11, "#777", style="italic")

    s += line(rx + 170, 176, rx + 170, 208, ORANGE, 1.8, marker="arOrange")
    s += txt(rx + 253, 196, "score every row", 11, ORANGE)

    s += box(rx + 60, 216, 220, 34, "#ffffff", GREY)
    s += txt(rx + 170, 238, "50257 numbers", 12.5, DARK, mono=True)
    s += txt(rx + 170, 268, "one logit per vocabulary entry", 11, "#777", style="italic")
    s += txt(rx + 170, 288, "shape (50257,)", 11, "#777", mono=True)

    s += txt(W / 2, 322, "Same matrix behind both panels. Opposite directions.", 12, "#666")
    save("two-ends.svg", s)


# ---------------------------------------------------------------------------
# Diagram 2: forward pass column with shapes, tying line
# ---------------------------------------------------------------------------
def d2():
    W, H = 760, 560
    s = head(W, H)
    s += title(W / 2, 26, "The forward pass, with the real shapes of a 2 x 3 batch")

    cx = 300
    bw, bh = 300, 40

    def stage(y, label, sub, fill, stroke):
        t = box(cx - bw / 2, y, bw, bh, fill, stroke, sw=1.6)
        t += txt(cx, y + 19, label, 13, DARK, weight="bold", mono=True)
        t += txt(cx, y + 33, sub, 10, "#777")
        return t

    def shape(y, sh):
        return txt(cx, y, sh, 11.5, BLUE, mono=True)

    def arrow(y1, y2):
        return line(cx, y1, cx, y2, GREY, 1.5, marker="arGrey")

    y = 56
    s += shape(y, "idx  (2, 3)  int64")
    s += txt(cx, y + 15, "2 sequences, 3 token ids each", 10, "#888")
    s += arrow(y + 24, y + 46)

    y = 108
    s += stage(y, "wte(idx)", "look up one row per id", "#f7fafc", BLUE)
    wte_y = y
    y += bh + 4
    s += shape(y + 12, "tok_emb  (2, 3, 768)")
    s += arrow(y + 20, y + 42)

    y += 54
    s += stage(y, "+ wpe(pos)", "add position rows, (3, 768)", "#f7fafc", GREY)
    y += bh + 4
    s += shape(y + 12, "x  (2, 3, 768)")
    s += arrow(y + 20, y + 42)

    y += 54
    s += box(cx - bw / 2, y, bw, 56, "#f6f6f6", GREY, sw=1.6, dash="5,4")
    s += txt(cx, y + 22, "12 x Block", 13, DARK, weight="bold", mono=True)
    s += txt(cx, y + 38, "shape never changes", 10, "#777")
    y += 62
    s += shape(y + 12, "x  (2, 3, 768)")
    s += arrow(y + 20, y + 42)

    y += 54
    s += stage(y, "lm_head(x)", "score x against all 50257 rows", "#fffaf5", ORANGE)
    lm_y = y
    y += bh + 4
    s += shape(y + 12, "logits  (2, 3, 50257)")
    s += arrow(y + 20, y + 42)

    y += 54
    s += stage(y, "softmax", "one probability per vocabulary entry", "#ffffff", RED)

    lx = cx - bw / 2 - 44
    s += path(f"M {cx - bw/2} {wte_y + 20} L {lx} {wte_y + 20} L {lx} {lm_y + 20} "
              f"L {cx - bw/2} {lm_y + 20}", PURPLE, 1.8, dash="6,4")
    mid = (wte_y + lm_y) / 2 + 20
    s += txt(lx - 8, mid - 14, "the same", 11.5, PURPLE, anchor="end", weight="bold")
    s += txt(lx - 8, mid, "50257 x 768", 11.5, PURPLE, anchor="end", mono=True)
    s += txt(lx - 8, mid + 14, "tensor", 11.5, PURPLE, anchor="end", weight="bold")

    save("forward-pass.svg", s)


# ---------------------------------------------------------------------------
# Diagram 3: the wte grid, one row pulled out
# ---------------------------------------------------------------------------
def d3():
    W, H = 780, 400
    s = head(W, H)
    s += title(W / 2, 26, "wte is one table: 50257 rows, 768 columns")

    gx, gy, gw, gh = 70, 60, 300, 280
    s += box(gx, gy, gw, gh, "#ffffff", DARK, sw=1.8)

    rows = 14
    rh = gh / rows
    hi = 6
    for i in range(rows):
        yy = gy + i * rh
        if i == hi:
            s += box(gx, yy, gw, rh, "#e8f0f7", BLUE, sw=1.8, rx=0)
        elif i in (0, rows - 1):
            pass
        else:
            s += line(gx, yy, gx + gw, yy, LGREY, 0.8)
    s += line(gx, gy + hi * rh, gx + gw, gy + hi * rh, BLUE, 1.6)
    s += line(gx, gy + (hi + 1) * rh, gx + gw, gy + (hi + 1) * rh, BLUE, 1.6)

    s += txt(gx - 10, gy + 14, "row 0", 11, "#777", anchor="end", mono=True)
    s += txt(gx - 10, gy + hi * rh + 14, "row 4171", 11, BLUE, anchor="end", mono=True, weight="bold")
    s += txt(gx - 10, gy + gh - 4, "row 50256", 11, "#777", anchor="end", mono=True)
    s += txt(gx + gw / 2, gy + gh + 22, "768 columns", 12, "#666")
    s += txt(gx + gw / 2, gy + gh + 38, "one learned coordinate each", 10.5, "#888", style="italic")

    # pulled-out row
    px, py, pw, ph = 450, 150, 270, 34
    s += path(f"M {gx + gw} {gy + hi * rh + rh/2} C {gx + gw + 50} {gy + hi*rh}, "
              f"{px - 50} {py + ph/2}, {px} {py + ph/2}", BLUE, 1.8, marker="ar")

    cells = 8
    cw = pw / cells
    vals = ["0.021", "-0.114", "0.008", "0.076", "...", "-0.033", "0.052", "0.019"]
    for i in range(cells):
        s += box(px + i * cw, py, cw, ph, "#e8f0f7", BLUE, rx=0, sw=1.0)
        s += txt(px + i * cw + cw / 2, py + 22, vals[i], 8.5, DARK, mono=True)
    s += txt(px + pw / 2, py - 12, 'wte.weight[4171]   the row for " blue"', 12, BLUE,
             weight="bold", mono=True)
    s += txt(px + pw / 2, py + ph + 20, "768 floats. This is what the lookup returns,", 11, "#666")
    s += txt(px + pw / 2, py + ph + 36, "and what the matmul scores against.", 11, "#666")

    s += txt(px + pw / 2, py + ph + 70, "50257 x 768 = 38,597,376 parameters", 12, DARK,
             weight="bold", mono=True)
    s += txt(px + pw / 2, py + ph + 88, "154.4 MB in fp32", 11, "#777")

    save("wte-grid.svg", s)


# ---------------------------------------------------------------------------
# Diagram 4: neighbourhood sketch
# ---------------------------------------------------------------------------
def d4():
    W, H = 720, 400
    s = head(W, H)
    s += title(W / 2, 26, "Rows sit somewhere. Nearby rows behave alike.")

    s += box(60, 58, 600, 282, "#fcfcfc", LGREY)
    s += txt(360, 364, "A 2D sketch of a 768-dimensional space. Positions are illustrative.",
             11, "#888", style="italic")

    groups = [
        (185, 142, GREEN, "fruit", [(" apple", -38, -22), (" orange", 30, -34), (" banana", -6, 26)]),
        (400, 232, ORANGE, "colour", [(" blue", -34, 18), (" red", 34, 6), (" green", -4, -30)]),
        (560, 132, PURPLE, "animal", [(" cat", -30, 14), (" dog", 28, -8)]),
    ]
    for cx, cy, col, label, members in groups:
        s += circle(cx, cy, 62, col, op=0.08)
        s += f'  <circle cx="{cx}" cy="{cy}" r="62" fill="none" stroke="{col}" stroke-width="1.2" stroke-dasharray="5,4"/>\n'
        s += txt(cx, cy - 74, label, 12, col, weight="bold")
        for name, dx, dy in members:
            s += circle(cx + dx, cy + dy, 4.5, col)
            s += txt(cx + dx, cy + dy - 10, name, 10.5, DARK, mono=True)

    save("neighbourhood.svg", s)


# ---------------------------------------------------------------------------
# Diagram 5: one-hot times wte
# ---------------------------------------------------------------------------
def d5():
    W, H = 800, 330
    s = head(W, H)
    s += title(W / 2, 26, "The lookup, written as the matrix multiply it stands for")

    # one-hot row vector
    ox, oy = 40, 130
    cells = 9
    cw, ch = 34, 30
    hot = 4
    for i in range(cells):
        f = "#e8f0f7" if i == hot else "#ffffff"
        st = BLUE if i == hot else LGREY
        s += box(ox + i * cw, oy, cw, ch, f, st, rx=0, sw=1.4 if i == hot else 0.9)
        s += txt(ox + i * cw + cw / 2, oy + 20, "1" if i == hot else "0", 12,
                 BLUE if i == hot else "#aaa", mono=True,
                 weight="bold" if i == hot else "normal")
    s += txt(ox + cells * cw / 2, oy - 14, "one-hot,  (1, 50257)", 11.5, DARK, mono=True)
    s += txt(ox + hot * cw + cw / 2, oy + ch + 18, "index 4171", 10, BLUE, mono=True)

    s += txt(ox + cells * cw + 22, oy + 20, "x", 16, DARK, weight="bold")

    # the table
    tx, ty = ox + cells * cw + 44, 60
    tw = 210
    rows = 9
    rh = 180 / rows
    s += box(tx, ty, tw, 180, "#ffffff", DARK, sw=1.6)
    for i in range(rows):
        yy = ty + i * rh
        if i == hot:
            s += box(tx, yy, tw, rh, "#e8f0f7", BLUE, rx=0, sw=1.6)
            s += txt(tx + tw / 2, yy + rh - 5, "0.021  -0.114  ...  0.019", 8.5, DARK, mono=True)
        else:
            s += line(tx, yy, tx + tw, yy, LGREY, 0.8)
            s += txt(tx + tw / 2, yy + rh - 5, "-  -  -  -  -  -", 8.5, "#ccc", mono=True)
    s += txt(tx + tw / 2, ty - 14, "wte.weight,  (50257, 768)", 11.5, DARK, mono=True)

    s += txt(tx + tw + 22, oy + 20, "=", 16, DARK, weight="bold")

    # result
    rx2 = tx + tw + 44
    outc = 6
    ow = 32
    for i in range(outc):
        s += box(rx2 + i * ow, oy, ow, ch, "#e8f0f7", BLUE, rx=0, sw=1.2)
    s += txt(rx2 + outc * ow / 2, oy + 20, "0.021 -0.114 ... 0.019", 8, DARK, mono=True)
    s += txt(rx2 + outc * ow / 2, oy - 14, "(1, 768)", 11.5, DARK, mono=True)
    s += txt(rx2 + outc * ow / 2, oy + ch + 18, "row 4171, unchanged", 10, BLUE)

    s += txt(W / 2, 268, "Every other row is multiplied by 0 and contributes nothing.", 12, "#555")
    s += txt(W / 2, 292, "Materialising these one-hots for a (4, 1024) batch in fp32 would take 823 GB,",
             11.5, RED)
    s += txt(W / 2, 310, "so the lookup is what runs. The result is identical.", 11.5, RED)

    save("one-hot-matmul.svg", s)


# ---------------------------------------------------------------------------
# Diagram 6: gather forward, scatter-add backward
# ---------------------------------------------------------------------------
def d6():
    W, H = 800, 380
    s = head(W, H)
    s += title(W / 2, 26, "Forward is a gather. Backward is a scatter-add.")

    def table(x, y, label, col):
        t = box(x, y, 130, 210, "#ffffff", DARK, sw=1.5)
        for i in range(7):
            t += line(x, y + i * 30, x + 130, y + i * 30, LGREY, 0.8)
        t += txt(x + 65, y - 10, label, 11.5, col, mono=True, weight="bold")
        return t

    # left: forward
    s += txt(200, 56, "forward", 13, GREEN, weight="bold")
    s += table(60, 90, "wte.weight", DARK)
    # highlight row 2
    s += box(60, 90 + 2 * 30, 130, 30, "#eaf5ee", GREEN, rx=0, sw=1.6)
    s += txt(125, 90 + 2 * 30 + 20, 'row 262  " the"', 9.5, DARK, mono=True)

    for i, yy in enumerate([120, 190, 260]):
        s += path(f"M 190 {90 + 2*30 + 15} C 240 {90 + 2*30 + 15}, 250 {yy + 15}, 290 {yy + 15}",
                  GREEN, 1.5, marker="arGreen")
        s += box(290, yy, 74, 30, "#eaf5ee", GREEN, rx=3, sw=1.2)
        s += txt(327, yy + 20, f"pos {[3, 17, 91][i]}", 10, DARK, mono=True)
    s += txt(327, 300, "three positions in the batch", 10.5, "#777")
    s += txt(327, 316, "hold token id 262", 10.5, "#777")
    s += txt(200, 344, "one row read, copied three times", 11, GREEN)

    # right: backward
    s += txt(620, 56, "backward", 13, RED, weight="bold")
    s += table(670, 90, "wte.weight.grad", DARK)
    s += box(670, 90 + 2 * 30, 130, 30, "#fdecea", RED, rx=0, sw=1.6)
    s += txt(735, 90 + 2 * 30 + 20, "row 262", 9.5, DARK, mono=True)

    for i, yy in enumerate([120, 190, 260]):
        s += path(f"M 570 {yy + 15} C 610 {yy + 15}, 620 {90 + 2*30 + 15}, 668 {90 + 2*30 + 15}",
                  RED, 1.5, marker="arRed")
        s += box(496, yy, 74, 30, "#fdecea", RED, rx=3, sw=1.2)
        s += txt(533, yy + 20, "grad", 10, DARK, mono=True)
    s += txt(533, 300, "three separate gradients", 10.5, "#777")
    s += txt(533, 316, "land on the same row", 10.5, "#777")
    s += txt(620, 344, "three writes added, so it needs atomics", 11, RED)

    save("gather-scatter.svg", s)


if __name__ == "__main__":
    d1(); d2(); d3(); d4(); d5(); d6()
