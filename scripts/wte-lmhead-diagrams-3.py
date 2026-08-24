#!/usr/bin/env python3
"""Diagrams for the two sections added after review:
   - two-contractions.svg : same set of 50257 vectors, two axes to travel
   - dog-slot.svg         : ' cat' is a near neighbour of ' dog' in the table
                            and a terrible prediction after 'The dog'
All numbers printed from real GPT-2 124M weights.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
c = import_module("wte-lmhead-common")

head, save, box, txt, title, line, path, circle = (
    c.head, c.save, c.box, c.txt, c.title, c.line, c.path, c.circle)
BLUE, GREEN, ORANGE, RED, PURPLE, GREY, LGREY, DARK = (
    c.BLUE, c.GREEN, c.ORANGE, c.RED, c.PURPLE, c.GREY, c.LGREY, c.DARK)


def shape(x, y, parts, size=11):
    """Shape label like (50257, 768) with one axis coloured red."""
    # SVG collapses a trailing space inside a tspan, so use a non-breaking one
    inner = "".join(
        f'<tspan fill="{col}" font-weight="{"bold" if col == RED else "normal"}">'
        f'{c.esc(t).replace(" ", chr(160))}</tspan>'
        for t, col in parts)
    return (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
            f'{c.MONO}>{inner}</text>\n')


def two_contractions():
    W, H = 760, 366
    s = head(W, H)
    s += title(W / 2, 26, "The same 50257 vectors, travelled along two different axes")

    NROW, RH = 11, 13      # stylised stack standing in for 50257 rows
    TOP = 108
    PICK = 4               # which stylised row is the selected one
    NOTE, SUB = 300, 320

    # ================= left panel: index the 50257 axis =================
    lx, lw = 96, 120
    lcx = 196
    s += txt(lcx, 62, "lookup", size=14, fill=BLUE, weight="bold")
    s += txt(lcx, 78, "you have an index, you want a vector", size=10,
             fill=GREY, style="italic")

    for r in range(NROW):
        y = TOP + r * RH
        hit = r == PICK
        s += box(lx, y, lw, RH - 2, fill=(GREEN if hit else PURPLE),
                 op=(0.34 if hit else 0.10), stroke=(GREEN if hit else LGREY),
                 sw=(1.5 if hit else 0.8), rx=2)
    ly1 = TOP + NROW * RH

    s += txt(lx + lw / 2, ly1 + 18, "W", size=13, fill=PURPLE, weight="bold", mono=True)
    s += shape(lx + lw / 2, ly1 + 34, [("(", GREY), ("50257", RED), (", 768)", GREY)])

    # the indexed axis runs down the stack
    s += line(lx - 14, TOP, lx - 14, ly1 - 2, stroke=RED, sw=1.6, dash="4,3")
    s += txt(lx - 22, (TOP + ly1) / 2 - 3, "50257", size=10, fill=RED,
             weight="bold", anchor="end", mono=True)
    s += txt(lx - 22, (TOP + ly1) / 2 + 11, "indexed", size=9.5, fill=RED, anchor="end")

    # pull the selected row out
    py = TOP + PICK * RH
    s += txt(lx + lw + 62, py - 16, "pulled out whole", size=9.5, fill=GREY, style="italic")
    s += line(lx + lw + 4, py + RH / 2 - 1, lx + lw + 22, py + RH / 2 - 1,
              stroke=GREEN, sw=1.6, marker="arGreen")
    s += box(lx + lw + 26, py - 4, 72, RH + 6, fill=GREEN, op=0.28, stroke=GREEN, rx=2)
    s += txt(lx + lw + 62, py + 9, "W[i]", size=11, fill=GREEN, weight="bold", mono=True)
    s += txt(lx + lw + 62, py + 27, "(768,)", size=10, fill=GREY, mono=True)

    s += txt(lcx, NOTE, "one row read", size=11, fill=BLUE, weight="bold")
    s += txt(lcx, SUB, "the other 50,256 untouched", size=10, fill=GREY)

    # ================= divider =================
    s += line(380, 76, 380, 340, stroke=LGREY, sw=1.2, dash="5,4")

    # ================= right panel: sum over the 768 axis =================
    rx0, rw_ = 512, 120
    rcx = 578
    s += txt(rcx - 34, 62, "unembedding", size=14, fill=ORANGE, weight="bold")
    s += txt(rcx - 34, 78, "you have a vector, you want scores", size=10,
             fill=GREY, style="italic")

    for r in range(NROW):
        y = TOP + r * RH
        s += box(rx0, y, rw_, RH - 2, fill=PURPLE, op=0.10, stroke=LGREY, sw=0.8, rx=2)
        # every row is read, and every row emits exactly one number
        s += line(rx0 + rw_ + 3, y + RH / 2 - 1, rx0 + rw_ + 20, y + RH / 2 - 1,
                  stroke=ORANGE, sw=0.9)
        s += circle(rx0 + rw_ + 27, y + RH / 2 - 1, 3.0, fill=ORANGE, op=0.75)
    ry1 = TOP + NROW * RH

    s += txt(rx0 + rw_ / 2, ry1 + 18, "W", size=13, fill=PURPLE, weight="bold", mono=True)
    s += shape(rx0 + rw_ / 2, ry1 + 34, [("(50257, ", GREY), ("768", RED), (")", GREY)])

    # the summed axis runs across a row
    s += line(rx0, TOP - 12, rx0 + rw_, TOP - 12, stroke=RED, sw=1.6, dash="4,3")
    s += txt(rx0 + rw_ / 2, TOP - 18, "768 summed", size=10, fill=RED, weight="bold")

    # x, the query vector
    qy = TOP + 4 * RH
    s += txt(rx0 - 46, qy - 14, "the query", size=9.5, fill=GREY, style="italic")
    s += box(rx0 - 76, qy - 4, 60, RH + 6, fill=ORANGE, op=0.18, stroke=ORANGE, rx=2)
    s += txt(rx0 - 46, qy + 9, "x", size=12, fill=ORANGE, weight="bold", mono=True)
    s += txt(rx0 - 46, qy + 27, "(768,)", size=10, fill=GREY, mono=True)
    s += line(rx0 - 14, qy + RH / 2 - 1, rx0 - 3, qy + RH / 2 - 1,
              stroke=ORANGE, sw=1.4, marker="arOrange")

    s += txt(rx0 + rw_ + 30, TOP + 34, "one number", size=9.5, fill=GREY,
             style="italic", anchor="start")
    s += txt(rx0 + rw_ + 30, TOP + 46, "per row", size=9.5, fill=GREY,
             style="italic", anchor="start")
    s += txt(rx0 + rw_ + 27, ry1 + 34, "(50257,)", size=10, fill=GREY, mono=True)

    s += txt(rcx, NOTE, "every row read", size=11, fill=ORANGE, weight="bold")
    s += txt(rcx, SUB, "each crushed to a single score", size=10, fill=GREY)

    save("two-contractions.svg", s)


def dog_slot():
    W, H = 760, 400
    s = head(W, H)
    s += title(W / 2, 26, 'The same matrix, asked two questions about " dog"')

    s += line(W / 2, 52, W / 2, 372, stroke=LGREY, sw=1.2, dash="5,4")

    # ---- left: table neighbourhood ----
    s += txt(180, 62, "in the table: nearest rows to wte[ dog]", size=12,
             fill=BLUE, weight="bold")
    s += txt(180, 79, "cosine between rows, meaning", size=10, fill=GREY, style="italic")
    nb = [(" dogs", "0.796"), (" Dog", "0.734"), (" canine", "0.652"),
          (" puppy", "0.621"), (" pet", "0.551"), (" cat", "0.550")]
    y = 100
    for tok, v in nb:
        hi = tok == " cat"
        s += box(52, y, 256, 26, fill=(GREEN if hi else BLUE), op=(0.20 if hi else 0.08),
                 stroke=(GREEN if hi else LGREY), sw=(1.6 if hi else 1.0))
        s += txt(70, y + 18, f'"{tok}"', size=11, fill=DARK, anchor="start", mono=True,
                 weight=("bold" if hi else "normal"))
        s += txt(290, y + 18, v, size=11, fill=(GREEN if hi else DARK), anchor="end",
                 mono=True, weight=("bold" if hi else "normal"))
        y += 31
    s += txt(180, y + 18, '" cat" is a close neighbour', size=11, fill=GREEN, weight="bold")
    s += txt(180, y + 35, "rank 9 of 50257", size=10, fill=GREY)

    # ---- right: predictions ----
    s += txt(580, 62, 'at the output: what follows "The dog"', size=12,
             fill=ORANGE, weight="bold")
    s += txt(580, 79, "dot of final x with every row, slot", size=10, fill=GREY, style="italic")
    pr = [(" was", "0.143", 42), ("'s", "0.104", 31), (",", "0.085", 25),
          (" is", "0.073", 22), (" had", "0.048", 14), (" cat", "0.00015", 1)]
    y = 100
    for tok, v, bw in pr:
        hi = tok == " cat"
        s += box(452, y, 256, 26, fill="#ffffff", op=1.0, stroke=LGREY, sw=1.0)
        # bar track stops short of the value column so a full bar never
        # runs into the number
        s += box(452, y, max(bw * 3.6, 3), 26, fill=(RED if hi else ORANGE),
                 op=(0.30 if hi else 0.22), stroke="none", sw=0)
        s += txt(468, y + 18, f'"{tok}"', size=11, fill=DARK, anchor="start", mono=True,
                 weight=("bold" if hi else "normal"))
        s += txt(692, y + 18, v, size=11, fill=(RED if hi else DARK), anchor="end",
                 mono=True, weight=("bold" if hi else "normal"))
        y += 31
    s += txt(580, y + 18, '" cat" is a bad answer', size=11, fill=RED, weight="bold")
    s += txt(580, y + 35, "rank 423 of 50257", size=10, fill=GREY)

    s += txt(W / 2, 388, "right meaning, wrong slot",
             size=12, fill=DARK, weight="bold")
    save("dog-slot.svg", s)


def query_shadow():
    """The output projection as a similarity test: x is a query, and each
    vocabulary row scores by how much of it lies along x."""
    import math
    W_, H_ = 760, 442
    s = head(W_, H_)
    s += title(W_ / 2, 26, "The unembedding asks: which vocabulary vector points this way?")

    OX, OY = 120, 318          # origin
    AXL = 392                  # length of the x axis
    VL = 232                   # drawn length of every vocabulary vector

    # the query axis
    s += line(OX, OY, OX + AXL, OY, stroke=ORANGE, sw=2.2, marker="arOrange")
    s += txt(OX + AXL / 2, OY + 62, "x, the description of what fits next", size=12,
             fill=ORANGE, weight="bold")
    s += txt(OX + AXL / 2, OY + 78, "after 12 blocks, at the last position of \"The sky is\"",
             size=10, fill=GREY, style="italic")
    s += circle(OX, OY, 4.0, fill=DARK)

    # vocabulary rows, angle stands in for agreement with x
    rows = [
        (" the",     14, "0.152", GREEN),
        (" blue",    26, "0.086", GREEN),
        (" falling", 38, "0.073", GREEN),
        (" is",      58, "0.000", RED),
        (" cat",     78, "0.000", RED),
    ]
    for tok, deg, p, col in rows:
        r = math.radians(deg)
        tx, ty = OX + VL * math.cos(r), OY - VL * math.sin(r)
        sx = OX + VL * math.cos(r)          # foot of the perpendicular
        # the vector itself
        s += line(OX, OY, tx, ty, stroke=col, sw=1.6, marker=("arGreen" if col == GREEN else "arRed"))
        # drop a perpendicular onto the x axis
        s += line(tx, ty, sx, OY, stroke=col, sw=1.0, dash="3,3")
        # the shadow it casts, drawn just above the axis
        s += (f'  <line x1="{OX}" y1="{OY - 5}" x2="{sx}" y2="{OY - 5}" '
              f'stroke="{col}" stroke-width="3.2" stroke-opacity="0.5"/>\n')
        s += circle(sx, OY, 3.0, fill=col)
        # label past the tip, clear of the arrowhead
        s += txt(tx + 14, ty - 3, f'"{tok}"', size=11, fill=col, weight="bold",
                 anchor="start", mono=True)
        s += txt(tx + 14, ty + 11, f"p {p}", size=10, fill=GREY, anchor="start", mono=True)

    # what the shadow means, parked below the axis, right-aligned to the arrow tip
    s += txt(OX + AXL, OY + 22, "the shadow on x is the score,", size=10,
             fill=DARK, weight="bold", anchor="end")
    s += txt(OX + AXL, OY + 35, "dot(x, W[i])", size=10, fill=GREY,
             anchor="end", mono=True)

    # the reading
    s += box(544, 236, 196, 92, fill=BLUE, op=0.06, stroke=LGREY, rx=5)
    s += txt(642, 258, "same direction", size=11, fill=GREEN, weight="bold")
    s += txt(642, 274, "long shadow, high score", size=10, fill=GREY)
    s += txt(642, 298, "wide angle", size=11, fill=RED, weight="bold")
    s += txt(642, 314, "short shadow, low score", size=10, fill=GREY)

    s += txt(W_ / 2, 428,
             "2D sketch of a 768D geometry, angles exaggerated; in 768D every pair is near right angles",
             size=9.5, fill=GREY, style="italic")
    save("query-shadow.svg", s)


def regularizer_signature():
    """Tying makes the model fit the training set worse and held-out text
    better. That divergence is the definition of a regularizer.
    Numbers from Press and Wolf 2017, table 5, PTB, large model + Bayesian
    dropout."""
    W_, H_ = 700, 372
    s = head(W_, H_)
    s += title(W_ / 2, 26, "What tying does to a large model: train worse, test better")

    LX, RX = 196, 470          # untied column, tied column
    TOP, BOT = 100, 288        # y for perplexity 32 and 86
    PMIN, PMAX = 32.0, 86.0

    def py(p):
        return TOP + (p - PMIN) / (PMAX - PMIN) * (BOT - TOP)

    # frame
    s += line(LX, TOP - 16, LX, BOT + 16, stroke=LGREY, sw=1.0)
    s += line(RX, TOP - 16, RX, BOT + 16, stroke=LGREY, sw=1.0)
    s += txt(LX, TOP - 26, "untied", size=13, fill=DARK, weight="bold")
    s += txt(LX, TOP - 12, "66M params", size=9.5, fill=GREY)
    s += txt(RX, TOP - 26, "tied", size=13, fill=PURPLE, weight="bold")
    s += txt(RX, TOP - 12, "51M params", size=9.5, fill=PURPLE)

    series = [
        ("test perplexity", 78.4, 74.3, GREEN, "generalizes better"),
        ("train perplexity", 37.8, 48.5, RED, "fits the training set worse"),
    ]
    for label, a, b, col, note in series:
        ya, yb = py(a), py(b)
        s += line(LX, ya, RX, yb, stroke=col, sw=2.4)
        s += circle(LX, ya, 5.0, fill=col)
        s += circle(RX, yb, 5.0, fill=col)
        s += txt(LX - 14, ya + 4, f"{a}", size=12, fill=col, weight="bold",
                 anchor="end", mono=True)
        s += txt(RX + 14, yb + 4, f"{b}", size=12, fill=col, weight="bold",
                 anchor="start", mono=True)
        # series label sits left of the untied column
        s += txt(LX - 14, ya - 14, label, size=10, fill=col, anchor="end")
        s += txt(RX + 14, yb - 15, note, size=10, fill=GREY,
                 anchor="start", style="italic")

    # the gap between the two lines is what closed
    s += txt(W_ / 2, 336,
             "the two moved in opposite directions, which is what a regularizer does",
             size=11, fill=DARK, weight="bold")
    s += txt(W_ / 2, 354,
             "Press and Wolf 2017, table 5: PTB, large LSTM with Bayesian dropout, perplexity lower is better",
             size=9, fill=GREY, style="italic")
    save("regularizer-signature.svg", s)


def two_routes():
    """The same tensor is reached by two backward routes of very different
    length: straight off the loss, and all the way down through 12 blocks."""
    W_, H_ = 760, 486
    s = head(W_, H_)
    s += title(W_ / 2, 26, "One tensor, two routes back from the loss")

    CX = 300                    # the spine of the forward pass
    TOPY, BOTY = 44, 358

    # ---- the forward stack ----
    stack = [
        (BOTY,      "wte lookup",     BLUE,   "reads row i"),
        (BOTY - 62, "12 blocks",      GREY,   "attention + MLP"),
        (BOTY - 124, "ln_f",          GREY,   ""),
        (BOTY - 186, "lm_head",       ORANGE, "x @ W.T"),
        (TOPY + 62,  "cross entropy", RED,    "loss"),
    ]
    for y, label, col, sub in stack:
        h = 34
        s += box(CX - 78, y - h / 2, 156, h, fill=col, op=0.12, stroke=col, rx=4)
        s += txt(CX, y + (1 if not sub else -2), label, size=12, fill=col, weight="bold")
        if sub:
            s += txt(CX, y + 11, sub, size=9, fill=GREY, mono=True)
    # forward arrows
    for a, b in [(BOTY, BOTY - 62), (BOTY - 62, BOTY - 124),
                 (BOTY - 124, BOTY - 186), (BOTY - 186, TOPY + 62)]:
        s += line(CX, a - 17, CX, b + 17, stroke=LGREY, sw=1.2, marker="arGrey")

    # ---- the shared tensor ----
    WY = 430
    s += box(CX - 78, WY - 17, 156, 34, fill=PURPLE, op=0.18, stroke=PURPLE, rx=4)
    s += txt(CX, WY - 1, "W", size=13, fill=PURPLE, weight="bold", mono=True)
    s += txt(CX, WY + 12, "(50257, 768)", size=9, fill=PURPLE, mono=True)

    # both routes land directly on W, separated horizontally
    JY = 394
    BLZ, ORZ = CX - 46, CX + 46      # where each arrowhead meets W
    s += txt(CX, JY - 10, "accumulated into the same .grad", size=9.5,
             fill=PURPLE, style="italic")

    LOSSY = TOPY + 62

    # ---- short route: loss -> lm_head -> W, down the right ----
    RX = CX + 132
    s += path(f"M {CX + 80} {LOSSY} L {RX} {LOSSY} L {RX} {JY} L {ORZ} {JY} "
              f"L {ORZ} {WY - 17}", stroke=ORANGE, sw=2.0, marker="arOrange")
    s += txt(RX + 10, 152, "short route", size=12, fill=ORANGE,
             weight="bold", anchor="start")
    s += txt(RX + 10, 168, "one matmul from", size=9.5, fill=GREY,
             anchor="start", style="italic")
    s += txt(RX + 10, 180, "the loss", size=9.5, fill=GREY,
             anchor="start", style="italic")
    s += txt(RX + 10, 200, "all 50257 rows", size=10, fill=ORANGE, anchor="start")
    s += txt(RX + 10, 214, "norm 52.94", size=10, fill=ORANGE, anchor="start", mono=True)

    # ---- long route: loss -> every block -> wte -> W, down the left ----
    LX = CX - 132
    s += path(f"M {CX - 80} {LOSSY} L {LX} {LOSSY} L {LX} {JY} L {BLZ} {JY} "
              f"L {BLZ} {WY - 17}", stroke=BLUE, sw=2.0, marker="ar")
    # stubs that actually touch each stage, showing the route passes through all of them
    for y, _l, _c, _s in stack[:4]:
        s += line(LX, y, CX - 78, y, stroke=BLUE, sw=1.0, dash="3,3")
        s += circle(CX - 78, y, 2.6, fill=BLUE)
    s += txt(LX - 10, 152, "long route", size=12, fill=BLUE,
             weight="bold", anchor="end")
    s += txt(LX - 10, 168, "back through", size=9.5, fill=GREY,
             anchor="end", style="italic")
    s += txt(LX - 10, 180, "every block", size=9.5, fill=GREY,
             anchor="end", style="italic")
    s += txt(LX - 14, 200, "only 7 rows", size=10, fill=BLUE, anchor="end")
    s += txt(LX - 14, 214, "norm 0.83", size=10, fill=BLUE, anchor="end", mono=True)

    s += txt(W_ / 2, 474,
             "both land in the same .grad buffer; the short route carries 64x the norm",
             size=10, fill=GREY, style="italic")
    save("two-routes.svg", s)


if __name__ == "__main__":
    two_contractions()
    query_shadow()
    regularizer_signature()
    two_routes()
    dog_slot()
