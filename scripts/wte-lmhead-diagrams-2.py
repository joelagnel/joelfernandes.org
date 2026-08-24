#!/usr/bin/env python3
"""Diagrams 7-12 for the wte / lm_head weight tying post."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
c = SourceFileLoader("c", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "wte-lmhead-common.py")).load_module()
from c import (head, title, txt, box, line, path, circle, save,
               BLUE, GREEN, ORANGE, RED, PURPLE, GREY, LGREY, DARK)


# ---------------------------------------------------------------------------
# Diagram 7: hidden vector scored against every row
# ---------------------------------------------------------------------------
def d7():
    W, H = 780, 400
    s = head(W, H)
    s += title(W / 2, 26, 'Unembedding: score x against every row, real GPT-2 numbers')

    hx, hy = 40, 170
    cells = 6
    cw, ch = 30, 30
    for i in range(cells):
        s += box(hx + i * cw, hy, cw, ch, "#eaf5ee", GREEN, rx=0, sw=1.2)
    s += txt(hx + cells * cw / 2, hy + ch + 18, "x  (768,)", 10.5, DARK, mono=True)
    s += txt(hx + cells * cw / 2, hy - 26, "final x at the last", 11.5, GREEN, weight="bold")
    s += txt(hx + cells * cw / 2, hy - 12, 'position of "The sky is"', 11.5, GREEN, weight="bold")

    tx, ty = 330, 62
    tw, rh = 200, 38
    rows = [("row 262   the", "#fff4e8", ORANGE, "-93.77"),
            ("row 4171  blue", "#fff4e8", ORANGE, "-94.34"),
            ("row 318   is", "#ffffff", LGREY, "-102.04"),
            ("row 3797  cat", "#ffffff", LGREY, "-103.84"),
            ("... 50253 more rows", "#ffffff", LGREY, "...")]
    for i, (lab, fill, st_, val) in enumerate(rows):
        yy = ty + i * rh
        s += box(tx, yy, tw, rh - 5, fill, st_, rx=2, sw=1.6 if st_ == ORANGE else 0.9)
        s += txt(tx + tw / 2, yy + 21, lab, 10, DARK, mono=True)
        s += path(f"M {hx + cells*cw + 8} {hy + 15} C 230 {hy + 15}, 250 {yy + 16}, {tx - 11} {yy + 16}",
                  GREY if st_ != ORANGE else ORANGE, 1.0 if st_ != ORANGE else 1.7,
                  marker="arGrey" if st_ != ORANGE else "arOrange")
        s += txt(tx + tw + 52, yy + 21, val, 11, DARK if st_ != ORANGE else ORANGE,
                 mono=True, weight="normal" if st_ != ORANGE else "bold")

    s += txt(tx + tw / 2, ty - 14, "every row of wte, 50257 of them", 11.5, DARK)
    s += txt(tx + tw + 52, ty - 14, "logit", 11.5, DARK, weight="bold")

    s += txt(W / 2, 300, "logit for row i  =  dot(x, wte[i])", 13, DARK, mono=True, weight="bold")
    s += txt(W / 2, 326, "Doing all 50257 dot products at once is one matrix multiply:", 11.5, "#555")
    s += txt(W / 2, 346, "logits = x @ wte.T", 12, DARK, mono=True)
    s += txt(W / 2, 372, 'The row for the input token " is" scores low. '
                         'Rows that could come next score high.', 11.5, "#777")

    save("scoring.svg", s)


# ---------------------------------------------------------------------------
# Diagram 8: one buffer, two readers
# ---------------------------------------------------------------------------
def d8():
    W, H = 780, 360
    s = head(W, H)
    s += title(W / 2, 26, "One buffer in memory, read two different ways")

    # buffer in centre
    bx, by, bw, bh = 290, 130, 200, 110
    s += box(bx, by, bw, bh, "#f3eef8", PURPLE, sw=2.0)
    s += txt(bx + bw / 2, by + 34, "one tensor", 13, PURPLE, weight="bold")
    s += txt(bx + bw / 2, by + 56, "(50257, 768)", 12, DARK, mono=True)
    s += txt(bx + bw / 2, by + 78, "38,597,376 floats", 11, "#777")
    s += txt(bx + bw / 2, by + 96, "one data_ptr()", 10.5, "#777", mono=True)

    # left reader
    s += box(40, 100, 200, 80, "#f7fafc", BLUE, sw=1.6)
    s += txt(140, 126, "wte", 14, BLUE, weight="bold", mono=True)
    s += txt(140, 146, "reads row i", 11.5, DARK)
    s += txt(140, 164, "no transpose", 10.5, "#777")
    s += path(f"M 240 140 C 265 140, 265 170, 288 170", BLUE, 1.8, marker="ar")

    # right reader
    s += box(540, 100, 200, 80, "#fffaf5", ORANGE, sw=1.6)
    s += txt(640, 126, "lm_head", 14, ORANGE, weight="bold", mono=True)
    s += txt(640, 146, "computes x @ W.T", 11.5, DARK, mono=True)
    s += txt(640, 164, "transpose at call time", 10.5, "#777")
    s += path(f"M 492 170 C 515 170, 515 140, 538 140", ORANGE, 1.8, marker="arOrange")

    s += txt(W / 2, 286, "self.transformer.wte.weight = self.lm_head.weight", 13, DARK,
             mono=True, weight="bold")
    s += txt(W / 2, 312, "nn.Linear stores .weight as (out_features, in_features) = (50257, 768),",
             11.5, "#666")
    s += txt(W / 2, 330, "which already matches wte, so the assignment needs no transpose.",
             11.5, "#666")

    save("one-buffer.svg", s)


# ---------------------------------------------------------------------------
# Diagram 9: fork in forward, join with + in backward
# ---------------------------------------------------------------------------
def d9():
    W, H = 800, 380
    s = head(W, H)
    s += title(W / 2, 26, "Used twice going forward, so two gradients come back")

    # forward, left half
    s += txt(200, 58, "forward", 13, GREEN, weight="bold")
    s += box(150, 76, 100, 40, "#f3eef8", PURPLE, sw=1.8)
    s += txt(200, 101, "W", 15, PURPLE, weight="bold", mono=True)

    s += path("M 175 116 C 160 150, 130 160, 110 176", GREEN, 1.6, marker="arGreen")
    s += path("M 225 116 C 240 150, 270 160, 290 176", GREEN, 1.6, marker="arGreen")

    s += box(50, 178, 120, 36, "#f7fafc", BLUE, sw=1.4)
    s += txt(110, 202, "wte lookup", 11, DARK, mono=True)
    s += box(230, 178, 120, 36, "#fffaf5", ORANGE, sw=1.4)
    s += txt(290, 202, "lm_head matmul", 10, DARK, mono=True)

    s += path("M 110 214 C 110 250, 180 258, 195 272", GREY, 1.4, marker="arGrey")
    s += path("M 290 214 C 290 250, 220 258, 205 272", GREY, 1.4, marker="arGrey")
    s += box(150, 274, 100, 36, "#ffffff", RED, sw=1.6)
    s += txt(200, 298, "loss", 12, DARK, mono=True)

    # divider
    s += line(400, 60, 400, 350, LGREY, 1.2, dash="5,5")

    # backward, right half
    s += txt(600, 58, "backward", 13, RED, weight="bold")
    s += box(550, 76, 100, 40, "#f3eef8", PURPLE, sw=1.8)
    s += txt(600, 101, "W.grad", 12, PURPLE, weight="bold", mono=True)

    s += path("M 510 176 C 530 160, 560 150, 575 118", RED, 1.6, marker="arRed")
    s += path("M 690 176 C 670 160, 640 150, 625 118", RED, 1.6, marker="arRed")
    s += circle(600, 140, 13, "#ffffff", RED, 1.8)
    s += txt(600, 145, "+", 16, RED, weight="bold")

    s += box(450, 178, 120, 36, "#f7fafc", BLUE, sw=1.4)
    s += txt(510, 196, "from wte", 10.5, DARK, mono=True)
    s += txt(510, 209, "sparse: 84 rows", 8.5, "#777", mono=True)
    s += box(630, 178, 120, 36, "#fffaf5", ORANGE, sw=1.4)
    s += txt(690, 196, "from lm_head", 10.5, DARK, mono=True)
    s += txt(690, 209, "dense: all 50257", 8.5, "#777", mono=True)

    s += path("M 510 272 C 510 250, 510 230, 510 216", GREY, 1.4, marker="arGrey")
    s += path("M 690 272 C 690 250, 690 230, 690 216", GREY, 1.4, marker="arGrey")
    s += box(550, 274, 100, 36, "#ffffff", RED, sw=1.6)
    s += txt(600, 298, "loss", 12, DARK, mono=True)

    s += txt(W / 2, 348, "Measured on one backward pass: the tied gradient equals the two "
                         "separate paths added, to the last bit.", 11.5, "#555")

    save("fork-join.svg", s)


# ---------------------------------------------------------------------------
# Diagram 10: two forces on one row
# ---------------------------------------------------------------------------
def d10():
    W, H = 780, 430
    s = head(W, H)
    s += title(W / 2, 26, "Two updates land on the same row and are added")
    s += txt(W / 2, 46, "Angle is drawn to the measured cosine of -0.007, essentially a right angle.",
             11, "#888", style="italic")
    s += txt(W / 2, 62, "Lengths are not to scale: lm_head is really 80x wte, drawn 3x so both stay visible.",
             11, "#888", style="italic")

    ox, oy = 232, 296          # the row's value before the step

    # angle drawn exactly to the measured cosine (-0.0069, i.e. 90.4 degrees).
    # lengths are compressed: the true per-row norm ratio is ~80x.
    # lm_head points RIGHT-up so nothing runs off the left edge.
    lm = (186, -54)
    wt = (-17, -57)
    sm = (lm[0] + wt[0], lm[1] + wt[1])

    # parallelogram construction
    s += line(ox + lm[0], oy + lm[1], ox + sm[0], oy + sm[1], LGREY, 1.0, dash="4,3")
    s += line(ox + wt[0], oy + wt[1], ox + sm[0], oy + sm[1], LGREY, 1.0, dash="4,3")

    s += line(ox, oy, ox + lm[0], oy + lm[1], ORANGE, 2.4, marker="arOrange")
    s += line(ox, oy, ox + wt[0], oy + wt[1], BLUE, 2.4, marker="ar")
    s += line(ox, oy, ox + sm[0], oy + sm[1], GREEN, 3.0, marker="arGreen")

    # origin, label below and left, clear of every shaft
    s += circle(ox, oy, 5, DARK)
    s += txt(ox - 8, oy + 26, "wte.weight[4171]", 10.5, "#777", anchor="middle")
    s += txt(ox - 8, oy + 39, "before the step", 10.5, "#777", anchor="middle")

    # lm_head label sits BELOW its shaft, which is empty space
    s += txt(ox + lm[0] - 6, oy + lm[1] + 34, "from lm_head", 11.5, ORANGE,
             anchor="end", weight="bold")
    s += txt(ox + lm[0] - 6, oy + lm[1] + 48, "pulls it toward being a detector", 10,
             "#777", anchor="end")

    # wte label, left of its tip
    s += txt(ox + wt[0] - 16, oy + wt[1] + 2, "from wte", 11.5, BLUE,
             anchor="end", weight="bold")
    s += txt(ox + wt[0] - 16, oy + wt[1] + 16, "pulls it toward being", 10, "#777", anchor="end")
    s += txt(ox + wt[0] - 16, oy + wt[1] + 29, "a representation", 10, "#777", anchor="end")

    # sum label, above the arrow and left-anchored into open space
    s += txt(ox + sm[0] - 14, oy + sm[1] - 30, "sum", 12, GREEN, anchor="end", weight="bold")
    s += txt(ox + sm[0] - 14, oy + sm[1] - 16, "the update actually applied", 10, GREEN,
             anchor="end")

    px = 498
    s += box(px, 128, 250, 202, "#fcfcfc", LGREY)
    s += txt(px + 125, 152, "measured, one backward pass", 11, DARK, weight="bold")
    rows = [
        ("rows both paths touch", "84", None),
        ("cosine, mean", "-0.0069", None),
        ("cosine, sd", "0.0463", None),
        ("share with cosine below 0", "54.8%", None),
        ("per-row norm, lm_head", "2.004", ORANGE),
        ("per-row norm, wte", "0.025", BLUE),
    ]
    for i, (k, v, col) in enumerate(rows):
        yy = 182 + i * 23
        if col:
            s += box(px + 12, yy - 8, 9, 9, col, col, rx=2, sw=0.0)
            s += txt(px + 27, yy, k, 9.5, "#666", anchor="start")
        else:
            s += txt(px + 12, yy, k, 9.5, "#666", anchor="start")
        s += txt(px + 238, yy, v, 9.5, DARK, anchor="end", mono=True)

    s += txt(W / 2, 384, "A shuffled control gives the same tilt, so these two are best read as",
             11.5, "#555")
    s += txt(W / 2, 402, "unrelated directions, not opposing forces. The optimizer adds them either way.",
             11.5, "#555")

    save("two-forces.svg", s)


# alias kept so the old call site still works


# ---------------------------------------------------------------------------
# Diagram 11: reach asymmetry
# ---------------------------------------------------------------------------
def d11():
    W, H = 760, 380
    s = head(W, H)
    s += title(W / 2, 26, "The two paths do not reach the same number of rows")

    import random
    random.seed(3)

    def column(x, label, col, dense, note1, note2):
        t = txt(x + 60, 62, label, 12.5, col, weight="bold")
        t += box(x, 76, 120, 220, "#ffffff", DARK, sw=1.5)
        n = 44
        rh = 220 / n
        touched = list(range(n)) if dense else sorted(random.sample(range(n), 4))
        for i in touched:
            yy = 76 + i * rh
            t += box(x, yy, 120, rh, col, col, rx=0, sw=0.0, op=0.30)
            t += line(x - 26, yy + rh / 2, x - 4, yy + rh / 2, col, 1.3, marker=(
                "arOrange" if col == ORANGE else "ar"))
        t += txt(x + 60, 314, note1, 11, DARK, weight="bold")
        t += txt(x + 60, 330, note2, 10, "#777")
        return t

    s += column(220, "from lm_head", ORANGE, True, "all 50257 rows", "every row, every step")
    s += column(500, "from wte", BLUE, False, "84 rows", "only ids present in the batch")

    s += txt(W / 2, 360, "Same matrix, same step. A rare token's row is shaped almost entirely "
                         "by the dense path.", 11.5, "#555")

    save("reach.svg", s)


# ---------------------------------------------------------------------------
# Diagram 12: the pull. hidden state and target row converge
# ---------------------------------------------------------------------------
def d12():
    W, H = 800, 430
    s = head(W, H)
    s += title(W / 2, 26, 'Cross entropy pulls the hidden state and the target row together')

    s += box(50, 50, 700, 300, "#fcfcfc", LGREY)

    bx, by = 470, 150   # wte[" blue"]
    hx, hy = 220, 260   # hidden state

    # target row
    s += circle(bx, by, 7, ORANGE)
    s += txt(bx + 14, by - 6, 'wte[4171]', 12, ORANGE, anchor="start", mono=True, weight="bold")
    s += txt(bx + 14, by + 10, 'the row for " blue"', 10.5, "#777", anchor="start")

    # hidden state
    s += circle(hx, hy, 7, GREEN)
    s += txt(hx - 14, hy + 6, "h", 13, GREEN, anchor="end", mono=True, weight="bold")
    s += txt(hx - 14, hy + 22, 'final state after "The sky is"', 10.5, "#777", anchor="end")

    # pull arrows toward each other
    s += path(f"M {hx + 12} {hy - 6} C 300 220, 360 190, {bx - 60} {by + 34}",
              GREEN, 2.2, marker="arGreen")
    s += txt(320, 200, "-dL/dh points at the target row", 10.5, GREEN)

    s += path(f"M {bx - 12} {by + 10} C 420 200, 360 230, {hx + 60} {hy - 22}",
              ORANGE, 2.2, marker="arOrange")
    s += txt(400, 262, "-dL/dW[4171] points at h", 10.5, ORANGE)

    # a pushed-away row
    s += circle(620, 290, 5.5, GREY)
    s += txt(632, 294, 'wte[3797]  " cat"', 10.5, "#888", anchor="start", mono=True)
    s += path("M 612 296 C 660 320, 690 330, 700 336", GREY, 1.4, dash="4,3", marker="arGrey")
    s += txt(660, 356, "every other row is pushed the other way", 10, "#888")

    s += txt(W / 2, 384, "Verified numerically: cos(-dL/dW[4171], h) = 1.000000  and  "
                         "cos(-dL/dh, wte[4171]) = 0.844641", 11.5, DARK, mono=True)
    s += txt(W / 2, 408, "For a non-target row, cos(-dL/dW[3797], h) = -1.000000",
             11.5, "#777", mono=True)

    save("the-pull.svg", s)


if __name__ == "__main__":
    d7(); d8(); d9(); d10(); d11(); d12()
