#!/usr/bin/env python3
"""Replace combining-circumflex labels in graphviz SVG output with an explicit
caret path, so the hat sits squarely over the x regardless of font support.

Graphviz emits the label "x̂₁" as a single <text> run. Fonts vary wildly in how
they place U+0302, so instead we drop the combining mark from the text and draw
the caret ourselves, centred on the x glyph.
"""
import glob
import os
import re
import sys

SUBS = "\u2081\u2082\u2083\u2084"          # subscript 1..4
COMB = "\u0302"                              # combining circumflex


def fix(path):
    src = open(path, encoding="utf-8").read()
    if COMB not in src:
        return 0

    out = []
    pos = 0
    n = 0
    # <text ... x="X" y="Y" font-size="F" ...>x̂₁</text>
    pat = re.compile(
        r'<text([^>]*?)x="([-\d.]+)"([^>]*?)y="([-\d.]+)"([^>]*?)>'
        r'x' + COMB + r'([' + SUBS + r'])</text>'
    )
    for m in pat.finditer(src):
        pre, x, mid, y, post, sub = m.groups()
        x, y = float(x), float(y)
        fs = 15.0
        fm = re.search(r'font-size="([\d.]+)"', m.group(0))
        if fm:
            fs = float(fm.group(1))

        # honour the run's text-anchor: graphviz uses start for these labels,
        # so x is the LEFT edge, and the x glyph centre is half a glyph in.
        anchor = "middle"
        am = re.search(r'text-anchor="(\w+)"', m.group(0))
        if am:
            anchor = am.group(1)
        x_glyph = fs * 0.56          # advance width of "x" at this size
        sub_w = fs * 0.30
        if anchor == "start":
            cx = x + x_glyph / 2.0
        elif anchor == "end":
            cx = x - sub_w - x_glyph / 2.0
        else:
            cx = x - sub_w / 2.0
        half = fs * 0.20
        top = y - fs * 0.78
        bot = y - fs * 0.56

        caret = (
            f'<path d="M {cx-half:.2f} {bot:.2f} L {cx:.2f} {top:.2f} '
            f'L {cx+half:.2f} {bot:.2f}" fill="none" stroke="#1a1a1a" '
            f'stroke-width="{fs*0.085:.2f}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
        text = (f'<text{pre}x="{x}"{mid}y="{y}"{post}>x{sub}</text>')

        out.append(src[pos:m.start()])
        out.append(text + caret)
        pos = m.end()
        n += 1

    out.append(src[pos:])
    if n:
        open(path, "w", encoding="utf-8").write("".join(out))
    return n


if __name__ == "__main__":
    total = 0
    for f in sys.argv[1:] or glob.glob("*.svg"):
        c = fix(f)
        if c:
            print(f"  {os.path.basename(f)}: {c} hats redrawn")
        total += c
    if total == 0:
        print("  no combining circumflex found", file=sys.stderr)
