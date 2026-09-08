#!/usr/bin/env python3
"""Generate original SVG teaching diagrams for the Flash Attention post."""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "images" / "flash-attention"
C = {
    "ink": "#0f172a", "muted": "#475569", "q": "#dbeafe", "qs": "#2563eb",
    "k": "#ede9fe", "ks": "#7c3aed", "v": "#ffedd5", "vs": "#ea580c",
    "state": "#ccfbf1", "ss": "#0f766e", "hbm": "#fce7f3", "hs": "#be185d",
    "sram": "#ecfccb", "srs": "#4d7c0f", "line": "#475569",
}


def open_svg(width, height, title, desc):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(desc)}</desc>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#475569"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def text(parts, x, y, value, size=28, weight=400, anchor="start", color=None):
    color = color or C["ink"]
    for n, line in enumerate(value.split("\n")):
        parts.append(
            f'<text x="{x}" y="{y + n * (size + 8)}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(line)}</text>'
        )


def box(parts, x, y, width, height, value, fill, stroke, size=25, weight=600, radius=16):
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    lines = value.split("\n")
    top = y + height / 2 - ((len(lines) - 1) * (size + 6)) / 2 + size * 0.35
    text(parts, x + width / 2, top, value, size, weight, "middle")


def arrow(parts, x1, y1, x2, y2, label=None, label_y=None):
    parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{C["line"]}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>')
    if label:
        text(parts, (x1 + x2) / 2, label_y if label_y is not None else min(y1, y2) - 13,
             label, 20, 600, "middle", C["muted"])


def polyarrow(parts, points):
    path = " L".join(f"{x},{y}" for x, y in points)
    parts.append(f'<path d="M{path}" stroke="{C["line"]}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>')


def tag(parts, x, y, value, fill, stroke):
    width = max(110, len(value) * 15 + 34)
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="38" rx="19" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    text(parts, x + width / 2, y + 26, value, 18, 700, "middle")


def save(name, parts):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def tile_redraw():
    parts = open_svg(
        820, 1150, "A whiteboard-derived redraw of row two processed in two K and V tiles",
        "HBM holds K and V in a two-row tile followed by a one-row tile. For query row two, temporary score weights are immediately multiplied by full V vectors and combined into m, l, and o on chip.",
    )
    text(parts, 48, 58, "Tile-local scores become", 34, 700)
    text(parts, 48, 100, "value-vector contributions immediately", 34, 700)
    text(parts, 48, 145, "Whiteboard-informed redraw: no full T x T score matrix reaches HBM.", 18, color=C["muted"])

    parts.append(f'<rect x="55" y="195" width="300" height="405" rx="16" fill="{C["hbm"]}" stroke="{C["hs"]}" stroke-width="3"/>')
    text(parts, 205, 245, "HBM / VRAM", 27, 700, "middle")
    box(parts, 90, 295, 230, 120, "K / V tile 1\nK0,V0\nK1,V1", C["hbm"], C["hs"], 24)
    box(parts, 90, 465, 230, 90, "K / V tile 2\nK2,V2", C["hbm"], C["hs"], 24)

    parts.append(f'<rect x="440" y="195" width="325" height="405" rx="16" fill="{C["sram"]}" stroke="{C["srs"]}" stroke-width="3"/>')
    text(parts, 602, 245, "on-chip SRAM", 27, 700, "middle")
    box(parts, 482, 295, 240, 210, "Q2\n\nfresh state\nm=-inf, l=0\no=[0,0]", C["sram"], C["srs"], 25)
    arrow(parts, 320, 355, 482, 355)
    arrow(parts, 320, 510, 482, 510)

    box(parts, 70, 670, 680, 135, "tile 1: S=[0,0], weights=[1,1]\n1*V0 + 1*V1  =>  o=[10,20], l=2", C["v"], C["vs"], 25)
    box(parts, 70, 845, 680, 135, "tile 2: S=[0], weight=1\n1*V2  =>  o=[40,50], l=3", C["v"], C["vs"], 25)
    box(parts, 165, 1020, 490, 88, "final: Y2 = o / l = [13.333, 16.667]", "#dcfce7", "#15803d", 24)
    arrow(parts, 602, 505, 410, 670)
    arrow(parts, 410, 805, 410, 845)
    arrow(parts, 410, 980, 410, 1020)
    save("whiteboard-tile-redraw.svg", parts)


def tiled_flow():
    parts = open_svg(
        820, 1150, "Flash Attention keeps tile-local work on chip",
        "HBM provides a Q tile and a stream of K and V tiles. On-chip SRAM retains the Q tile, forms temporary score tiles, and separately updates m, l, and o for every query row. Only a completed output tile returns to HBM.",
    )
    text(parts, 48, 58, "Flash Attention: score tiles", 34, 700)
    text(parts, 48, 100, "stay on chip", 34, 700)
    text(parts, 48, 145, "HBM is large storage. SRAM and registers are the small, fast workspace.", 18, color=C["muted"])

    parts.append(f'<rect x="60" y="195" width="700" height="185" rx="16" fill="{C["hbm"]}" stroke="{C["hs"]}" stroke-width="3"/>')
    text(parts, 410, 245, "HBM / VRAM", 27, 700, "middle")
    box(parts, 110, 280, 220, 72, "Q tile", C["q"], C["qs"], 25)
    box(parts, 485, 268, 225, 96, "K / V tile stream\nnext tile arrives", C["hbm"], C["hs"], 22)

    parts.append(f'<rect x="60" y="455" width="700" height="545" rx="16" fill="{C["sram"]}" stroke="{C["srs"]}" stroke-width="3"/>')
    text(parts, 410, 505, "on-chip SRAM / registers", 27, 700, "middle")
    box(parts, 112, 545, 245, 92, "Q_i stays resident", C["q"], C["qs"], 23)
    box(parts, 463, 545, 245, 92, "K_j + V_j\ncurrent tile", C["k"], C["ks"], 23)
    box(parts, 205, 690, 410, 125, "temporary score tile\nS_ij = Q_i K_j^T\nmask, exponentials", C["sram"], C["srs"], 23)
    box(parts, 205, 855, 410, 105, "separate m, l, o for each query row\nupdated for every tile", C["state"], C["ss"], 22)

    box(parts, 250, 1050, 320, 74, "completed O tile in HBM", "#dcfce7", "#15803d", 22)
    arrow(parts, 220, 352, 235, 545)
    arrow(parts, 597, 364, 585, 545)
    arrow(parts, 235, 637, 340, 690)
    arrow(parts, 585, 637, 480, 690)
    arrow(parts, 410, 815, 410, 855)
    arrow(parts, 410, 960, 410, 1050)
    save("flash-tiled-flow.svg", parts)


def correction():
    parts = open_svg(
        820, 1040, "Online softmax corrects earlier state when a new maximum arrives",
        "A first score tile contains scores one and two and establishes maximum two. A later score of four changes the maximum, so the current denominator and value numerator state are rescaled by exp of minus two before the new contribution is added.",
    )
    text(parts, 48, 58, "When a later tile finds a larger", 32, 700)
    text(parts, 48, 98, "score, rescale the current state", 32, 700)
    text(parts, 48, 143, "Online normalizer recurrence, applied to the value numerator too.", 18, color=C["muted"])
    box(parts, 60, 210, 230, 130, "tile 1 scores\n[1, 2]", C["k"], C["ks"], 26)
    box(parts, 400, 190, 350, 170, "after tile 1\nm = 2\nl = exp(-1)+1\no = exp(-1)*V0 + V1", C["state"], C["ss"], 22)
    arrow(parts, 290, 275, 400, 275)
    box(parts, 100, 455, 620, 155, "r = exp(m - m') = exp(-2)\n\nm was 2.  r puts l and o on m'=4's scale.", "#fff7ed", "#c2410c", 22)
    arrow(parts, 575, 360, 410, 455)
    box(parts, 60, 720, 230, 125, "tile 2 score\n[4]", C["k"], C["ks"], 26)
    box(parts, 400, 695, 350, 170, "after tile 2\nm' = 4\nl = r*l + 1\no = r*o + V2", C["state"], C["ss"], 22)
    arrow(parts, 290, 782, 400, 782)
    polyarrow(parts, [(720, 532), (775, 532), (775, 780), (750, 780)])
    save("online-maximum-correction.svg", parts)


if __name__ == "__main__":
    tile_redraw()
    tiled_flow()
    correction()
