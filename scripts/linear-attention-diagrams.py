#!/usr/bin/env python3
"""Render compact, original mathematical diagrams for the linear-attention post."""
from html import escape
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1] / "images" / "linear-attention"
INK = "#17333b"
TEAL = "#236e65"
BLUE = "#315fa3"
GOLD = "#80500b"
RED = "#9b3939"


def text(x, y, value, size=24, color=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')


def matrix(xs, ys, values, color):
    parts = []
    left, right = min(xs) - 35, max(xs) + 35
    top, bottom = min(ys) - 28, max(ys) + 12
    for x, dx in [(left, 9), (right, -9)]:
        parts.append(f'<path d="M{x+dx} {top} H{x} V{bottom} H{x+dx}" '
                     f'fill="none" stroke="{color}" stroke-width="2.5"/>')
    for y, row in zip(ys, values):
        for x, value in zip(xs, row):
            parts.append(text(x, y, str(value), 29, color))
    return "".join(parts)


def save(name, title, desc, height, parts):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="{height}" '
           f'viewBox="0 0 600 {height}" role="img" aria-labelledby="title desc">'
           f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>'
           '<rect width="100%" height="100%" rx="8" fill="#ffffff"/>'
           '<g font-family="DejaVu Sans, sans-serif">' + "".join(parts) + '</g></svg>')
    path = ROOT / f"{name}.svg"
    path.write_text(svg)
    with path.with_suffix('.png').open('wb') as output:
        subprocess.run(['rsvg-convert', '-w', '600', str(path)], stdout=output,
                       check=True, timeout=15)


save('readout', 'Read: a memory matrix times a query',
     'A 3 by 2 state multiplies a 2 by 1 query to produce a 3 by 1 output. '
     'Each output component is the dot product of one state row and the query.', 272, [
    text(300, 31, 'Read the memory: y = S q', 26, weight='bold'),
    text(125, 72, 'S', 25, TEAL), text(316, 72, 'q', 25, BLUE), text(500, 72, 'y', 25, GOLD),
    matrix([94, 155], [112, 148, 184], [[3, 0], [1, 2], [2, 1]], TEAL),
    text(238, 148, '×', 31),
    matrix([316], [130, 166], [[0.6], [0.8]], BLUE),
    text(410, 148, '=', 31),
    matrix([500], [112, 148, 184], [[1.8], [2.2], ['2.0']], GOLD),
    text(125, 221, '3 × 2', 23, TEAL), text(316, 221, '2 × 1', 23, BLUE),
    text(500, 221, '3 × 1', 23, GOLD),
    text(300, 256, 'S rows: value features. S columns: key features.', 19),
])

save('write', 'Write: a value-key outer product',
     'Value column [3,1,2] transpose times key row [1,0] gives the 3 by 2 outer '
     'product with rows [3,0], [1,0], [2,0]. This matrix can be added to the state.', 272, [
    text(300, 31, 'Write an association: v kᵀ', 26, weight='bold'),
    text(85, 72, 'v', 25, GOLD), text(275, 72, 'kᵀ', 25, BLUE),
    text(485, 72, 'v kᵀ', 25, TEAL),
    matrix([85], [112, 148, 184], [[3], [1], [2]], GOLD),
    text(170, 148, '×', 31),
    matrix([245, 305], [148], [[1, 0]], BLUE),
    text(385, 148, '=', 31),
    matrix([455, 515], [112, 148, 184], [[3, 0], [1, 0], [2, 0]], TEAL),
    text(85, 221, '3 × 1', 23, GOLD), text(275, 221, '1 × 2', 23, BLUE),
    text(485, 221, '3 × 2', 23, TEAL),
    text(300, 256, 'Every value component × every key component.', 19),
])

save('gdn', 'The gated delta rule in three terms',
     'S new equals alpha S old minus beta times the predicted value alpha S old k '
     'times k transpose, plus beta v k transpose. The terms decay memory, remove '
     'the old association along the key, and write the new association. Read the '
     'updated state with q and carry the state to the next token.', 384, [
    text(300, 32, 'Gated delta: decay, correct, write', 26, weight='bold'),
    text(300, 72, 'S_new = α S_old − β (α S_old k) kᵀ + β v kᵀ', 22),
    '<rect x="18" y="96" width="564" height="67" rx="6" fill="#fff7d6"/>',
    text(44, 124, 'α S_old', 25, GOLD, 'start'),
    text(260, 121, 'Decay the old memory', 21, GOLD, 'start', 'bold'),
    text(260, 148, 'Scale all previous entries.', 18, INK, 'start'),
    '<rect x="18" y="173" width="564" height="67" rx="6" fill="#fceaea"/>',
    text(44, 209, '−β (α S_old k) kᵀ', 22, RED, 'start'),
    text(260, 198, 'Remove the old association', 20, RED, 'start', 'bold'),
    text(260, 225, 'Target this key direction.', 18, INK, 'start'),
    '<rect x="18" y="250" width="564" height="67" rx="6" fill="#e1f3f0"/>',
    text(44, 287, '+β v kᵀ', 25, TEAL, 'start'),
    text(260, 275, 'Write the new association', 20, TEAL, 'start', 'bold'),
    text(260, 302, 'Same 3 × 2 state shape.', 18, INK, 'start'),
    text(300, 346, 'Read: y = S_new q  →  3 × 1 output', 22, BLUE),
    text(300, 374, 'Carry S_new, not y, to the next token.', 18),
])

print('Rendered 3 compact SVG/PNG equation diagrams (600 px wide).')
