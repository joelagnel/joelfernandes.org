#!/usr/bin/env python3
"""Shared SVG helpers for the wte / lm_head weight tying post."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "wte-lmhead")

BLUE = "#2c5f8a"
GREEN = "#2e7d52"
ORANGE = "#b8681c"
RED = "#c0392b"
PURPLE = "#6b4d8f"
GREY = "#888"
LGREY = "#ccc"
DARK = "#333"

FONT = 'font-family="Helvetica, Arial, sans-serif"'
MONO = 'font-family="SFMono-Regular, Consolas, Menlo, monospace"'


def head(w, h):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
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
        f'    <marker id="arOrange" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
        f'      <polygon points="0 0, 10 3.5, 0 7" fill="{ORANGE}"/>\n'
        f'    </marker>\n'
        f'    <marker id="arPurple" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
        f'      <polygon points="0 0, 10 3.5, 0 7" fill="{PURPLE}"/>\n'
        f'    </marker>\n'
        f'  </defs>\n'
        f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n'
    )


def esc(t):
    """SVG is XML, so bare &, < and > in label text make the file unparseable.
    Every text helper routes through here."""
    return (str(t).replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;"))


def title(x, y, t, size=16):
    return (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
            f'font-weight="bold" fill="{DARK}">{esc(t)}</text>\n')


def txt(x, y, t, size=12, fill="#555", anchor="middle", weight="normal", style="normal", mono=False):
    f = MONO if mono else ""
    return (f'  <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" font-style="{style}" {f}>{esc(t)}</text>\n')


def box(x, y, w, h, fill="#ffffff", stroke=GREY, rx=4, sw=1.4, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def line(x1, y1, x2, y2, stroke=GREY, sw=1.4, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n'


def path(d, stroke=GREY, sw=1.4, fill="none", dash=None, marker=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'  <path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{ds}{m}/>\n'


def circle(cx, cy, r, fill=BLUE, stroke="none", sw=1.0, op=1.0):
    return (f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" fill-opacity="{op}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>\n')


def save(name, s):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(s + "</svg>\n")
    print("wrote", os.path.normpath(p))
