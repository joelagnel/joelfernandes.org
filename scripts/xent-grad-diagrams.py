#!/usr/bin/env python3
"""Generate the SVG figures for the cross-entropy gradient post.

Style matches the earlier backprop posts: white background, Helvetica,
blue (#2c5f8a) strokes, light fills, bold title line at the top.
"""

import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "images", "xent-grad")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c5f8a"
LIGHT = "#eaf2f8"
GREEN = "#dff0e4"
GREY = "#f2f2f2"
RED = "#c0392b"
INK = "#1a1a1a"
MUTED = "#666"

FONT = 'font-family="Helvetica, Arial, sans-serif"'
ITAL_A = '<tspan font-style="italic">a</tspan>'
ITAL_K = '<tspan font-style="italic">k</tspan>'


def head(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT}>
  <defs>
    <marker id="ar" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{BLUE}"/>
    </marker>
    <marker id="arRed" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{RED}"/>
    </marker>
    <marker id="arGrey" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#888"/>
    </marker>
  </defs>
  <rect width="{w}" height="{h}" fill="#ffffff"/>
  <text x="{w//2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">{title}</text>
'''


def sup(base, ex):
    """base with a superscript, e.g. sup('e','a') -> e^a in real SVG."""
    return f'{base}<tspan font-size="0.72em" dy="-0.42em">{ex}</tspan><tspan dy="0.42em"></tspan>'


def frac(cx, cy, num, den, size=18, half=None):
    """A real fraction with a horizontal bar, centred on cx."""
    half = half or max(len(num), len(den)) * size * 0.30
    return (f'  <text x="{cx}" y="{cy - size*0.42:.1f}" text-anchor="middle" '
            f'font-size="{size}" fill="{INK}">{num}</text>\n'
            f'  <line x1="{cx-half:.1f}" y1="{cy:.1f}" x2="{cx+half:.1f}" y2="{cy:.1f}" '
            f'stroke="{INK}" stroke-width="1.4"/>\n'
            f'  <text x="{cx}" y="{cy + size*1.06:.1f}" text-anchor="middle" '
            f'font-size="{size}" fill="{INK}">{den}</text>\n')


def write(name, body):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write(body + "</svg>\n")
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------- figure 1
# The cross-entropy block: what goes in, what comes out, what we want back.

def fig_pipeline():
    w, h = 760, 360
    s = head(w, h, "The cross-entropy block, and the gradient we want back out")

    # input boxes, shifted right to leave a clear return channel on the left
    s += f'''
  <g stroke="{BLUE}" stroke-width="2">
    <rect x="90" y="72" width="170" height="46" rx="6" fill="{LIGHT}"/>
    <rect x="90" y="156" width="170" height="46" rx="6" fill="{LIGHT}"/>
    <rect x="340" y="98" width="140" height="86" rx="6" fill="#ffffff"/>
    <rect x="570" y="114" width="150" height="54" rx="6" fill="{GREEN}"/>
  </g>
  <g font-size="15" text-anchor="middle" fill="{INK}">
    <text x="175" y="94">logits</text>
    <text x="175" y="178">Yb</text>
    <text x="410" y="134">cross</text>
    <text x="410" y="158">entropy</text>
    <text x="645" y="138">loss L</text>
  </g>
  <g font-size="12" text-anchor="middle" fill="{MUTED}">
    <text x="175" y="111">32 \u00d7 27</text>
    <text x="175" y="195">32 correct characters</text>
    <text x="645" y="156">one number</text>
  </g>
  <g stroke="{BLUE}" stroke-width="2" fill="none" marker-end="url(#ar)">
    <line x1="260" y1="95" x2="334" y2="122"/>
    <line x1="260" y1="179" x2="334" y2="160"/>
    <line x1="480" y1="141" x2="564" y2="141"/>
  </g>
'''
    # what the loss means
    s += f'''
  <line x1="410" y1="184" x2="410" y2="212" stroke="#888" stroke-width="2" marker-end="url(#arGrey)" fill="none"/>
  <text x="410" y="232" text-anchor="middle" font-size="13" fill="{MUTED}">L says how likely you were to predict Yb</text>
'''
    # gradient path: down from the loss, along the bottom, up the free left
    # channel, and into the left edge of the logits box
    s += f'''
  <g stroke="{RED}" stroke-width="2" fill="none" marker-end="url(#arRed)" stroke-dasharray="6 4">
    <path d="M 645 172 L 645 288 Q 645 300, 631 300 L 54 300 Q 40 300, 40 286 L 40 109 Q 40 95, 54 95 L 84 95"/>
  </g>
  <text x="420" y="330" text-anchor="middle" font-size="15" fill="{RED}">what we want: dL / dlogits, also 32 \u00d7 27</text>
'''
    write("pipeline.svg", s)


# ---------------------------------------------------------------- figure 2
# Shapes: 32x27 logits -> softmax -> pluck one per row -> -log -> mean

def fig_shapes():
    w, h = 760, 580
    s = head(w, h, "From a 32 \u00d7 27 grid of logits down to a single number")

    gx, gy, gw, gh = 90, 72, 330, 150
    ncol, nrow = 27, 12   # 12 drawn rows standing in for 32

    def grid(x, y, ww, hh, fill="#ffffff", mark=None, markfill=RED):
        cw = ww / ncol
        ch = hh / nrow
        out = f'  <rect x="{x}" y="{y}" width="{ww}" height="{hh}" fill="{fill}" stroke="none"/>\n'
        out += f'  <g stroke="#c8d8e4" stroke-width="0.7">\n'
        for c in range(1, ncol):
            out += f'    <line x1="{x+c*cw:.1f}" y1="{y}" x2="{x+c*cw:.1f}" y2="{y+hh}"/>\n'
        for r in range(1, nrow):
            out += f'    <line x1="{x}" y1="{y+r*ch:.1f}" x2="{x+ww}" y2="{y+r*ch:.1f}"/>\n'
        out += '  </g>\n'
        if mark:
            # inset so the marker never sits on a grid line or the outer border
            ins = 3.0
            out += f'  <g fill="{markfill}">\n'
            for r, c in mark:
                out += (f'    <rect x="{x+c*cw+ins:.1f}" y="{y+r*ch+ins:.1f}" '
                        f'width="{cw-2*ins:.1f}" height="{ch-2*ins:.1f}"/>\n')
            out += '  </g>\n'
        # outer border last, so it always sits on top of any marker
        out += (f'  <rect x="{x}" y="{y}" width="{ww}" height="{hh}" '
                f'fill="none" stroke="{BLUE}" stroke-width="2"/>\n')
        return out

    # keep markers off the outermost ring so none reads as touching the border
    targets = [(0, 4), (1, 19), (2, 1), (3, 11), (4, 24), (5, 7),
               (6, 14), (7, 2), (8, 22), (9, 9), (10, 17), (11, 5)]

    # logits grid
    s += grid(gx, gy, gw, gh)
    s += f'''
  <text x="{gx+gw/2}" y="{gy-12}" text-anchor="middle" font-size="13" fill="{MUTED}">27 output neurons, one per character</text>
  <text x="{gx-14}" y="{gy+gh/2-2}" text-anchor="end" font-size="13" fill="{MUTED}">32</text>
  <text x="{gx-14}" y="{gy+gh/2+15}" text-anchor="end" font-size="13" fill="{MUTED}">examples</text>
  <text x="{gx+gw+16}" y="{gy+gh/2+5}" font-size="15" fill="{INK}">logits</text>
  <text x="{gx+gw+16}" y="{gy+gh/2+23}" font-size="11" fill="#999">rows abbreviated</text>
  <text x="{gx+gw+16}" y="{gy+gh/2+38}" font-size="11" fill="#999">in the drawing</text>
'''
    # arrow down to probs
    y2 = gy + gh + 56
    s += f'''
  <line x1="{gx+gw/2}" y1="{gy+gh+6}" x2="{gx+gw/2}" y2="{y2-6}" stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"/>
  <text x="{gx+gw/2+12}" y="{gy+gh+36}" font-size="13" fill="{MUTED}">softmax along each row</text>
'''
    # probs grid with targets marked
    s += grid(gx, y2, gw, gh, mark=targets)
    s += f'''
  <text x="{gx+gw+16}" y="{y2+gh/2-4}" font-size="15" fill="{INK}">probs</text>
  <text x="{gx+gw+16}" y="{y2+gh/2+14}" font-size="12" fill="{MUTED}">rows sum to 1</text>
'''
    # pluck arrow to column
    colx = 656
    s += f'''
  <text x="{gx+gw/2}" y="{y2+gh+28}" text-anchor="middle" font-size="13" fill="{RED}">the red cell in row i is the character Yb[i] that should have come next</text>
'''
    cw = gw / ncol
    ch = gh / nrow
    s += f'  <rect x="{colx}" y="{y2}" width="{cw*2:.1f}" height="{gh}" fill="#ffffff" stroke="{BLUE}" stroke-width="2"/>\n'
    s += f'  <g fill="{RED}">\n'
    for r in range(nrow):
        s += f'    <rect x="{colx}" y="{y2+r*ch:.1f}" width="{cw*2:.1f}" height="{ch:.1f}"/>\n'
    s += '  </g>\n'
    s += f'''
  <g stroke="#c8d8e4" stroke-width="0.7">
'''
    for r in range(1, nrow):
        s += f'    <line x1="{colx}" y1="{y2+r*ch:.1f}" x2="{colx+cw*2:.1f}" y2="{y2+r*ch:.1f}"/>\n'
    s += '  </g>\n'
    s += f'''
  <line x1="{gx+gw+142}" y1="{y2+gh/2}" x2="{colx-12}" y2="{y2+gh/2}" stroke="{RED}" stroke-width="2" marker-end="url(#arRed)"/>
  <text x="{(gx+gw+142+colx)/2:.0f}" y="{y2+gh/2-14}" text-anchor="middle" font-size="12" fill="{RED}">pluck</text>
  <text x="{colx+cw}" y="{y2-14}" text-anchor="middle" font-size="12" fill="{MUTED}">32 \u00d7 1</text>
'''
    # down to loss
    ly = y2 + gh + 76
    s += f'''
  <line x1="{colx+cw}" y1="{y2+gh+8}" x2="{colx+cw}" y2="{ly-8}" stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"/>
  <text x="{colx-22}" y="{ly+6}" text-anchor="end" font-size="13" fill="{MUTED}">take -log of each,</text>
  <text x="{colx-22}" y="{ly+23}" text-anchor="end" font-size="13" fill="{MUTED}">then average over 32</text>
  <rect x="{colx-6}" y="{ly}" width="62" height="38" rx="6" fill="{GREEN}" stroke="{BLUE}" stroke-width="2"/>
  <text x="{colx+25}" y="{ly+24}" text-anchor="middle" font-size="15" fill="{INK}">L</text>
'''
    write("shapes.svg", s)


# ---------------------------------------------------------------- figure 3
# The K trick: one row, one live variable, everything else frozen.

def fig_ktrick():
    w, h = 740, 372
    s = head(w, h, "One row of 27: only one exponential is a variable, the rest are frozen")

    n = 27
    cw = 22
    x0 = (w - n * cw) / 2
    y0 = 78
    ch = 46
    live = 13

    s += f'  <g stroke="{BLUE}" stroke-width="1.4">\n'
    for c in range(n):
        fill = "#ffe3d6" if c == live else GREY
        s += (f'    <rect x="{x0+c*cw:.1f}" y="{y0}" width="{cw}" height="{ch}" '
              f'fill="{fill}"/>\n')
    s += '  </g>\n'
    s += (f'  <rect x="{x0+live*cw+1.3:.1f}" y="{y0+1.3}" width="{cw-2.6}" height="{ch-2.6}" '
          f'fill="none" stroke="{RED}" stroke-width="2.6"/>\n')

    s += f'''
  <text x="{x0+live*cw+cw/2:.1f}" y="{y0-14}" text-anchor="middle" font-size="15" fill="{RED}">e<tspan font-size="0.72em" dy="-0.42em">a</tspan></text>
  <text x="{x0+live*cw+cw/2:.1f}" y="{y0+ch+22}" text-anchor="middle" font-size="12" fill="{RED}">the one we differentiate</text>
'''
    # braces over the frozen parts
    by = y0 + ch + 44
    s += f'''
  <path d="M {x0:.1f} {by} v 10 h {live*cw-10:.1f} v -10" fill="none" stroke="#888" stroke-width="1.6"/>
  <path d="M {x0+(live+1)*cw+10:.1f} {by+10} v -10 h {(n-live-1)*cw-10:.1f} v 10" fill="none" stroke="#888" stroke-width="1.6"/>
  <text x="{w//2}" y="{by+54}" text-anchor="middle" font-size="14" fill="{INK}">the other 26 exponentials never change when <tspan font-style="italic">a</tspan> changes</text>
  <text x="{w//2}" y="{by+76}" text-anchor="middle" font-size="15" fill="{INK}">call the whole pile <tspan font-style="italic">k</tspan></text>
'''
    # resulting equation
    ey = by + 118
    s += f'''
  <rect x="{w//2-150}" y="{ey-32}" width="300" height="66" rx="8" fill="{LIGHT}" stroke="{BLUE}" stroke-width="2"/>
  <text x="{w//2-43}" y="{ey+8}" text-anchor="middle" font-size="20" fill="{INK}">SM  =</text>
  <text x="{w//2}" y="{ey+64}" text-anchor="middle" font-size="14" fill="{MUTED}">one variable, one constant, and now it is ordinary single variable calculus</text>
'''
    s += frac(w // 2 + 25, ey + 1, sup("e", ITAL_A), sup("e", ITAL_A) + " + " + ITAL_K, size=19, half=40)
    write("k-trick.svg", s)


# ---------------------------------------------------------------- figure 4
# The two cases side by side.

def fig_cases():
    w, h = 760, 430
    s = head(w, h, "Two cases, and the only difference between them is a single \u22121")

    def panel(x, title, sub, expr_lines, result, note):
        out = f'''
  <rect x="{x}" y="56" width="330" height="300" rx="10" fill="#ffffff" stroke="{BLUE}" stroke-width="2"/>
  <rect x="{x}" y="56" width="330" height="40" rx="10" fill="{LIGHT}" stroke="{BLUE}" stroke-width="2"/>
  <text x="{x+165}" y="82" text-anchor="middle" font-size="15" font-weight="bold" fill="{INK}">{title}</text>
  <text x="{x+165}" y="118" text-anchor="middle" font-size="13" fill="{MUTED}">{sub}</text>
'''
        yy = 152
        for line, colour, size in expr_lines:
            out += (f'  <text x="{x+165}" y="{yy}" text-anchor="middle" '
                    f'font-size="{size}" fill="{colour}">{line}</text>\n')
            yy += 30
        out += f'''
  <line x1="{x+40}" y1="{yy+2}" x2="{x+290}" y2="{yy+2}" stroke="#ddd" stroke-width="1.5"/>
  <text x="{x+165}" y="{yy+40}" text-anchor="middle" font-size="22" fill="{RED}">{result}</text>
  <text x="{x+165}" y="{yy+70}" text-anchor="middle" font-size="12" fill="{MUTED}">{note}</text>
'''
        return out

    s += panel(
        30,
        "Case A: a is the correct character",
        "e^a sits on top and underneath",
        [("L = -log( e^a / ( e^a + k ) )", INK, 15),
         ("a appears twice", MUTED, 13),
         ("dL/da = -k / ( k + e^a )", INK, 15),
         ("= -( 1 - SM )", INK, 15)],
        "SM \u2212 1",
        "always negative, so this logit gets pushed up")

    s += panel(
        400,
        "Case B: a is a wrong character",
        "e^a sits underneath only",
        [("L = -log( k / ( e^a + k ) )", INK, 15),
         ("k is constant, it drops out", MUTED, 13),
         ("dL/da = e^a / ( e^a + k )", INK, 15),
         ("nothing to subtract", MUTED, 13)],
        "SM",
        "always positive, so this logit gets pushed down")

    s += f'''
  <text x="{w//2}" y="404" text-anchor="middle" font-size="14" fill="{INK}">both at once:  dL/da = SM - 1 if a is correct, else SM</text>
'''
    write("two-cases.svg", s)


# ---------------------------------------------------------------- figure 5
# Forces: one row of the gradient, drawn as pushes and pulls.

def fig_forces():
    w, h = 860, 430
    s = head(w, h, "One row of the gradient, read as forces on the logits")

    # a plausible softmax row over 27 characters, correct one at index 11
    raw = [0.9, 1.4, 0.6, 2.1, 0.4, 1.9, 0.8, 1.2, 0.5, 2.4, 1.0,
           3.1, 0.7, 1.6, 0.9, 0.3, 2.0, 1.1, 0.6, 1.8, 0.4, 1.3,
           0.9, 2.2, 0.5, 1.5, 0.8]
    e = [math.exp(v) for v in raw]
    tot = sum(e)
    p = [v / tot for v in e]
    correct = 11
    grad = [p[i] - (1.0 if i == correct else 0.0) for i in range(27)]

    n = 27
    cw = 22
    x0 = (w - n * cw) / 2
    axis = 200
    scale = 150.0  # gradient of -1 would be 150px

    s += f'  <line x1="{x0-16:.1f}" y1="{axis}" x2="{x0+n*cw+16:.1f}" y2="{axis}" stroke="#999" stroke-width="1.5"/>\n'
    s += f'  <text x="{x0-24:.1f}" y="{axis+5}" text-anchor="end" font-size="12" fill="{MUTED}">0</text>\n'

    for i, g in enumerate(grad):
        cx = x0 + i * cw + cw / 2
        ln = abs(g) * scale
        if g < 0:
            s += (f'  <rect x="{cx-7:.1f}" y="{axis-ln:.1f}" width="14" height="{ln:.1f}" '
                  f'fill="{RED}" opacity="0.85"/>\n')
        else:
            s += (f'  <rect x="{cx-7:.1f}" y="{axis:.1f}" width="14" height="{max(ln,1.2):.1f}" '
                  f'fill="{BLUE}" opacity="0.65"/>\n')

    s += f'''
  <text x="{x0+correct*cw+cw/2:.1f}" y="{axis-abs(grad[correct])*scale-14:.1f}" text-anchor="middle" font-size="13" fill="{RED}">SM \u2212 1</text>
  <text x="{x0+correct*cw+cw/2:.1f}" y="{axis-abs(grad[correct])*scale-32:.1f}" text-anchor="middle" font-size="12" fill="{MUTED}">the correct character</text>
  <text x="{x0+n*cw+24:.1f}" y="{axis+26}" font-size="13" fill="{BLUE}">SM</text>
  <text x="{x0+n*cw+24:.1f}" y="{axis+42}" font-size="11" fill="{MUTED}">on the other 26</text>
  <line x1="{x0+n*cw+20:.1f}" y1="{axis+22}" x2="{x0+(n-3)*cw+cw/2:.1f}" y2="{axis+10}" stroke="#bbb" stroke-width="1.2"/>
'''
    # explanation band
    s += f'''
  <text x="{w//2}" y="{axis+96}" text-anchor="middle" font-size="14" fill="{INK}">one long pull up on the character that should have come next</text>
  <text x="{w//2}" y="{axis+118}" text-anchor="middle" font-size="14" fill="{INK}">26 small pushes down on everything else</text>
  <rect x="{w//2-215}" y="{axis+140}" width="430" height="46" rx="8" fill="{GREY}" stroke="#ccc"/>
  <text x="{w//2}" y="{axis+168}" text-anchor="middle" font-size="15" fill="{INK}">the whole row adds up to exactly zero</text>
  <text x="{w//2}" y="{axis+208}" text-anchor="middle" font-size="12" fill="{MUTED}">the pull up and the pushes down are always perfectly balanced</text>
'''
    write("forces.svg", s)


# ---------------------------------------------------------------- figure 6
# -log(p): why a low probability on the right answer is expensive.

def fig_neglog():
    w, h = 620, 380
    s = head(w, h, "Why the loss is -log(p) of the correct character")

    ox, oy = 90, 300
    pw, ph = 450, 230

    s += f'''
  <line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#999" stroke-width="1.5"/>
  <line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#999" stroke-width="1.5"/>
  <text x="{ox+pw/2}" y="{oy+40}" text-anchor="middle" font-size="13" fill="{MUTED}">probability the model gave the correct character</text>
  <text x="{ox-58}" y="{oy-ph/2}" text-anchor="middle" font-size="13" fill="{MUTED}" transform="rotate(-90 {ox-58} {oy-ph/2})">loss</text>
  <text x="{ox}" y="{oy+20}" text-anchor="middle" font-size="12" fill="{MUTED}">0</text>
  <text x="{ox+pw}" y="{oy+20}" text-anchor="middle" font-size="12" fill="{MUTED}">1</text>
'''
    ymax = 4.0
    pts = []
    N = 300
    for i in range(N + 1):
        pr = 0.018 + (1.0 - 0.018) * i / N
        val = -math.log(pr)
        if val > ymax:
            continue
        x = ox + pr * pw
        y = oy - (val / ymax) * ph
        pts.append(f"{x:.1f},{y:.1f}")
    s += f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>\n'

    # annotations
    s += f'''
  <circle cx="{ox+0.05*pw}" cy="{oy-(-math.log(0.05)/ymax)*ph}" r="5" fill="{RED}"/>
  <text x="{ox+0.05*pw+14}" y="{oy-(-math.log(0.05)/ymax)*ph-6}" font-size="12" fill="{RED}">p = 0.05, loss = 3.0</text>
  <circle cx="{ox+0.95*pw}" cy="{oy-(-math.log(0.95)/ymax)*ph}" r="5" fill="{BLUE}"/>
  <text x="{ox+0.95*pw-16}" y="{oy-(-math.log(0.95)/ymax)*ph-30}" text-anchor="end" font-size="12" fill="{BLUE}">p = 0.95, loss = 0.05</text>
  <text x="{ox+pw/2+30}" y="{oy-ph+8}" font-size="13" fill="{MUTED}">confident and wrong is expensive</text>
  <text x="{ox+pw/2+30}" y="{oy-ph+28}" font-size="13" fill="{MUTED}">confident and right costs almost nothing</text>
'''
    write("neglog.svg", s)


fig_pipeline()
fig_shapes()
fig_ktrick()
fig_cases()
fig_forces()
fig_neglog()


# ---------------------------------------------------------------- figure 7
# Why the signs come out the way they do: loss as a function of one logit.

def fig_why_signs():
    w, h = 780, 440
    s = head(w, h, "Why one gradient carries a \u22121 and the other does not")

    pw, ph = 300, 200

    def axes(ox, oy, xlabel):
        return f'''
  <line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#999" stroke-width="1.5"/>
  <line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#999" stroke-width="1.5"/>
  <text x="{ox+pw/2}" y="{oy+38}" text-anchor="middle" font-size="13" fill="{MUTED}">{xlabel}</text>
  <text x="{ox-30}" y="{oy-ph/2}" text-anchor="middle" font-size="13" fill="{MUTED}" transform="rotate(-90 {ox-30} {oy-ph/2})">loss</text>
'''

    def curve(ox, oy, fn, lo, hi, vmax):
        pts = []
        N = 220
        for i in range(N + 1):
            av = lo + (hi - lo) * i / N
            val = fn(av)
            x = ox + (av - lo) / (hi - lo) * pw
            y = oy - min(val / vmax, 1.0) * ph
            pts.append(f"{x:.1f},{y:.1f}")
        return f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>\n'

    K = 6.0
    vmax = 3.2      # left panel scale
    vmaxR = 1.05    # right panel scale, so its curve fills the box too

    # ---- left panel: the correct character's logit
    ox, oy = 88, 272
    s += axes(ox, oy, "the correct character's logit")
    fL = lambda a: math.log(1 + K * math.exp(-a))
    lo, hi = -1.6, 3.4
    s += curve(ox, oy, fL, lo, hi, vmax)

    a0 = 0.15
    p0 = math.exp(a0) / (math.exp(a0) + K)
    y0 = oy - min(fL(a0) / vmax, 1.0) * ph
    x0 = ox + (a0 - lo) / (hi - lo) * pw
    # tangent, slope in data units converted to pixels
    slope = (p0 - 1.0)
    dx = 62.0
    dax = dx / pw * (hi - lo)
    dy = slope * dax / vmax * ph
    s += f'''
  <line x1="{x0-dx}" y1="{y0+dy}" x2="{x0+dx}" y2="{y0-dy}" stroke="{RED}" stroke-width="2.2"/>
  <circle cx="{x0:.1f}" cy="{y0:.1f}" r="5" fill="{RED}"/>
  <text x="{ox+pw-4}" y="{oy-ph+16}" text-anchor="end" font-size="13" fill="{RED}">slope = p \u2212 1</text>
  <text x="{ox+pw-4}" y="{oy-ph+33}" text-anchor="end" font-size="13" fill="{RED}">which is negative</text>
  <line x1="{x0}" y1="{oy+14}" x2="{x0+82}" y2="{oy+14}" stroke="{BLUE}" stroke-width="2.5" marker-end="url(#ar)"/>
  <text x="{x0+90}" y="{oy+19}" font-size="12" fill="{BLUE}">push it up</text>
  <text x="{ox+pw/2}" y="{oy+62}" text-anchor="middle" font-size="12.5" fill="{INK}">raise this logit and the loss falls,</text>
  <text x="{ox+pw/2}" y="{oy+79}" text-anchor="middle" font-size="12.5" fill="{INK}">so descent raises it</text>
'''

    # ---- right panel: a wrong character's logit
    ox2 = 438
    s += axes(ox2, oy, "a wrong character's logit")
    Kc = 6.0
    fR = lambda a: math.log(math.exp(a) + Kc) - math.log(Kc) + 0.06
    lo2, hi2 = -3.4, 2.15
    s += curve(ox2, oy, fR, lo2, hi2, vmaxR)

    a1 = 1.30
    p1 = math.exp(a1) / (math.exp(a1) + Kc)
    y1 = oy - min(fR(a1) / vmaxR, 1.0) * ph
    x1 = ox2 + (a1 - lo2) / (hi2 - lo2) * pw
    slope1 = p1
    dxR = 26.0
    dax1 = dxR / pw * (hi2 - lo2)
    dy1 = slope1 * dax1 / vmaxR * ph
    s += f'''
  <line x1="{x1-dxR}" y1="{y1+dy1}" x2="{x1+dxR}" y2="{y1-dy1}" stroke="{RED}" stroke-width="2.2"/>
  <circle cx="{x1:.1f}" cy="{y1:.1f}" r="5" fill="{RED}"/>
  <text x="{x1-46}" y="{y1-30}" text-anchor="end" font-size="13" fill="{RED}">slope = p</text>
  <text x="{x1-46}" y="{y1-14}" text-anchor="end" font-size="13" fill="{RED}">which is positive</text>
  <line x1="{x1}" y1="{oy+14}" x2="{x1-82}" y2="{oy+14}" stroke="{BLUE}" stroke-width="2.5" marker-end="url(#ar)"/>
  <text x="{x1-90}" y="{oy+19}" text-anchor="end" font-size="12" fill="{BLUE}">push it down</text>
  <text x="{ox2+pw/2}" y="{oy+62}" text-anchor="middle" font-size="12.5" fill="{INK}">raise this logit and the loss climbs,</text>
  <text x="{ox2+pw/2}" y="{oy+79}" text-anchor="middle" font-size="12.5" fill="{INK}">so descent lowers it</text>
'''

    s += f'''
  <rect x="{w//2-300}" y="{oy+98}" width="600" height="42" rx="8" fill="{GREY}" stroke="#ccc"/>
  <text x="{w//2}" y="{oy+124}" text-anchor="middle" font-size="14" fill="{INK}">descent moves against the slope, so the sign of the gradient already encodes the direction</text>
'''
    write("why-signs.svg", s)


# ---------------------------------------------------------------- figure 8
# Magnitude: how hard the push is, as a function of how wrong you were.

def fig_magnitude():
    w, h = 720, 440
    s = head(w, h, "How hard the push is depends on how wrong the model was")

    ox, oy = 90, 280
    pw, ph = 470, 200

    s += f'''
  <line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#999" stroke-width="1.5"/>
  <line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#999" stroke-width="1.5"/>
  <text x="{ox-42}" y="{oy-ph/2}" text-anchor="middle" font-size="13" fill="{MUTED}" transform="rotate(-90 {ox-42} {oy-ph/2})">size of the push</text>
  <text x="{ox}" y="{oy+20}" text-anchor="middle" font-size="12" fill="{MUTED}">0</text>
  <text x="{ox+pw}" y="{oy+20}" text-anchor="middle" font-size="12" fill="{MUTED}">1</text>
  <text x="{ox-12}" y="{oy-ph+5}" text-anchor="end" font-size="12" fill="{MUTED}">1</text>
  <text x="{ox-12}" y="{oy+4}" text-anchor="end" font-size="12" fill="{MUTED}">0</text>
  <text x="{ox+pw/2}" y="{oy-ph-14}" text-anchor="middle" font-size="13" fill="{MUTED}">p, the probability given to the correct character</text>
'''
    # the line, size of the push = 1 - p
    s += (f'  <line x1="{ox}" y1="{oy-ph}" x2="{ox+pw}" y2="{oy}" '
          f'stroke="{BLUE}" stroke-width="2.5"/>\n')
    s += (f'  <text x="{ox+pw*0.60}" y="{oy-ph*0.30+30}" font-size="14" '
          f'fill="{BLUE}">1 \u2212 p</text>\n')

    # three sample points, each with a drop line to a label under the axis
    marks = [
        (0.06, "badly wrong", "nearly the whole push"),
        (0.50, "unsure", "half the push"),
        (0.95, "already right", "almost no push"),
    ]
    for frac, top, bottom in marks:
        px = ox + frac * pw
        py = oy - (1 - frac) * ph
        # keep the centred text block inside the canvas
        tx = min(max(px, 96.0), w - 96.0)
        s += f'''
  <line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{oy+46}" stroke="#bbb" stroke-width="1.2" stroke-dasharray="4 3"/>
  <circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{RED}"/>
  <text x="{tx:.1f}" y="{oy+66}" text-anchor="middle" font-size="12.5" fill="{INK}">{top}</text>
  <text x="{tx:.1f}" y="{oy+83}" text-anchor="middle" font-size="12" fill="{MUTED}">{bottom}</text>
'''

    s += f'''
  <text x="{w//2}" y="{oy+122}" text-anchor="middle" font-size="14" fill="{INK}">a model that is already correct gets left alone, which is what you want</text>
'''
    write("magnitude.svg", s)



fig_why_signs()
fig_magnitude()
