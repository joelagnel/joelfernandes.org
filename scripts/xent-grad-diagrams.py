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
    <text x="175" y="111">32 x 27</text>
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
  <text x="420" y="330" text-anchor="middle" font-size="15" fill="{RED}">what we want: dL / dlogits, also 32 x 27</text>
'''
    write("pipeline.svg", s)


# ---------------------------------------------------------------- figure 2
# Shapes: 32x27 logits -> softmax -> pluck one per row -> -log -> mean

def fig_shapes():
    w, h = 760, 580
    s = head(w, h, "From a 32 x 27 grid of logits down to a single number")

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
  <text x="{colx+cw}" y="{y2-14}" text-anchor="middle" font-size="12" fill="{MUTED}">32 x 1</text>
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
    live = 11

    s += f'  <g stroke="{BLUE}" stroke-width="1.4">\n'
    for c in range(n):
        fill = "#ffe3d6" if c == live else GREY
        s += (f'    <rect x="{x0+c*cw:.1f}" y="{y0}" width="{cw}" height="{ch}" '
              f'fill="{fill}"/>\n')
    s += '  </g>\n'
    s += (f'  <rect x="{x0+live*cw:.1f}" y="{y0}" width="{cw}" height="{ch}" '
          f'fill="none" stroke="{RED}" stroke-width="3"/>\n')

    s += f'''
  <text x="{x0+live*cw+cw/2:.1f}" y="{y0-14}" text-anchor="middle" font-size="15" fill="{RED}">e^a</text>
  <text x="{x0+live*cw+cw/2:.1f}" y="{y0+ch+22}" text-anchor="middle" font-size="12" fill="{RED}">the one we differentiate</text>
'''
    # braces over the frozen parts
    by = y0 + ch + 44
    s += f'''
  <path d="M {x0:.1f} {by} v 10 h {live*cw-8:.1f} v 10" fill="none" stroke="#888" stroke-width="1.6"/>
  <path d="M {x0+(live+1)*cw+8:.1f} {by} v 10 h {(n-live-1)*cw-8:.1f} v 10" fill="none" stroke="#888" stroke-width="1.6"/>
  <text x="{w//2}" y="{by+54}" text-anchor="middle" font-size="14" fill="{INK}">the other 26 exponentials never change when a changes</text>
  <text x="{w//2}" y="{by+76}" text-anchor="middle" font-size="15" fill="{INK}">call the whole pile K</text>
'''
    # resulting equation
    ey = by + 118
    s += f'''
  <rect x="{w//2-190}" y="{ey-30}" width="380" height="62" rx="8" fill="{LIGHT}" stroke="{BLUE}" stroke-width="2"/>
  <text x="{w//2}" y="{ey+8}" text-anchor="middle" font-size="20" fill="{INK}">SM  =  e^a / ( e^a + K )</text>
  <text x="{w//2}" y="{ey+64}" text-anchor="middle" font-size="14" fill="{MUTED}">one variable, one constant, and now it is ordinary single variable calculus</text>
'''
    write("k-trick.svg", s)


# ---------------------------------------------------------------- figure 4
# The two cases side by side.

def fig_cases():
    w, h = 760, 430
    s = head(w, h, "Two cases, and the only difference between them is a single -1")

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
        [("L = -log( e^a / ( e^a + K ) )", INK, 15),
         ("a appears twice", MUTED, 13),
         ("dL/da = -K / ( K + e^a )", INK, 15),
         ("= -( 1 - SM )", INK, 15)],
        "SM - 1",
        "always negative, so this logit gets pushed up")

    s += panel(
        400,
        "Case B: a is a wrong character",
        "e^a sits underneath only",
        [("L = -log( C / ( e^a + K ) )", INK, 15),
         ("C is constant, it drops out", MUTED, 13),
         ("dL/da = e^a / ( e^a + K )", INK, 15),
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
    w, h = 740, 420
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
  <text x="{x0+correct*cw+cw/2:.1f}" y="{axis-abs(grad[correct])*scale-14:.1f}" text-anchor="middle" font-size="13" fill="{RED}">SM - 1</text>
  <text x="{x0+correct*cw+cw/2:.1f}" y="{axis-abs(grad[correct])*scale-32:.1f}" text-anchor="middle" font-size="12" fill="{MUTED}">the correct character</text>
  <text x="{x0+n*cw+26:.1f}" y="{axis+30}" font-size="13" fill="{BLUE}">SM</text>
  <text x="{x0+n*cw+26:.1f}" y="{axis+48}" font-size="11" fill="{MUTED}">the other 26</text>
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
  <text x="{ox+0.95*pw-10}" y="{oy-(-math.log(0.95)/ymax)*ph-12}" text-anchor="end" font-size="12" fill="{BLUE}">p = 0.95, loss = 0.05</text>
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
