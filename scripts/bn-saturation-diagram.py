#!/usr/bin/env python3
"""Figure for the saturation section of the batchnorm post.

Left panel:  tanh(z), with the useful band and the two saturated tails marked.
Right panel: its derivative 1 - tanh(z)^2, which is what actually multiplies
             the gradient on the way back.
"""

import math
import os

OUT = "/home/joel/.openclaw/workspace/joelfernandes.org/images/bn-grad"
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c5f8a"
RED = "#c0392b"
INK = "#1a1a1a"
MUTED = "#666"
FLAT = "#f2ece4"          # saturated band fill
GOOD = "#e8f2e9"          # useful band fill

FONT = 'font-family="Helvetica, Arial, sans-serif"'

W, H = 1080, 500
PW, PH = 390, 275          # plot box
OX1, OY = 95, 395          # left plot origin (bottom-left)
OX2 = 625

ZMIN, ZMAX = -10.5, 10.5


def sx(ox, z):
    return ox + (z - ZMIN) / (ZMAX - ZMIN) * PW


def head():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" {FONT}>
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 9 3, 0 6" fill="{RED}"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="#ffffff"/>
  <text x="{W//2}" y="34" text-anchor="middle" font-size="24.8" font-weight="bold" fill="#333">Why the scale of the pre-activation matters</text>
'''


s = head()

# ---------------------------------------------------------------- left panel
# shaded bands: saturated tails and the useful middle
for z0, z1, fill in [(ZMIN, -2.5, FLAT), (-2.5, 2.5, GOOD), (2.5, ZMAX, FLAT)]:
    x0, x1 = sx(OX1, z0), sx(OX1, z1)
    s += (f'  <rect x="{x0:.1f}" y="{OY-PH}" width="{x1-x0:.1f}" height="{PH}" '
          f'fill="{fill}"/>\n')

# axes
s += f'  <line x1="{OX1}" y1="{OY-PH/2:.1f}" x2="{OX1+PW}" y2="{OY-PH/2:.1f}" stroke="#bbb" stroke-width="1.2"/>\n'
s += f'  <line x1="{sx(OX1,0):.1f}" y1="{OY}" x2="{sx(OX1,0):.1f}" y2="{OY-PH}" stroke="#bbb" stroke-width="1.2"/>\n'

# tanh curve, mapped so y in [-1,1] fills the box
pts = []
N = 400
for i in range(N + 1):
    z = ZMIN + (ZMAX - ZMIN) * i / N
    t = math.tanh(z)
    pts.append(f"{sx(OX1,z):.1f},{OY - PH/2 - t*(PH/2 - 22):.1f}")
s += f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>\n'

s += f'''  <text x="{OX1+PW/2:.0f}" y="{OY+52}" text-anchor="middle" font-size="19.4" fill="{MUTED}">pre-activation z</text>
  <text x="{OX1-48}" y="{OY-PH/2:.0f}" text-anchor="middle" font-size="19.4" fill="{MUTED}" transform="rotate(-90 {OX1-48} {OY-PH/2:.0f})">tanh(z)</text>
  <text x="{OX1-12}" y="{OY-PH/2-(PH/2-22)+4:.0f}" text-anchor="end" font-size="17.1" fill="{MUTED}">+1</text>
  <text x="{OX1-12}" y="{OY-PH/2+4:.0f}" text-anchor="end" font-size="17.1" fill="{MUTED}">0</text>
  <text x="{OX1-12}" y="{OY-PH/2+(PH/2-22)+4:.0f}" text-anchor="end" font-size="17.1" fill="{MUTED}">-1</text>
  <text x="{sx(OX1,0):.0f}" y="{OY+24}" text-anchor="middle" font-size="17.1" fill="{MUTED}">0</text>
  <text x="{sx(OX1,-8):.0f}" y="{OY+24}" text-anchor="middle" font-size="17.1" fill="{MUTED}">-8</text>
  <text x="{sx(OX1,8):.0f}" y="{OY+24}" text-anchor="middle" font-size="17.1" fill="{MUTED}">8</text>
  <text x="{sx(OX1,9.6):.0f}" y="{OY+24}" text-anchor="middle" font-size="17.1" fill="{RED}">9</text>
  <text x="{sx(OX1,0):.0f}" y="{OY-PH-12}" text-anchor="middle" font-size="17.8" fill="#4a7a4f">useful range</text>
  <text x="{sx(OX1,6.2):.0f}" y="{OY-PH-12}" text-anchor="middle" font-size="17.8" fill="#8a6a3a">saturated</text>
  <text x="{sx(OX1,-6.2):.0f}" y="{OY-PH-12}" text-anchor="middle" font-size="17.8" fill="#8a6a3a">saturated</text>
'''

# mark z = 8 and z = 9, which the text quotes
for z, lab in [(8, "8"), (9, "9")]:
    t = math.tanh(z)
    px, py = sx(OX1, z), OY - PH/2 - t*(PH/2 - 22)
    s += f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="4.0" fill="{RED}" stroke="#ffffff" stroke-width="1.4"/>\n'

s += f'''  <text x="{sx(OX1,9.9):.0f}" y="{OY-PH/2+52:.0f}" text-anchor="end" font-size="17.1" fill="{RED}">tanh(8) = 0.99999977</text>
  <text x="{sx(OX1,9.9):.0f}" y="{OY-PH/2+75:.0f}" text-anchor="end" font-size="17.1" fill="{RED}">tanh(9) = 0.99999997</text>
  <text x="{sx(OX1,9.9):.0f}" y="{OY-PH/2+98:.0f}" text-anchor="end" font-size="17.1" fill="{MUTED}">a whole unit apart, and</text>
  <text x="{sx(OX1,9.9):.0f}" y="{OY-PH/2+121:.0f}" text-anchor="end" font-size="17.1" fill="{MUTED}">the outputs barely differ</text>
'''

# ---------------------------------------------------------------- right panel
for z0, z1, fill in [(ZMIN, -2.5, FLAT), (-2.5, 2.5, GOOD), (2.5, ZMAX, FLAT)]:
    x0, x1 = sx(OX2, z0), sx(OX2, z1)
    s += (f'  <rect x="{x0:.1f}" y="{OY-PH}" width="{x1-x0:.1f}" height="{PH}" '
          f'fill="{fill}"/>\n')

s += f'  <line x1="{OX2}" y1="{OY-26}" x2="{OX2+PW}" y2="{OY-26}" stroke="#d8d8d8" stroke-width="1.0"/>\n'
s += f'  <line x1="{sx(OX2,0):.1f}" y1="{OY}" x2="{sx(OX2,0):.1f}" y2="{OY-PH}" stroke="#bbb" stroke-width="1.2"/>\n'

pts = []
for i in range(N + 1):
    z = ZMIN + (ZMAX - ZMIN) * i / N
    d = 1.0 - math.tanh(z)**2
    pts.append(f"{sx(OX2,z):.1f},{OY - 26 - d*(PH - 44):.1f}")
s += f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>\n'

s += f'''  <text x="{OX2+PW/2:.0f}" y="{OY+52}" text-anchor="middle" font-size="19.4" fill="{MUTED}">pre-activation z</text>
  <text x="{OX2-48}" y="{OY-PH/2:.0f}" text-anchor="middle" font-size="19.4" fill="{MUTED}" transform="rotate(-90 {OX2-48} {OY-PH/2:.0f})">gradient factor</text>
  <text x="{OX2-12}" y="{OY-26-(PH-44)+4:.0f}" text-anchor="end" font-size="17.1" fill="{MUTED}">+1</text>
  <text x="{OX2-12}" y="{OY-22}" text-anchor="end" font-size="17.1" fill="{MUTED}">0</text>
  <text x="{sx(OX2,0):.0f}" y="{OY+24}" text-anchor="middle" font-size="17.1" fill="{MUTED}">0</text>
  <text x="{sx(OX2,-8):.0f}" y="{OY+24}" text-anchor="middle" font-size="17.1" fill="{MUTED}">-8</text>
  <text x="{sx(OX2,8):.0f}" y="{OY+24}" text-anchor="middle" font-size="17.1" fill="{MUTED}">8</text>
  <text x="{sx(OX2,0):.0f}" y="{OY-PH-12}" text-anchor="middle" font-size="17.8" fill="#4a7a4f">gradient flows</text>
  <text x="{sx(OX2,6.9):.0f}" y="{OY-PH-12}" text-anchor="middle" font-size="17.8" fill="#8a6a3a">nothing back</text>
  <text x="{sx(OX2,-6.9):.0f}" y="{OY-PH-12}" text-anchor="middle" font-size="17.8" fill="#8a6a3a">nothing back</text>
'''

# annotate the collapse at z = 8
d8 = 1.0 - math.tanh(8)**2
px, py = sx(OX2, 8), OY - 26 - d8*(PH - 44)
s += f'''  <circle cx="{px:.1f}" cy="{py:.1f}" r="4.0" fill="{RED}" stroke="#ffffff" stroke-width="1.4"/>
  <line x1="{px-8:.1f}" y1="{py-58:.1f}" x2="{px-1:.1f}" y2="{py-8:.1f}" stroke="{RED}" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="{sx(OX2,9.4):.0f}" y="{py-74:.0f}" text-anchor="end" font-size="17.1" fill="{RED}">at z = 8 the factor is</text>
  <text x="{sx(OX2,9.4):.0f}" y="{py-59:.0f}" text-anchor="end" font-size="17.1" fill="{RED}">0.00000045</text>
  <text x="{sx(OX2,-9.6):.0f}" y="{OY-PH+34:.0f}" text-anchor="start" font-size="17.1" fill="{MUTED}">peak is 1.0 at z = 0</text>
'''

s += f'''  <text x="{W//2}" y="{H-16}" text-anchor="middle" font-size="19.4" fill="{INK}">the backward pass multiplies by the right-hand curve, so a unit sitting in a shaded band learns almost nothing</text>
'''

s += "</svg>\n"
open(os.path.join(OUT, "saturation.svg"), "w").write(s)
print("wrote", os.path.join(OUT, "saturation.svg"))
