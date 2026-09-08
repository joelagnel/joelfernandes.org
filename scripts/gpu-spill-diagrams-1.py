#!/usr/bin/env python3
"""Diagrams for the GPU memory spillover post."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "gpuspill")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c5f8a"
GREEN = "#2e7d52"
ORANGE = "#b8681c"
RED = "#c0392b"
PURPLE = "#6b4c9a"
GREY = "#888"
DARK = "#333"

FONT = 'font-family="Helvetica, Arial, sans-serif"'
MONO = 'font-family="Menlo, Consolas, monospace"'


def head(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" {FONT}>\n'
            f'  <defs>\n'
            f'    <marker id="ar" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{BLUE}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arRed" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{RED}"/>\n'
            f'    </marker>\n'
            f'    <marker id="arGreen" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">\n'
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{GREEN}"/>\n'
            f'    </marker>\n'
            f'  </defs>\n'
            f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n')


def title(x, y, t, size=16):
    return (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
            f'font-weight="bold" fill="{DARK}">{t}</text>\n')


def txt(x, y, t, size=12, fill="#555", anchor="middle", weight="normal", mono=False):
    f = MONO if mono else ""
    return (f'  <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" {f}>{t}</text>\n')


def box(x, y, w, h, fill="#ffffff", stroke=GREY, rx=4, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def line(x1, y1, x2, y2, stroke=GREY, sw=1.4, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n'


# ---------------------------------------------------------------------------
# Diagram 1: who decides, on each platform
# ---------------------------------------------------------------------------
def d1():
    W, H = 880, 400
    s = head(W, H)
    s += title(W / 2, 26, "Where the allocation decision is actually made")

    pw, ph = 380, 290
    for j, (plat, col) in enumerate([("Windows (WDDM)", BLUE), ("Linux", GREEN)]):
        x = 50 + j * (pw + 70)
        s += txt(x + pw / 2, 54, plat, 13.5, col, "middle", "bold")

        # stack
        layers = [
            ("application: cudaMalloc / vkAllocateMemory", "#f7f9fb", 34),
            ("NVIDIA user-mode driver", "#f7f9fb", 30),
        ]
        if j == 0:
            layers.append(("VidMm (Microsoft video memory manager)", "#fdf3e8", 38))
        layers.append(("NVIDIA kernel driver", "#f7f9fb", 30))
        layers.append(("VRAM", "#eef4fa", 30))

        y = 72
        for lab, fill, hh in layers:
            hi = "VidMm" in lab
            s += box(x, y, pw, hh, fill, ORANGE if hi else GREY, 4, 1.6 if hi else 1.2)
            s += txt(x + pw / 2, y + hh / 2 + 4, lab, 10.8,
                     ORANGE if hi else "#555", "middle", "bold" if hi else "normal")
            y += hh + 8

        # verdict
        vy = y + 8
        if j == 0:
            s += txt(x + pw / 2, vy + 10,
                     "VidMm may back an allocation with system pages", 11, ORANGE, "middle", "bold")
            s += txt(x + pw / 2, vy + 28,
                     "and page it in over PCIe when a kernel runs", 10.5, "#666")
        else:
            s += txt(x + pw / 2, vy + 10,
                     "no equivalent layer: the request reaches the", 11, GREEN, "middle", "bold")
            s += txt(x + pw / 2, vy + 28,
                     "driver, and past free VRAM it simply fails", 10.5, "#666")

    s += txt(W / 2, H - 16,
             "The spillover people attribute to the NVIDIA driver on Windows is performed by a Microsoft component.",
             11, "#666")
    s += "</svg>\n"
    open(os.path.join(OUT, "who-decides.svg"), "w").write(s)


# ---------------------------------------------------------------------------
# Diagram 2: measured results
# ---------------------------------------------------------------------------
def d2():
    W, H = 880, 500
    s = head(W, H)
    s += title(W / 2, 26, "A 25.04 GiB model on a 24564 MiB card: six outcomes")

    rows = [
        ("default, -ngl 99", "fails to load", 0, 0, RED,
         "ErrorOutOfDeviceMemory on Vulkan, cudaMalloc failed on CUDA", ""),
        ("--fit on (auto split)", "loads", 21590, 8.94, GREEN,
         "Vulkan, 21590 MiB GPU + 4041 MiB CPU", "8.94 tok/s"),
        ("GGML_VK_ALLOW_SYSMEM_FALLBACK=1", "loads", 25023, 6.32, ORANGE,
         "Vulkan, 25023 MiB of model buffer on a 24564 MiB card", "6.32 tok/s"),
        ("-ngl 40 (manual split)", "loads", 15486, 3.28, BLUE,
         "Vulkan, 15486 MiB GPU + 10145 MiB CPU", "3.28 tok/s"),
        ("-ngl 0 (CPU only)", "loads", 0, 1.64, GREY,
         "all weights in system RAM, no GPU compute", "1.64 tok/s"),
        ("GGML_CUDA_ENABLE_UNIFIED_MEMORY=1", "loads", 25023, 0.204, PURPLE,
         "CUDA, cudaMallocManaged, 21.3 GiB of host RAM consumed", "0.204 tok/s"),
    ]

    y = 62
    rh = 62
    barx = 430
    barw = 300
    maxmb = 26000

    for lab, status, mb, tg, col, detail, tgs in rows:
        s += box(40, y, W - 80, rh - 8, "#fdfdfd", "#ddd", 4, 1.0)
        s += txt(56, y + 20, lab, 11.5, DARK, "start", "bold", mono=True)
        s += txt(56, y + 38, detail, 10.5, "#666", "start")

        if status == "fails to load":
            s += txt(barx + barw / 2, y + 30, "does not load", 12, RED, "middle", "bold")
        else:
            # vram bar
            frac = min(1.0, mb / maxmb) if mb else 0.02
            s += box(barx, y + 12, barw, 16, "#f4f4f4", "#ccc", 2, 1.0)
            # card capacity marker
            capx = barx + barw * (24564 / maxmb)
            s += line(capx, y + 8, capx, y + 32, RED, 1.4, "3,2")
            s += box(barx, y + 12, barw * frac, 16, "#eef4fa" if col != GREY else "#f7f7f7", col, 2, 1.3)
            s += txt(barx + barw + 10, y + 24, tgs, 11.5, col, "start", "bold", mono=True)
        y += rh

    s += txt(barx + barw * (24564 / maxmb), 54, "card capacity", 10, RED, "middle")

    s += txt(W / 2, H - 34,
             "Reference: Qwen3-8B-Q8_0 fits with room to spare and runs at 97 tok/s on Vulkan, 103 tok/s on CUDA.",
             11, "#555")
    s += txt(W / 2, H - 16,
             "Every option that exceeds the card is between 11x and 504x slower, and the bars show why: "
             "the slow ones are not on the GPU.",
             11, "#555")
    s += "</svg>\n"
    open(os.path.join(OUT, "results.svg"), "w").write(s)


if __name__ == "__main__":
    d1()
    d2()
    print("wrote", os.path.abspath(OUT))
