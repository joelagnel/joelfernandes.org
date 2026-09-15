#!/usr/bin/env python3
"""Render local SVG figures for the build-nanoGPT activation draft."""

import html
import json
import math
from pathlib import Path


SITE_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path("/home/joel/repo/research-reports/nanogpt-activation-comparison")
OUT = SITE_ROOT / "images" / "nanogpt-activation"

COLORS = {"gelu": "#1f77b4", "swiglu": "#c94f1a", "relu": "#7442a8"}
LABELS = {"gelu": "GELU", "swiglu": "SwiGLU", "relu": "ReLU"}


def write_svg(path, title, description, body, width, height):
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(description)}</desc>
  <style>
    .title {{ font: 700 40px Arial, sans-serif; fill: #20252b; }}
    .subtitle {{ font: 24px Arial, sans-serif; fill: #4b5563; }}
    .label {{ font: 28px Arial, sans-serif; fill: #20252b; }}
    .small {{ font: 24px Arial, sans-serif; fill: #4b5563; }}
    .axis {{ stroke: #3b4550; stroke-width: 2; }}
    .grid {{ stroke: #d8dee5; stroke-width: 1; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
{body}
</svg>''',
        encoding="utf-8",
    )


def points(series, xscale, yscale):
    return " ".join(f"{xscale(x):.2f},{yscale(y):.2f}" for x, y in series)


def gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def render_gelu_curve():
    width, height = 1080, 700
    left, right, top, bottom = 125, 70, 120, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -0.25, 4.1
    xscale = lambda x: left + (x - x_min) / (x_max - x_min) * plot_w
    yscale = lambda y: top + (y_max - y) / (y_max - y_min) * plot_h

    body = [
        '<text x="540" y="48" text-anchor="middle" class="title">GELU is a smooth input-dependent scale</text>',
        '<text x="540" y="78" text-anchor="middle" class="subtitle">GELU(x) = x Φ(x), where Φ is the standard normal cumulative distribution</text>',
    ]
    for y in (0, 1, 2, 3, 4):
        yy = yscale(y)
        body.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" class="grid"/>')
        body.append(f'<text x="{left-18}" y="{yy+6:.2f}" text-anchor="end" class="small">{y}</text>')
    for x in (-4, -2, 0, 2, 4):
        xx = xscale(x)
        body.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" class="grid"/>')
        body.append(f'<text x="{xx:.2f}" y="{height-bottom+31}" text-anchor="middle" class="small">{x}</text>')
    body.extend([
        f'<line x1="{left}" y1="{yscale(0):.2f}" x2="{width-right}" y2="{yscale(0):.2f}" class="axis"/>',
        f'<line x1="{xscale(0):.2f}" y1="{top}" x2="{xscale(0):.2f}" y2="{height-bottom}" class="axis"/>',
    ])
    identity = [(x, x) for x in [x_min + i * (x_max - x_min) / 160 for i in range(161)]]
    curve = [(x, gelu(x)) for x in [x_min + i * (x_max - x_min) / 160 for i in range(161)]]
    body.append(f'<polyline points="{points(identity, xscale, yscale)}" fill="none" stroke="#9ca3af" stroke-width="3" stroke-dasharray="9 8"/>')
    body.append(f'<polyline points="{points(curve, xscale, yscale)}" fill="none" stroke="#1f77b4" stroke-width="6"/>')
    body.extend([
        f'<text x="{width-right-8}" y="{yscale(3.35):.2f}" text-anchor="end" class="label" fill="#1f77b4">GELU(x)</text>',
        f'<text x="{width-right-8}" y="{yscale(2.95):.2f}" text-anchor="end" class="small">dashed: identity x</text>',
        f'<text x="{left + plot_w/2:.2f}" y="{height-28}" text-anchor="middle" class="label">input x</text>',
        f'<text x="32" y="{top + plot_h/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 32 {top + plot_h/2:.2f})">output</text>',
    ])
    write_svg(
        OUT / "gelu-curve.svg",
        "GELU curve",
        "A blue curve plots GELU from input minus four to four. A dashed gray line shows the identity function. Positive inputs pass through smoothly while negative inputs are reduced rather than abruptly set to zero.",
        "\n".join(body), width, height,
    )


def load_evaluations(activation):
    path = REPORT_ROOT / "runs" / activation / "metrics.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["kind"] == "eval" and row["tokens"] > 0:
            rows.append(row)
    return rows


def render_training_validation():
    width, height = 1080, 870
    left, right, top, bottom = 150, 70, 230, 145
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 0, 190_000_000
    y_min, y_max = 4.1, 6.6
    xscale = lambda x: left + (x - x_min) / (x_max - x_min) * plot_w
    yscale = lambda y: top + (y_max - y) / (y_max - y_min) * plot_h
    shared = json.loads((REPORT_ROOT / "plots" / "comparison.json").read_text())["shared_tokens"]
    shared_tokens = shared["tokens"]

    body = [
        '<text x="540" y="48" text-anchor="middle" class="title">Loss decay in three build-nanoGPT runs</text>',
        '<text x="540" y="80" text-anchor="middle" class="subtitle">Solid: fixed held-out validation loss. Dashed: fixed training-prefix loss. Initialization is omitted.</text>',
        '<text x="150" y="122" class="label">Color</text>',
        '<line x1="238" y1="116" x2="294" y2="116" stroke="#1f77b4" stroke-width="6"/>',
        '<text x="306" y="123" class="label">GELU</text>',
        '<line x1="416" y1="116" x2="472" y2="116" stroke="#c94f1a" stroke-width="6"/>',
        '<text x="484" y="123" class="label">SwiGLU</text>',
        '<line x1="630" y1="116" x2="686" y2="116" stroke="#7442a8" stroke-width="6"/>',
        '<text x="698" y="123" class="label">ReLU</text>',
        '<text x="150" y="166" class="label">Line</text>',
        '<line x1="238" y1="160" x2="294" y2="160" stroke="#20252b" stroke-width="6"/>',
        '<text x="306" y="167" class="label">held-out validation</text>',
        '<line x1="564" y1="160" x2="620" y2="160" stroke="#20252b" stroke-width="5" stroke-dasharray="12 9"/>',
        '<text x="632" y="167" class="label">fixed training prefix</text>',
        '<circle cx="926" cy="160" r="9" fill="white" stroke="#20252b" stroke-width="4"/>',
        '<text x="944" y="167" class="small">interpolated</text>',
        '<text x="1010" y="207" text-anchor="end" class="small">vertical marker: shared 176.2M-token comparison</text>',
    ]
    for y in (4.5, 5.0, 5.5, 6.0, 6.5):
        yy = yscale(y)
        body.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" class="grid"/>')
        body.append(f'<text x="{left-18}" y="{yy+7:.2f}" text-anchor="end" class="small">{y:.1f}</text>')
    for x, label in ((0, "0"), (50_000_000, "50M"), (100_000_000, "100M"), (150_000_000, "150M")):
        xx = xscale(x)
        body.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" class="grid"/>')
        body.append(f'<text x="{xx:.2f}" y="{height-bottom+34}" text-anchor="middle" class="small">{label}</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    shared_x = xscale(shared_tokens)
    body.append(f'<line x1="{shared_x:.2f}" y1="{top}" x2="{shared_x:.2f}" y2="{height-bottom}" stroke="#374151" stroke-width="3" stroke-dasharray="10 9"/>')
    for activation in ("gelu", "swiglu", "relu"):
        color = COLORS[activation]
        rows = load_evaluations(activation)
        train = [(row["tokens"], row["train_loss"]) for row in rows]
        validation = [(row["tokens"], row["val_loss"]) for row in rows]
        body.append(f'<polyline points="{points(train, xscale, yscale)}" fill="none" stroke="{color}" stroke-width="4" stroke-dasharray="12 9" opacity="0.88"/>')
        body.append(f'<polyline points="{points(validation, xscale, yscale)}" fill="none" stroke="{color}" stroke-width="6"/>')
        value = shared[f"{activation}_val_loss"]
        fill = color if activation == "swiglu" else "white"
        body.append(f'<circle cx="{shared_x:.2f}" cy="{yscale(value):.2f}" r="10" fill="{fill}" stroke="{color}" stroke-width="5"/>')
    body.extend([
        f'<text x="{left + plot_w/2:.2f}" y="{height-24}" text-anchor="middle" class="label">processed training tokens</text>',
        f'<text x="39" y="{top + plot_h/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 39 {top + plot_h/2:.2f})">cross-entropy loss, nats/token</text>',
    ])
    write_svg(
        OUT / "training-validation-loss.svg",
        "GELU, SwiGLU, and ReLU training and validation loss curves",
        "One shared-axis chart shows the recorded fixed training-prefix and held-out validation cross-entropy losses of GELU, SwiGLU, and ReLU build-nanoGPT runs as processed training tokens increase. Each activation has a colored solid validation curve and a matching dashed training curve. A vertical dashed line at 176.2 million tokens marks the shared comparison point. Open circles show interpolated GELU and ReLU validation values and a filled circle shows the measured SwiGLU value.",
        "\n".join(body), width, height,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    render_gelu_curve()
    render_training_validation()


if __name__ == "__main__":
    main()
