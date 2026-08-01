#!/usr/bin/env python3
"""Generate figures for the temperature/softmax blog post."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "temp-softmax")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- figure 1
# Two-token softmax as a function of the logit gap, for several temperatures.
gap = np.linspace(-6, 6, 800)
temps = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0]
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(temps)))

fig, ax = plt.subplots(figsize=(8, 5))
for T, c in zip(temps, colors):
    p = 1.0 / (1.0 + np.exp(-gap / T))
    ax.plot(gap, p, color=c, lw=2.2, label=f"T = {T}")

ax.axhline(0.5, color="#999999", lw=0.9, ls=":")
ax.axvline(0.0, color="#999999", lw=0.9, ls=":")
ax.set_xlabel("logit gap  $z_1 - z_2$")
ax.set_ylabel("softmax probability of token 1")
ax.set_title("Temperature reshapes the same logit gap")
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(-6, 6)
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "temperature-curves.svg"))
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Same five logits, five temperatures, as probability bars.
logits = np.array([3.0, 2.4, 1.8, 0.6, -0.4])
labels = ["cat", "dog", "bird", "rock", "the"]
temps2 = [0.1, 0.5, 1.0, 2.0, 10.0]

fig, axes = plt.subplots(1, len(temps2), figsize=(11, 3.6), sharey=True)
for ax, T in zip(axes, temps2):
    e = np.exp((logits - logits.max()) / T)
    p = e / e.sum()
    ax.bar(labels, p, color="#4a7fb5", edgecolor="none")
    ax.set_title(f"T = {T}", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
axes[0].set_ylabel("probability", fontsize=12)
fig.suptitle("Identical logits [3.0, 2.4, 1.8, 0.6, -0.4], five temperatures", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(os.path.join(OUT, "temperature-bars.svg"))
plt.close(fig)

print("wrote", os.path.abspath(OUT))
