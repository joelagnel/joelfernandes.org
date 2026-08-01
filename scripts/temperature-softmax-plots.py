#!/usr/bin/env python3
"""Generate the figure for the temperature/softmax blog post."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "temp-softmax")
os.makedirs(OUT, exist_ok=True)

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
