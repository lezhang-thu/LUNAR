import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

labels = [
    "LUNAR (parallel)",
    "LUNAR (serial)",
    "LUNAR (serial/DeepSeek-V3.2)",
    "SCULP (serial/DeepSeek-V3.2)",
    "SCULP (parallel/DeepSeek-V3.2)",
    "SCULP (serial/GPT-5.2)",
]
values = [540.1, 1263.2, 1350.9, 1033.9, 294.0, 822.3]

# Style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
})

fig, ax = plt.subplots(figsize=(11, 3.8))

y = np.arange(len(labels))

colors = ["#b9d7a8", "#d9a7a7", "#c9c7ea", "#b8d9ea", "#f7df8b", "#f1b0b0"]
bars = ax.barh(y, values, color=colors, edgecolor="none")

# Hide method names on y-axis
ax.set_yticks(y)
ax.set_yticklabels([""] * len(labels))
ax.tick_params(axis="y", length=0)
ax.invert_yaxis()

ax.set_xlabel("Average time (seconds)")

# x-axis
ax.set_xlim(0, 2000)
ax.set_xticks([0, 500, 1000, 1500, 2000])

# Value labels
mx = max(values)
for rect, v in zip(bars, values):
    ax.text(v + mx * 0.01,
            rect.get_y() + rect.get_height() / 2,
            f"{v:.1f}",
            va="center",
            ha="left",
            fontsize=16)

# Legend
legend_handles = [Patch(facecolor=c, edgecolor="none", label=lab)
                  for c, lab in zip(colors, labels)]
ax.legend(handles=legend_handles,
          loc="upper right",
          fontsize=11,
          bbox_to_anchor=(1, 1),
          frameon=True,
          borderaxespad=0.3,
          labelspacing=0.6,
          handlelength=1.0,
          handleheight=0.8)

# Closed axes box
for side in ["left", "bottom", "top", "right"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.0)

# Optional grid
ax.xaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
ax.set_axisbelow(True)

plt.tight_layout()

# Save: PDF is vector (required)
plt.savefig("figure.pdf", bbox_inches="tight")
plt.savefig("figure.png", dpi=600, bbox_inches="tight")
plt.show()
