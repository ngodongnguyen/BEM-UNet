import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Circle

models = [
    "TransUNet","UCTransNet","MT-UNet", "Swin-UNet",
    "VM-UNet","MSSMamba", "Swin-UMamba", "DMM-UNet","MS-DCSNet", "Ours"
]

flops = [24.66,32.94,44.73,10.82, 3.15,25.47, 31.35, 3.42,31.33, 21.37]
dice  = [77.48,78.23,78.59, 79.13, 81.08,81.22, 82.26, 83.07,84.22, 84.94]

sizes  = [f * 20 for f in flops]
colors = cm.tab20(np.linspace(0, 1, len(models)))

fig, ax = plt.subplots(figsize=(10, 6))

circles = []
for x, y, s, c in zip(flops, dice, sizes, colors):
    r = (s ** 0.5) / 15
    circle = Circle((x, y), r, facecolor=c, edgecolor="black", alpha=0.8, zorder=3)
    ax.add_patch(circle)
    circles.append(circle)

for circle, x, y, name, c in zip(circles, flops, dice, models, colors):
    r = circle.get_radius()

    if name =="MSSMamba":
        ax.annotate(
            name,
            xy=(x, y + r),
            xytext=(x, y + r + 1.8),
            ha="center",
            va="bottom",
            fontsize=9,
            arrowprops=dict(
                arrowstyle="<-",
                lw=1.2,
                color=c
            ),
            zorder=4
        )
    elif name == "DMM-UNet" or name=="MS-DCSNet":
        ax.annotate(
            name,
            xy=(x, y + r),
            xytext=(x, y + r + 1.8),
            ha="center",
            va="bottom",
            fontsize=9,
            arrowprops=dict(
                arrowstyle="<-",
                lw=1.2,
                color=c
            ),
            zorder=4
        )
    elif name == "Swin-UMamba":
        ax.annotate(
            name,
            xy=(x + r, y),
            xytext=(x + r + 1.8, y),
            ha="left",
            va="center",
            fontsize=11,
            arrowprops=dict(arrowstyle="<-", lw=1.2, color=c),
            zorder=4
        )

    else:
        ax.annotate(
            name,
            xy=(x, y - r),
            xytext=(x, y - r - 1.8),
            ha="center",
            va="top",
            fontsize=11,
            arrowprops=dict(
                arrowstyle="<-",
                lw=1.2,
                color=c
            ),
            zorder=4
        )


ax.set_xlim(0, 60)
ax.set_ylim(60, 90)
ax.set_xlabel("FLOPs (G)")
ax.set_ylabel("Dice (%)")
ax.set_aspect("equal", adjustable="box")

ax.grid(True, linestyle="--", alpha=0.5)

legend_text = "\n".join(
    f"{m} (Dice: {d:.2f}, FLOPs: {f:.2f}G)"
    for m, d, f in zip(models, dice, flops)
)

ax.text(
    0.01, 0.33,
    legend_text,
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.tight_layout()
plt.savefig("model_comparison_bubble_chart.png", dpi=300,bbox_inches='tight')
plt.show()
