import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

font_cn = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
font_cn_small = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=9)
font_en = FontProperties(family="Times New Roman", size=10)

models = [
    "baseline",
    "noise_lora",
    "shuffle_lora",
    "SG-SPD",
]

display_models = [
    "baseline",
    "noise\nlora",
    "shuffle\nlora",
    "SG-SPD",
]

metrics = [
    "预测唯一率",
    "词元多样性",
    "最高重复率",
]

raw = np.array([
    [0.763, 0.065, 0.022],
    [0.892, 0.198, 0.117],
    [0.973, 0.206, 0.027],
    [0.883, 0.182, 0.054],
], dtype=float)

# 颜色方向：唯一率、TTR 越高越好；最高重复次数越低越好
score = raw.copy()
score[:, 2] = -score[:, 2]

mins = score.min(axis=0)
maxs = score.max(axis=0)
norm = (score - mins) / np.where(maxs - mins == 0, 1, maxs - mins)

# 圆点大小也用同一“优劣方向”，避免最高重复次数越大圆越大的误解
sizes = 320 + norm * 1250

cmap = LinearSegmentedColormap.from_list(
    "paper_teal_red",
    ["#B55A5A", "#F0E6D2", "#2F7F7B"]
)

fig, ax = plt.subplots(figsize=(7.2, 3.9))

for i in range(len(models)):
    for j in range(len(metrics)):
        ax.scatter(
            i, j,
            s=sizes[i, j],
            c=[norm[i, j]],
            cmap=cmap,
            vmin=0,
            vmax=1,
            edgecolors="#2B2B2B",
            linewidths=0.7,
            zorder=3,
        )

        if j < 2:
            label = f"{raw[i, j]:.3f}"
        else:
            label = f"{raw[i, j]:.3f}"

        ax.text(
            i, j,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontproperties=font_en,
            color="#111111",
            zorder=4,
        )

ax.set_xticks(np.arange(len(models)))
ax.set_xticklabels(display_models, fontproperties=font_en, fontsize=9)

ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics, fontproperties=font_cn, fontsize=10)

ax.set_xlim(-0.55, len(models) - 0.45)
ax.set_ylim(-0.55, len(metrics) - 0.45)
ax.invert_yaxis()

ax.set_axisbelow(True)
ax.grid(axis="both", linestyle="--", linewidth=0.6, color="#D6D6D6")

for spine in ax.spines.values():
    spine.set_linewidth(0.9)
    spine.set_color("#333333")

cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)),
    ax=ax,
    fraction=0.035,
    pad=0.025,
)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["较弱", "较优"])
for tick in cbar.ax.get_yticklabels():
    tick.set_fontproperties(font_cn_small)


plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("ch5_output_diversity_matrix.png", dpi=1200, bbox_inches="tight", pad_inches=0.15)
plt.savefig("ch5_output_diversity_matrix.pdf", bbox_inches="tight", pad_inches=0.15)
plt.show()