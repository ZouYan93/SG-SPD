import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ===== 字体 =====
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

font_cn = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=11)
font_en = FontProperties(family="Times New Roman", size=11)

models = ["baseline", "lora", "noise_lora", "shuffle_lora", "SG-SPD"]

metrics = [
    "BLEU-1",
    "BLEU-4",
    "SB",
    "R1-F",
    r"$\Delta$SB",
    r"$\Delta$R1-F",
    r"$\Delta$WER",
]

raw_data = {
    "baseline":     [15.99, 0.68, 0.524, 10.55, 0.058, 0.44, -2.72],
    "lora":         [18.65, 1.82, 0.728, 10.59, 0.039, 0.62, 10.57],
    "noise_lora":   [19.42, 1.64, 0.603, 11.44, 0.486, 9.00, 15.08],
    "shuffle_lora": [18.78, 1.52, 0.647, 10.74, 0.028, -0.20, 17.64],
    "SG-SPD":       [19.24, 1.53, 1.217, 11.49, 1.088, 8.35, 8.72],
}

# ===== 带最小显示半径的 min-max 归一化 =====
data = np.array([raw_data[m] for m in models], dtype=float)
mins = data.min(axis=0)
maxs = data.max(axis=0)
denom = np.where(maxs - mins == 0, 1, maxs - mins)

floor = 0.15
norm_data = floor + (1 - floor) * (data - mins) / denom

num_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
angles = np.concatenate([angles, angles[:1]])

styles = {
    "baseline":     {"color": "#3B4252", "linestyle": "-",  "marker": "o", "lw": 2.3},
    "lora":         {"color": "#5E81AC", "linestyle": "--", "marker": "s", "lw": 1.9},
    "noise_lora":   {"color": "#8FBC8F", "linestyle": "-.", "marker": "^", "lw": 1.9},
    "shuffle_lora": {"color": "#D08770", "linestyle": ":",  "marker": "D", "lw": 2.1},
    "SG-SPD":       {"color": "#BF616A", "linestyle": "-",  "marker": "*", "lw": 2.9},
}

fig = plt.figure(figsize=(9.2, 6.4))
ax = fig.add_subplot(111, polar=True)

for idx, model in enumerate(models):
    values = np.concatenate([norm_data[idx], norm_data[idx, :1]])
    st = styles[model]

    marker_size = 7.0 if model == "baseline" else (9.5 if model == "SG-SPD" else 6.0)

    ax.plot(
        angles,
        values,
        label=model,
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=st["lw"],
        marker=st["marker"],
        markersize=marker_size,
        markerfacecolor="white" if model != "SG-SPD" else st["color"],
        markeredgecolor=st["color"],
        markeredgewidth=1.4,
        zorder=6 if model in ["baseline", "SG-SPD"] else 4,
    )

    if model == "SG-SPD":
        ax.fill(angles, values, color=st["color"], alpha=0.10, zorder=1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=9, fontproperties=font_en)

ax.set_ylim(0, 1.10)
ax.set_yticks([floor, 0.35, 0.55, 0.75, 1.00])
ax.set_yticklabels(["min", "0.35", "0.55", "0.75", "max"], fontsize=7.5, fontproperties=font_en)
ax.set_rlabel_position(18)

ax.tick_params(axis="x", pad=8)
ax.tick_params(axis="y", pad=4)

ax.grid(True, linestyle="--", linewidth=0.75, color="#B0B0B0", alpha=0.8)
ax.spines["polar"].set_color("#333333")
ax.spines["polar"].set_linewidth(1.1)

legend = ax.legend(
    loc="center left",
    bbox_to_anchor=(1.13, 0.5),
    frameon=True,
    edgecolor="#CCCCCC",
    facecolor="white",
    title="实验设置",
    title_fontproperties=font_cn,
    prop=font_en,
    handlelength=2.6,
    borderpad=0.8,
    labelspacing=0.65,
)

plt.subplots_adjust(left=0.06, right=0.78, top=0.90, bottom=0.08)

plt.savefig("ch5_radar_main_results_clean.png", dpi=1200, bbox_inches="tight", pad_inches=0.25)
plt.savefig("ch5_radar_main_results_clean.pdf", bbox_inches="tight", pad_inches=0.25)
plt.show()