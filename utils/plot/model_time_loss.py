import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib as mpl

# ======================
# 字体设置
# ======================

font_cn = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)   # 宋体
font_en = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=14)    # Times New Roman

mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.family"] = font_en.get_name()

# ======================
# 数据
# x: 总训练时间/min
# y: 最优验证损失
# ======================

data = [
    ("BART",    "Transformer", 112.3, 7.2361),
    ("BART",    "MLP",         101.8, 7.2460),
    ("BART",    "CNN",         101.8, 3.9286),
    ("BART",    "LSTM",        123.3, 3.3008),

    ("Pegasus", "Transformer", 152.2, 3.8437),
    ("Pegasus", "MLP",         139.9, 3.7652),
    ("Pegasus", "CNN",         141.4, 3.7133),
    ("Pegasus", "LSTM",        163.8, 3.6934),

    ("T5",      "Transformer", 209.7, 3.3182),
    ("T5",      "MLP",         198.9, 3.2968),
    ("T5",      "CNN",         201.3, 3.2871),
    ("T5",      "LSTM",        222.6, 3.3161),
]

colors = {
    "BART": "#b79de0",
    "Pegasus": "#e4984e",
    "T5": "#00c0b2",
}

markers = {
    "Transformer": "o",
    "MLP": "s",
    "CNN": "^",
    "LSTM": "D",
}

sizes = {
    "Transformer": 90,
    "MLP": 55,
    "CNN": 260,
    "LSTM": 105,
}

# ======================
# 绘图
# ======================

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

for base, model, x, y in data:
    ax.scatter(
        x, y,
        s=sizes[model],
        marker=markers[model],
        color=colors[base],
        edgecolor="none",
        zorder=3
    )

# 坐标范围和刻度
ax.set_xlim(100, 232)
ax.set_ylim(3.0, 7.7)
ax.set_xticks([100, 125, 150, 175, 200, 225])
ax.set_yticks([3.0, 4.0, 5.0, 6.0, 7.0])

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_en)
    label.set_fontsize(12)

# 网格
ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
ax.grid(axis="x", visible=False)

# 隐藏默认边框
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="both", direction="out", length=3, width=0.8, colors="black")

# 画带箭头的坐标轴
ax.annotate(
    "",
    xy=(232, 3.0),
    xytext=(100, 3.0),
    arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
    annotation_clip=False
)

ax.annotate(
    "",
    xy=(100, 7.68),
    xytext=(100, 3.0),
    arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
    annotation_clip=False
)

# 坐标轴文字
ax.text(
    100.5, 7.73,
    "最优验证损失",
    fontproperties=font_cn,
    ha="left",
    va="bottom",
    fontsize=13,
    clip_on=False
)

ax.text(
    1.045, -0.015,
    "总训练时间",
    transform=ax.transAxes,
    fontproperties=font_cn,
    ha="left",
    va="center",
    fontsize=13,
    clip_on=False
)

ax.text(
    1.18, -0.015,
    "/min",
    transform=ax.transAxes,
    fontproperties=font_en,
    ha="left",
    va="center",
    fontsize=13,
    clip_on=False
)

# ======================
# 图例
# ======================

color_handles = [
    Line2D([0], [0], color=colors["BART"], lw=4, label="BART"),
    Line2D([0], [0], color=colors["Pegasus"], lw=4, label="Pegasus"),
    Line2D([0], [0], color=colors["T5"], lw=4, label="T5"),
]

marker_handles = [
    Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
           markersize=6, linestyle="None", label="Transformer"),
    Line2D([0], [0], marker="s", color="black", markerfacecolor="white",
           markersize=6, linestyle="None", label="MLP"),
    Line2D([0], [0], marker="^", color="black", markerfacecolor="white",
           markersize=7, linestyle="None", label="CNN"),
    Line2D([0], [0], marker="D", color="black", markerfacecolor="white",
           markersize=6, linestyle="None", label="LSTM"),
]

leg1 = ax.legend(
    handles=color_handles,
    loc="upper right",
    bbox_to_anchor=(0.73, 0.90),
    frameon=False,
    handlelength=1.1,
    handletextpad=0.4,
    labelspacing=0.35,
    prop=font_en
)

ax.add_artist(leg1)

ax.legend(
    handles=marker_handles,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.90),
    frameon=False,
    handletextpad=0.5,
    labelspacing=0.35,
    prop=font_en
)

# 手动调整边距，避免右下角文字被挤压
plt.subplots_adjust(left=0.09, right=0.82, bottom=0.15, top=0.90)

# 保存图片
plt.savefig("model_time_loss.png", dpi=300)
plt.savefig("model_time_loss.pdf", bbox_inches="tight", pad_inches=0.25)
plt.show()
