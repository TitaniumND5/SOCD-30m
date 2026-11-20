# -*- coding: utf-8 -*-
"""
全国SOCD反演组图右侧统计子图绘制代码：两个图分开出图
------------------------------------------------
内容：
1. CLCD分类策略下各组SOCD Mean / Std双柱图
2. 地理分区策略下各组SOCD Mean / Std双柱图
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

# =========================
# 1. 文件路径
# =========================
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
clcd_csv = PACKAGE_ROOT / "results" / "new_500m_v2_stats_by_CLCD.csv"
region_csv = PACKAGE_ROOT / "results" / "new_500m_v2_stats_by_region.csv"
out_dir = PACKAGE_ROOT / "outputs" / "fig9_right_statistics"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 基本参数设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MEAN_COLOR = "#7DAA5F"   # RGB(125, 170, 95)
MEAN_EDGE  = "#185F37"   # RGB(24, 95, 55)

STD_COLOR  = "#E2DC82"   # RGB(226, 220, 130)
STD_EDGE   = "#9A944A"   # 比黄绿略深的边框

GRID_COLOR = "#D9D9D9"
SPINE_COLOR = "#444444"
TEXT_COLOR = "#222222"

DPI = 600

# 单独子图尺寸：适配右侧上下两个矩形区域
FIG_W = 4.2
FIG_H = 4.8

BAR_W = 0.30
XTICK_SIZE = 9
YTICK_SIZE = 10.5
YLABEL_SIZE = 11
LEGEND_SIZE = 12.5
# 柱顶数值以可读性为优先：显著放大，并将两组标签错开。
VALUE_SIZE = 13.5

DECIMAL = 2

# =========================
# 3. 读取数据
# =========================
df_clcd = pd.read_csv(clcd_csv, encoding="utf-8-sig")
df_region = pd.read_csv(region_csv, encoding="utf-8-sig")

df_clcd.columns = [c.strip() for c in df_clcd.columns]
df_region.columns = [c.strip() for c in df_region.columns]

clcd_order = ["Cropland", "Forest", "Shrubland", "Grassland", "Barren", "Other"]
region_order = ["Northeast", "North China", "East China", "South China", "Northwest", "Southwest"]

df_clcd["CLCD_class"] = pd.Categorical(df_clcd["CLCD_class"], categories=clcd_order, ordered=True)
df_clcd = df_clcd.sort_values("CLCD_class").reset_index(drop=True)

df_region["Region"] = pd.Categorical(df_region["Region"], categories=region_order, ordered=True)
df_region = df_region.sort_values("Region").reset_index(drop=True)

# =========================
# 4. 绘图辅助函数
# =========================
def add_value_labels(ax, x_positions, values, x_shift=0.0, offset_ratio=0.010):
    y_max = ax.get_ylim()[1]
    offset = y_max * offset_ratio

    for x, v in zip(x_positions, values):
        ax.text(
            x + x_shift, v + offset,
            f"{v:.{DECIMAL}f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            color=TEXT_COLOR
        )


def style_axis(ax, ymax):
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", linestyle="--", linewidth=0.75, color=GRID_COLOR, alpha=0.75)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(SPINE_COLOR)

    ax.tick_params(axis="x", labelsize=XTICK_SIZE, width=0.9, length=3.5, color=SPINE_COLOR)
    ax.tick_params(axis="y", labelsize=YTICK_SIZE, width=0.9, length=3.5, color=SPINE_COLOR)

    ax.set_ylabel("SOCD (kg m$^{-2}$)", fontsize=YLABEL_SIZE, color=TEXT_COLOR)


def plot_grouped_bars(ax, labels, means, stds):
    x = np.arange(len(labels))
    x_mean = x - BAR_W / 2
    x_std  = x + BAR_W / 2

    ax.bar(
        x_mean, means, width=BAR_W,
        color=MEAN_COLOR, edgecolor=MEAN_EDGE, linewidth=0.7,
        label="Mean", zorder=3
    )

    ax.bar(
        x_std, stds, width=BAR_W,
        color=STD_COLOR, edgecolor=STD_EDGE, linewidth=0.7,
        label="Std", zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # 为大字号柱顶标签预留更充足的顶部空间。
    ymax = max(np.max(means), np.max(stds)) * 1.35
    style_axis(ax, ymax)

    # 数值整体右移；标准差标签额外右偏，避免与均值柱及其标签重叠。
    add_value_labels(ax, x_mean, means, x_shift=0.055)
    add_value_labels(ax, x_std, stds, x_shift=0.115)


def add_legend(fig):
    legend_handles = [
        Patch(facecolor=MEAN_COLOR, edgecolor=MEAN_EDGE, label="Mean"),
        Patch(facecolor=STD_COLOR, edgecolor=STD_EDGE, label="Std")
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.0,
        handleheight=1.0,
        columnspacing=2.2,
        handletextpad=0.6
    )


def save_single_figure(fig, base_name):
    png_path = os.path.join(out_dir, base_name + ".png")
    svg_path = os.path.join(out_dir, base_name + ".svg")
    pdf_path = os.path.join(out_dir, base_name + ".pdf")

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


# =========================
# 5. 准备数据
# =========================
clcd_labels = df_clcd["CLCD_class"].tolist()
clcd_means = df_clcd["Mean"].values
clcd_stds  = df_clcd["Std"].values

region_labels_display = [
    "Northeast",
    "North\nChina",
    "East\nChina",
    "South\nChina",
    "Northwest",
    "Southwest"
]
region_means = df_region["Mean"].values
region_stds  = df_region["Std"].values

# =========================
# 6. 单独绘制 CLCD 图
# =========================
fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
plot_grouped_bars(ax1, clcd_labels, clcd_means, clcd_stds)
add_legend(fig1)

plt.subplots_adjust(
    left=0.12,
    right=0.97,
    top=0.86,
    bottom=0.16
)

save_single_figure(fig1, "Fig9c_500m_landcover_statistics")
plt.close(fig1)

# =========================
# 7. 单独绘制 Region 图
# =========================
fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
plot_grouped_bars(ax2, region_labels_display, region_means, region_stds)
add_legend(fig2)

plt.subplots_adjust(
    left=0.12,
    right=0.97,
    top=0.86,
    bottom=0.20
)

save_single_figure(fig2, "Fig9d_500m_region_statistics")
plt.close(fig2)

print("两个统计子图已分别绘制完成。")
