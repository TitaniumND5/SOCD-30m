# -*- coding: utf-8 -*-
"""
三产品 SOCD 一致性对比分析与组图绘制
------------------------------------------------
三产品字段来自最新 GEE 导出表格：

1. YOUR_SOCD_0_30cm_kg_m2
   -> This study (RF)
   -> 本研究 RF 生成的 0–30 cm SOCD 结果

2. SOILGRIDS_SOCD_0_30cm_kg_m2
   -> SoilGrids v2.0
   -> SoilGrids v2.0 0–30 cm SOCD product

3. HENGL2026_SOCD_0_30cm_kg_m2
   -> OpenLandMap-soildb
   -> OpenLandMap-soildb 0–30 cm SOCD converted from SOCd product,
      Hengl et al. (2026)

功能：
1. 读取 GEE 导出的三产品采样 CSV
2. 清洗无效值
3. 计算三产品基础统计
4. 计算 This study (RF) 与 SoilGrids v2.0、OpenLandMap-soildb 的一致性指标
5. 绘制论文用组图：
   (a) This study (RF) vs 两个外部产品散点图
   (b) 三产品 SOCD 分布图
6. 导出统计表、清洗后数据、绘图样本和图件

注意：
- SoilGrids v2.0 与 OpenLandMap-soildb 均不是实测真值。
- 本图应表述为产品间一致性比较，而不是严格精度验证。
"""

import os
from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================================================
# 1. 路径设置
# =========================================================

# 修改为你最新 GEE 导出的 CSV 路径
# 对应 GEE 导出文件名：
# SOCD_3products_grid_sampling_corrected_SoilGrids_Hengl2026.csv
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# The source sampling CSV is intentionally excluded; configure its local path here.
csv_path = r"PATH_TO_THREE_PRODUCT_SAMPLING_CSV.csv"
out_dir = str(PACKAGE_ROOT / "outputs" / "three_product_consistency")
os.makedirs(out_dir, exist_ok=True)

out_clean_csv = os.path.join(out_dir, "SOCD_3products_valid_samples_corrected.csv")
out_stats_csv = os.path.join(out_dir, "SOCD_3products_basic_statistics_corrected.csv")
out_metrics_csv = os.path.join(out_dir, "SOCD_3products_pairwise_agreement_metrics_corrected.csv")
out_scatter_sample_csv = os.path.join(out_dir, "SOCD_3products_scatter_samples_used_corrected.csv")

out_fig_png = os.path.join(out_dir, "SOCD_3products_comparison_panel_final_compact_v2.png")
out_fig_svg = os.path.join(out_dir, "SOCD_3products_comparison_panel_final_compact_v2.svg")
out_fig_tif = os.path.join(out_dir, "SOCD_3products_comparison_panel_final_compact_v2.tif")
out_fig_pdf = os.path.join(out_dir, "SOCD_3products_comparison_panel_final_compact_v2.pdf")


# =========================================================
# 2. 字段与产品名称设置
# =========================================================

COL_LON = "lon"
COL_LAT = "lat"
COL_GRID = "grid_id"
COL_CLCD = "CLCD"

COL_THIS = "YOUR_SOCD_0_30cm_kg_m2"
COL_SOILGRIDS = "SOILGRIDS_SOCD_0_30cm_kg_m2"
COL_OPENLANDMAP = "HENGL2026_SOCD_0_30cm_kg_m2"

# 可选检查列：正式分析不用，但如果 CSV 中有，会被保留在清洗结果里
COL_OPENLANDMAP_DENSITY = "HENGL2026_SOC_density_kg_m3"

LABEL_THIS = "This study (RF)"
LABEL_SOILGRIDS = "SoilGrids v2.0"
LABEL_OPENLANDMAP = "OpenLandMap-soildb"

FULL_LABEL_THIS = "This study (RF)"
FULL_LABEL_SOILGRIDS = "SoilGrids v2.0"
FULL_LABEL_OPENLANDMAP = "OpenLandMap-soildb (Hengl et al., 2026)"


# =========================================================
# 3. 参数设置
# =========================================================

# 当前 GEE 中已对 Hengl/OpenLandMap 的 32767 NoData 做过掩膜，
# 且 sampleRegions 后已经 notNull 过滤。
# 因此默认不再把 0 视为 NoData。
# 如果后续检查发现 OpenLandMap-soildb 中存在大量异常 0 值，可改为 True。
OPENLANDMAP_ZERO_AS_NODATA = False

# SOCD 合理范围。只用于剔除极端异常值。
# 三产品均为 kg m-2。
VALID_MIN = 0.0
VALID_MAX = 150.0

# 散点图展示点数。统计指标仍使用全部有效样本。
SCATTER_N = 4500

# OpenLandMap-soildb 转换后仍具有离散阶梯特征。
# 这里只对绘图 y 值加入很小抖动，统计指标仍使用原始值。
OPENLANDMAP_JITTER = 0.045
JITTER_RANDOM_SEED = 42

# 散点视觉参数
SCATTER_SIZE = 17
SOILGRIDS_ALPHA = 0.26
OPENLANDMAP_ALPHA = 0.34

# 分层抽样分位数箱数，用于保证散点图覆盖全值域
SCATTER_QUANTILE_BINS = 25

# 绘图值域上限。
# 建议 30.0：既保留主要高值，又不会把主体点云压得过小。
PLOT_MAX_MANUAL = 20.0
PLOT_MAX_CAP = 20.0

# 分布图 bins 数量
HIST_BINS = 60

# 图像参数
DPI = 600
FIG_W = 12.1
FIG_H = 5.55

# 论文图字号参数：面向 Word 插图可读性
TITLE_FS = 16.4
AXIS_LABEL_FS = 15.1
TICK_FS = 13.0
LEGEND_FS = 14
METRIC_FS = 10.6
METRIC_HEADER_FS = 11.6

# 字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 4. 配色方案
# =========================================================

COLOR_THIS = "#4F8F5B"         # This study: green
COLOR_SOILGRIDS = "#4C78A8"    # SoilGrids: blue
COLOR_OPENLANDMAP = "#E6862F"  # OpenLandMap-soildb: orange

EDGE_GREY = "#333333"
GRID_GREY = "#DADADA"
TEXT_GREY = "#222222"
LIGHT_EDGE = "#B8B8B8"


# =========================================================
# 5. 工具函数
# =========================================================

def read_csv_safely(path: str) -> pd.DataFrame:
    """兼容 utf-8-sig / utf-8 / gbk 编码读取 CSV。"""
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗三产品数据。"""
    required_cols = [
        COL_LON,
        COL_LAT,
        COL_THIS,
        COL_SOILGRIDS,
        COL_OPENLANDMAP
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV 缺少必要字段：{}\n"
            "请确认你读取的是最新 GEE 导出的三产品采样表。".format(missing)
        )

    numeric_cols = [
        COL_LON,
        COL_LAT,
        COL_THIS,
        COL_SOILGRIDS,
        COL_OPENLANDMAP
    ]

    optional_numeric_cols = [
        COL_CLCD,
        COL_GRID,
        COL_OPENLANDMAP_DENSITY
    ]

    for c in optional_numeric_cols:
        if c in df.columns:
            numeric_cols.append(c)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    valid = (
        df[COL_LON].notna() &
        df[COL_LAT].notna() &
        df[COL_THIS].notna() &
        df[COL_SOILGRIDS].notna() &
        df[COL_OPENLANDMAP].notna()
    )

    for c in [COL_THIS, COL_SOILGRIDS, COL_OPENLANDMAP]:
        valid &= df[c].between(VALID_MIN, VALID_MAX)

    if OPENLANDMAP_ZERO_AS_NODATA:
        valid &= df[COL_OPENLANDMAP] > 0

    df_clean = df.loc[valid].copy()

    if df_clean.empty:
        raise ValueError(
            "清洗后没有有效样本。请检查字段名、单位、VALID_MIN / VALID_MAX "
            "或 OPENLANDMAP_ZERO_AS_NODATA 设置。"
        )

    return df_clean


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """计算三产品基础统计。"""
    products = {
        FULL_LABEL_THIS: COL_THIS,
        FULL_LABEL_SOILGRIDS: COL_SOILGRIDS,
        FULL_LABEL_OPENLANDMAP: COL_OPENLANDMAP
    }

    records = []
    for name, col in products.items():
        x = df[col].dropna().values.astype(float)
        records.append({
            "Product": name,
            "Column": col,
            "N": len(x),
            "Mean": np.mean(x),
            "Std": np.std(x, ddof=1),
            "Median": np.median(x),
            "Min": np.min(x),
            "P05": np.percentile(x, 5),
            "P25": np.percentile(x, 25),
            "P75": np.percentile(x, 75),
            "P95": np.percentile(x, 95),
            "Max": np.max(x)
        })

    return pd.DataFrame(records)


def pairwise_metrics(df: pd.DataFrame, ref_col: str, comp_col: str, comp_label: str) -> dict:
    """
    计算 This study (RF) 与外部产品之间的一致性指标。
    x = This study (RF)
    y = external product
    bias = external product - This study (RF)

    注意：
    R2_corr 是 Pearson r 的平方，用于产品间一致性描述；
    不是以外部产品为真值的预测精度 R²。
    """
    x = df[ref_col].values.astype(float)
    y = df[comp_col].values.astype(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError(f"{comp_label} 有效样本过少，无法计算指标。")

    diff = y - x

    r = np.corrcoef(x, y)[0, 1]
    r2_corr = r ** 2

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    median_bias = np.median(diff)

    # y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    return {
        "Comparison": f"{comp_label} vs {LABEL_THIS}",
        "ReferenceProduct": comp_label,
        "N": len(x),
        "Pearson_r": r,
        "R2_corr": r2_corr,
        "RMSE": rmse,
        "MAE": mae,
        "Bias_Product_minus_ThisStudy": bias,
        "Median_Bias_Product_minus_ThisStudy": median_bias,
        "Slope": slope,
        "Intercept": intercept,
        "ThisStudy_Mean": np.mean(x),
        f"{comp_label}_Mean": np.mean(y)
    }


def stratified_sample_by_quantile(
    df: pd.DataFrame,
    value_col: str,
    n: int,
    bins: int = 25,
    random_state: int = 42
) -> pd.DataFrame:
    """
    按 This study (RF) 值域分位数分层抽样，用于散点图展示。
    统计指标仍使用全部有效样本。
    """
    if len(df) <= n:
        return df.copy()

    temp = df.copy()

    temp["_qbin"] = pd.qcut(
        temp[value_col],
        q=bins,
        labels=False,
        duplicates="drop"
    )

    valid_bins = temp["_qbin"].dropna().unique()
    n_bins = len(valid_bins)

    if n_bins == 0:
        return temp.sample(
            n=n,
            random_state=random_state
        ).drop(columns=["_qbin"], errors="ignore")

    per_bin = max(1, int(np.ceil(n / n_bins)))

    sampled_parts = []
    for _, sub in temp.groupby("_qbin"):
        take = min(len(sub), per_bin)
        sampled_parts.append(sub.sample(n=take, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)

    return sampled.drop(columns=["_qbin"], errors="ignore")


def determine_plot_max(df: pd.DataFrame) -> float:
    """确定绘图 x/y 轴上限。"""
    if PLOT_MAX_MANUAL is not None:
        return float(PLOT_MAX_MANUAL)

    all_values = pd.concat([
        df[COL_THIS],
        df[COL_SOILGRIDS],
        df[COL_OPENLANDMAP]
    ]).dropna().values.astype(float)

    q995 = np.percentile(all_values, 99.5)
    xmax = math.ceil(q995)
    xmax = min(max(xmax, 10.0), PLOT_MAX_CAP)

    return float(xmax)


def add_kde_or_hist(ax, data, color, label, x_grid, bins, x_range):
    """
    优先画 KDE；如果 scipy 不可用或 KDE 失败，则退化为密度直方图轮廓。
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[(data >= x_range[0]) & (data <= x_range[1])]

    if len(data) < 5:
        return

    if SCIPY_AVAILABLE:
        try:
            kde = gaussian_kde(data)
            y = kde(x_grid)
            ax.plot(x_grid, y, color=color, lw=2.35, label=label)
            ax.fill_between(x_grid, y, color=color, alpha=0.16)
            return
        except Exception:
            pass

    ax.hist(
        data,
        bins=bins,
        range=x_range,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=color,
        label=label
    )


def style_axes(ax):
    """统一图件风格。"""
    ax.grid(True, linestyle="--", linewidth=0.7, color=GRID_GREY, alpha=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(EDGE_GREY)

    ax.tick_params(
        axis="both",
        labelsize=TICK_FS,
        width=1.0,
        length=4,
        color=EDGE_GREY
    )


def get_metric_row(metrics_df: pd.DataFrame, product_label: str) -> pd.Series:
    """根据产品名提取指标行，避免依赖 Comparison 字符串。"""
    rows = metrics_df[metrics_df["ReferenceProduct"] == product_label]
    if rows.empty:
        raise ValueError(f"未找到 {product_label} 的指标结果。")
    return rows.iloc[0]


# =========================================================
# 6. 主流程
# =========================================================

def main():
    warnings.filterwarnings("ignore")

    print("Reading CSV...")
    df_raw = read_csv_safely(csv_path)
    print(f"Raw samples: {len(df_raw)}")
    print("CSV columns:")
    print(list(df_raw.columns))

    print("\nCleaning data...")
    df = clean_data(df_raw)
    print(f"Valid samples after cleaning: {len(df)}")

    df.to_csv(out_clean_csv, index=False, encoding="utf-8-sig")

    stats_df = basic_statistics(df)
    stats_df.to_csv(out_stats_csv, index=False, encoding="utf-8-sig")

    metrics = [
        pairwise_metrics(
            df,
            ref_col=COL_THIS,
            comp_col=COL_SOILGRIDS,
            comp_label=LABEL_SOILGRIDS
        ),
        pairwise_metrics(
            df,
            ref_col=COL_THIS,
            comp_col=COL_OPENLANDMAP,
            comp_label=LABEL_OPENLANDMAP
        )
    ]

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_metrics_csv, index=False, encoding="utf-8-sig")

    print("\nBasic statistics:")
    print(stats_df)

    print("\nPairwise agreement metrics:")
    print(metrics_df)

    df_scatter = stratified_sample_by_quantile(
        df,
        value_col=COL_THIS,
        n=SCATTER_N,
        bins=SCATTER_QUANTILE_BINS,
        random_state=42
    )

    df_scatter.to_csv(out_scatter_sample_csv, index=False, encoding="utf-8-sig")
    print(f"\nScatter samples used for plotting: {len(df_scatter)}")

    plot_max = determine_plot_max(df)
    xlim = (0, plot_max)
    ylim = (0, plot_max)

    n_sg_out = (df[COL_SOILGRIDS] > plot_max).sum()
    n_olm_out = (df[COL_OPENLANDMAP] > plot_max).sum()
    pct_sg_out = n_sg_out / len(df) * 100
    pct_olm_out = n_olm_out / len(df) * 100

    print(f"\nPlot max: {plot_max}")
    print(f"{LABEL_SOILGRIDS} points above plot max: {n_sg_out} ({pct_sg_out:.2f}%)")
    print(f"{LABEL_OPENLANDMAP} points above plot max: {n_olm_out} ({pct_olm_out:.2f}%)")

    # =====================================================
    # 7. 绘图
    # =====================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_W, FIG_H),
        dpi=DPI,
        gridspec_kw={"width_ratios": [1.02, 1.00]}
    )

    ax_scatter = axes[0]
    ax_dist = axes[1]

    # ---------------- 左图：散点图 ----------------
    x_this = df_scatter[COL_THIS].values.astype(float)
    y_soilgrids = df_scatter[COL_SOILGRIDS].values.astype(float)
    y_openlandmap_raw = df_scatter[COL_OPENLANDMAP].values.astype(float)

    # OpenLandMap-soildb 仅绘图显示时加轻微 vertical jitter。
    rng = np.random.default_rng(JITTER_RANDOM_SEED)
    y_openlandmap_plot = y_openlandmap_raw + rng.uniform(
        low=-OPENLANDMAP_JITTER,
        high=OPENLANDMAP_JITTER,
        size=len(y_openlandmap_raw)
    )

    # 两组散点高度重叠，因此散点作为背景信息，拟合线和指标作为主要比较信息。
    # 先画 SoilGrids，后画 OpenLandMap，保证橙色产品不会完全被蓝色点云淹没。
    ax_scatter.scatter(
        x_this,
        y_soilgrids,
        s=SCATTER_SIZE,
        alpha=SOILGRIDS_ALPHA,
        color=COLOR_SOILGRIDS,
        edgecolors="white",
        linewidths=0.10,
        label=LABEL_SOILGRIDS,
        rasterized=True,
        zorder=2
    )

    ax_scatter.scatter(
        x_this,
        y_openlandmap_plot,
        s=SCATTER_SIZE,
        alpha=OPENLANDMAP_ALPHA,
        color=COLOR_OPENLANDMAP,
        edgecolors="white",
        linewidths=0.10,
        label=LABEL_OPENLANDMAP,
        rasterized=True,
        zorder=2.1
    )

    # 1:1 reference line
    ax_scatter.plot(
        [xlim[0], xlim[1]],
        [ylim[0], ylim[1]],
        linestyle="--",
        color="#111111",
        linewidth=1.20,
        label="1:1 line",
        zorder=3
    )

    sg = get_metric_row(metrics_df, LABEL_SOILGRIDS)
    olm = get_metric_row(metrics_df, LABEL_OPENLANDMAP)

    xx = np.linspace(xlim[0], xlim[1], 200)

    yy_sg = sg["Slope"] * xx + sg["Intercept"]
    yy_olm = olm["Slope"] * xx + olm["Intercept"]

    ax_scatter.plot(
        xx,
        yy_sg,
        color=COLOR_SOILGRIDS,
        linewidth=2.35,
        alpha=0.94,
        zorder=4
    )

    ax_scatter.plot(
        xx,
        yy_olm,
        color=COLOR_OPENLANDMAP,
        linewidth=2.35,
        alpha=0.94,
        zorder=4
    )

    # a 图左上角仅保留必要图例：两个外部产品 + 1:1 线。
    # R²、RMSE 和 Bias 不再放入图中，建议以正文表格呈现，避免压缩散点图主体信息。
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=COLOR_SOILGRIDS,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=8.2,
            alpha=0.90,
            label=LABEL_SOILGRIDS
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=COLOR_OPENLANDMAP,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=8.2,
            alpha=0.90,
            label=LABEL_OPENLANDMAP
        ),
        Line2D(
            [0], [0],
            color="#111111",
            linestyle="--",
            linewidth=1.65,
            label="1:1 line"
        )
    ]

    ax_scatter.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=14.0,
        frameon=True,
        framealpha=0.90,
        edgecolor=LIGHT_EDGE,
        facecolor="white",
        borderpad=0.58,
        labelspacing=0.55,
        handlelength=1.65,
        handletextpad=0.62
    )

    ax_scatter.set_xlim(xlim)
    ax_scatter.set_ylim(ylim)
    ax_scatter.set_aspect("equal", adjustable="box")

    ax_scatter.set_xlabel(f"{LABEL_THIS} SOCD (kg m⁻²)", fontsize=AXIS_LABEL_FS, labelpad=8)
    ax_scatter.set_ylabel("External product SOCD (kg m⁻²)", fontsize=AXIS_LABEL_FS, labelpad=8)
    ax_scatter.set_title(
        "(a) Product comparison",
        fontsize=TITLE_FS,
        fontweight="bold",
        pad=14
    )

    # 左图图例只解释两个外部产品和 1:1 线；两条实线拟合线在图注中说明。

    ax_scatter.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_scatter.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axes(ax_scatter)

    # ---------------- 右图：分布图 ----------------
    x_grid = np.linspace(xlim[0], xlim[1], 400)

    add_kde_or_hist(
        ax_dist,
        df[COL_THIS].values,
        COLOR_THIS,
        LABEL_THIS,
        x_grid,
        HIST_BINS,
        xlim
    )

    add_kde_or_hist(
        ax_dist,
        df[COL_SOILGRIDS].values,
        COLOR_SOILGRIDS,
        LABEL_SOILGRIDS,
        x_grid,
        HIST_BINS,
        xlim
    )

    add_kde_or_hist(
        ax_dist,
        df[COL_OPENLANDMAP].values,
        COLOR_OPENLANDMAP,
        LABEL_OPENLANDMAP,
        x_grid,
        HIST_BINS,
        xlim
    )

    # 均值竖线
    for label, col, color in [
        (LABEL_THIS, COL_THIS, COLOR_THIS),
        (LABEL_SOILGRIDS, COL_SOILGRIDS, COLOR_SOILGRIDS),
        (LABEL_OPENLANDMAP, COL_OPENLANDMAP, COLOR_OPENLANDMAP)
    ]:
        mean_val = df[col].mean()
        ax_dist.axvline(
            mean_val,
            color=color,
            linestyle="--",
            linewidth=1.35,
            alpha=0.9
        )

    ax_dist.set_xlim(xlim)
    ax_dist.set_xlabel("SOCD (kg m⁻²)", fontsize=AXIS_LABEL_FS, labelpad=8)
    ax_dist.set_ylabel("Density", fontsize=AXIS_LABEL_FS, labelpad=8)
    ax_dist.set_title(
        "(b) Distribution of sampled SOCD",
        fontsize=TITLE_FS,
        fontweight="bold",
        pad=14
    )

    ax_dist.legend(
        loc="upper right",
        fontsize=LEGEND_FS,
        frameon=True,
        framealpha=0.88,
        edgecolor=LIGHT_EDGE,
        labelspacing=0.45,
        handlelength=2.2,
        borderpad=0.50
    )

    ax_dist.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_dist.yaxis.set_major_locator(MaxNLocator(nbins=5))
    style_axes(ax_dist)

    plt.subplots_adjust(
        left=0.078,
        right=0.985,
        top=0.890,
        bottom=0.155,
        wspace=0.04
    )

    fig.savefig(out_fig_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_fig_svg, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_fig_tif, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_fig_pdf, dpi=DPI, bbox_inches="tight", facecolor="white")

    print("\nFinished!")
    print(f"Clean samples saved to: {out_clean_csv}")
    print(f"Statistics saved to: {out_stats_csv}")
    print(f"Metrics saved to: {out_metrics_csv}")
    print(f"Scatter sample saved to: {out_scatter_sample_csv}")
    print(f"Figure saved to: {out_fig_png}")
    print(f"Figure saved to: {out_fig_svg}")
    print(f"Figure saved to: {out_fig_tif}")
    print(f"Figure saved to: {out_fig_pdf}")


if __name__ == "__main__":
    main()
