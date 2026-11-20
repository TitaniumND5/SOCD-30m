#!/usr/bin/env python
"""三产品 0–30 cm SOCD 一致性检验（新全国 30 m 产品）。

输入为 GEE 脚本 01 导出的 0.2°规则网格同位点 CSV。指标是产品间一致性，
不是对实测真值的精度验证。脚本不读取或写入任何原始制图结果。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

THIS = 'YOUR_SOCD_0_30cm_kg_m2'
SG = 'SOILGRIDS_SOCD_0_30cm_kg_m2'
OLM = 'HENGL2026_SOCD_0_30cm_kg_m2'
REQUIRED = ['lon', 'lat', THIS, SG, OLM]
PRODUCTS = [('This study (RF)', THIS), ('SoilGrids v2.0', SG),
            ('OpenLandMap-soildb (Hengl et al., 2026)', OLM)]


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [x for x in REQUIRED if x not in df.columns]
    if missing:
        raise ValueError(f'CSV 缺少字段: {missing}')
    out = df.copy()
    for col in REQUIRED:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    valid = out[REQUIRED].notna().all(axis=1)
    for col in (THIS, SG, OLM):
        valid &= out[col].between(0, 150, inclusive='left')
    out = out.loc[valid].reset_index(drop=True)
    if out.empty:
        raise ValueError('清洗后没有有效同位点；请核对 GEE 导出字段和单位。')
    return out


def product_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in PRODUCTS:
        x = df[col]
        rows.append({'product': label, 'n': len(x), 'mean_kg_m2': x.mean(),
                     'std_kg_m2': x.std(ddof=1), 'min_kg_m2': x.min(),
                     'p05_kg_m2': x.quantile(.05), 'median_kg_m2': x.median(),
                     'p95_kg_m2': x.quantile(.95), 'max_kg_m2': x.max()})
    return pd.DataFrame(rows)


def agreement(df: pd.DataFrame, label: str, external: str) -> dict:
    x, y = df[THIS].to_numpy(float), df[external].to_numpy(float)
    r, p = pearsonr(x, y)
    diff = y - x  # external product minus this study
    return {'comparison': f'This study (RF) vs {label}', 'n': len(x),
            'pearson_r': r, 'r_squared_from_r': r * r, 'rmse_kg_m2': np.sqrt(np.mean(diff ** 2)),
            'mae_kg_m2': np.mean(np.abs(diff)), 'bias_external_minus_this_kg_m2': diff.mean(),
            'median_bias_external_minus_this_kg_m2': np.median(diff), 'pearson_p': p}


def make_figure(df: pd.DataFrame, metrics: pd.DataFrame, out: Path, plot_max: float) -> None:
    plt.rcParams.update({'font.family': 'Times New Roman', 'axes.unicode_minus': False})
    rng = np.random.default_rng(42)
    n = min(4500, len(df))
    # Deterministic random display subset only; metrics always use all valid samples.
    display = df.sample(n=n, random_state=42) if len(df) > n else df
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    ax = axes[0]
    colors = {'SoilGrids v2.0': '#4C78A8', 'OpenLandMap-soildb': '#E6862F'}
    for label, col in [('SoilGrids v2.0', SG), ('OpenLandMap-soildb', OLM)]:
        y = display[col].to_numpy(float)
        if label == 'OpenLandMap-soildb':
            y = y + rng.uniform(-.045, .045, len(y))  # display only
        ax.scatter(display[THIS], y, s=13, alpha=.28, c=colors[label], edgecolors='none', label=label)
    ax.plot([0, plot_max], [0, plot_max], '--', color='#444444', lw=1)
    ax.set(xlim=(0, plot_max), ylim=(0, plot_max), xlabel='This study SOCD (kg m$^{-2}$)',
           ylabel='External-product SOCD (kg m$^{-2}$)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(alpha=.25, linestyle='--')
    lines = [f"{row.comparison.replace('This study (RF) vs ', '')}: r={row.pearson_r:.3f}, "
             f"RMSE={row.rmse_kg_m2:.2f}, Bias={row.bias_external_minus_this_kg_m2:.2f}"
             for row in metrics.itertuples()]
    ax.text(.02, .02, '\n'.join(lines), transform=ax.transAxes, fontsize=9, va='bottom')
    ax = axes[1]
    for label, col in PRODUCTS:
        ax.hist(df[col], bins=60, density=True, histtype='step', linewidth=1.8, label=label)
    ax.set(xlim=(0, plot_max), xlabel='SOCD (kg m$^{-2}$)', ylabel='Density')
    ax.grid(alpha=.25, linestyle='--'); ax.legend(frameon=False)
    fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True, help='GEE三产品规则网格采样CSV')
    parser.add_argument('--output-dir', type=Path, required=True, help='新建的结果目录；不得指向旧结果目录')
    parser.add_argument('--plot-max', type=float, default=20.0, help='图示上限；不影响统计')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_csv(args.input_csv)
    df = clean(raw)
    stats = product_stats(df)
    metrics = pd.DataFrame([agreement(df, 'SoilGrids v2.0', SG), agreement(df, 'OpenLandMap-soildb', OLM)])
    df.to_csv(args.output_dir / 'three_product_valid_grid_samples.csv', index=False, encoding='utf-8-sig')
    stats.to_csv(args.output_dir / 'product_distribution_statistics.csv', index=False, encoding='utf-8-sig')
    metrics.to_csv(args.output_dir / 'product_agreement_metrics.csv', index=False, encoding='utf-8-sig')
    make_figure(df, metrics, args.output_dir / 'three_product_agreement_panel', args.plot_max)
    print(f'Raw samples: {len(raw)}; valid paired samples: {len(df)}')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
