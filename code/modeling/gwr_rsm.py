# ======================== GWR最终版：按地理分区训练 + 空间独立测试集评价 + 投影坐标距离 + log(SOCD)目标变换 ========================
# 目的：
# 1) 对每个 Region 分区分别训练 GWR；
# 2) 每个分区内部使用与 RF / SVR / CNN 一致的 0.09° 空间网格划分训练集、验证集和测试集；
# 3) GWR 的跨模型比较精度只基于“训练集拟合 -> 测试集预测”的空间独立测试集结果；
# 4) GWR 内部空间权重距离使用投影坐标 x_proj / y_proj，经纬度仅用于输出和制图；
# 5) 保留每个分区的全样本 GWR 拟合结果，用于局部R²、局部系数、空间非平稳性解释；
# 6) 不再把全样本 fitted_values 当作跨模型比较精度；
# 7) GWR内部对 log(SOCD + eps) 建模，预测后反变换到原始 SOCD 尺度计算精度，避免无约束局部线性外推产生极端负值。

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

import geopandas as gpd
from shapely.geometry import Point
import geopy.distance
from scipy import stats
from scipy.stats import gaussian_kde
from pyproj import Transformer

warnings.filterwarnings('ignore')

# ======================== 0. 基础配置 ========================
plt.rcParams.clear()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.switch_backend('Agg')

# 多核优化
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['OPENBLAS_NUM_THREADS'] = '20'
os.environ['VECLIB_MAXIMUM_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '20'

REGION_EN_MAP = {
    'dongbei': '东北区',
    'huabei': '华北区',
    'huadong': '华东区',
    'huanan': '华南区',
    'xibei': '西北区',
    'xinan': '西南区'
}
ALL_REGIONS = list(REGION_EN_MAP.keys())

COL_NAME_MAP = {
    '真实土层厚度': 'True_Soil_Depth',
    '植被综合因子1': 'Vegetation_Factor1',
    '植被综合因子2': 'Vegetation_Factor2'
}

CONFIG = {
    # Set data_path to the local training CSV used in the study.
    'data_path': r"PATH_TO_TRAINING_DATA.csv",

    # 单分区模式输出路径；批量运行时会自动改成 output_base 下的各分区文件夹
    'output_path': os.path.join(os.path.dirname(__file__), "outputs", "gwr_rsm", "Region_dongbei_东北区") + os.sep,
    'output_base': os.path.join(os.path.dirname(__file__), "outputs", "gwr_rsm"),

    'test_size': 0.15,
    'val_size': 0.15,
    'random_state': 42,
    'grid_size': 0.09,

    # fixed=False 时，mgwr 返回的 bw 是邻居数量；这里仍使用 bisquare + adaptive，与原GWR代码一致。
    'kernel': 'bisquare',
    'fixed': False,

    # GWR 对共线性敏感，保留 VIF 控制。
    'vif_threshold': 15,

    # log-GWR：模型内部拟合 log(SOCD + eps)，反变换后在原始 SOCD 尺度计算精度。
    'use_log_target': True,
    'log_eps': 1e-6,

    # 置换重要性为避免耗时，最多抽取测试集500个样本计算。
    'perm_sample_size': 500,
    'perm_repeats': 3,

    # 单分区运行时修改此项；批量运行时会自动遍历全部分区。
    'target_region': 'dongbei'
}


# ======================== 1. 工具函数 ========================
def add_projected_coords(df):
    """
    将经纬度转换为投影坐标，仅用于 GWR 空间权重距离计算。
    与你最新全样本 GWR 代码保持一致：使用 EPSG:3857 作为投影坐标。
    若后续需要更严格的中国区域投影，可替换为 China Albers。
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(df['longitude'].values, df['latitude'].values)
    df = df.copy()
    df['x_proj'] = x
    df['y_proj'] = y
    return df


def ensure_output_dir(config):
    os.makedirs(config['output_path'], exist_ok=True)


def auto_calc_bw_range(df_input, target_region, n_features=0):
    """
    根据样本量、SOCD变异和平均邻近距离给出自适应带宽搜索范围。
    注意：fixed=False 下 bw 是邻居数量，而不是距离阈值。
    """
    sample_count = len(df_input)
    if sample_count <= max(n_features + 5, 20):
        raise ValueError(
            f"Region={target_region} 样本过少：{sample_count}，无法稳定拟合 GWR（自变量数={n_features}）。"
        )

    socd_mean = df_input['SOCD_kg_m2'].mean()
    socd_cv = df_input['SOCD_kg_m2'].std() / socd_mean if socd_mean != 0 else 1.0

    sample_size = min(100, sample_count)
    sample_coords = df_input[['latitude', 'longitude']].sample(sample_size, random_state=42).values
    distances = []
    for i in range(len(sample_coords) - 1):
        coord1 = (sample_coords[i][0], sample_coords[i][1])
        coord2 = (sample_coords[i + 1][0], sample_coords[i + 1][1])
        distances.append(geopy.distance.geodesic(coord1, coord2).km)
    avg_distance = np.mean(distances) if distances else 10.0

    print(f"\n📊 Region={target_region}（{REGION_EN_MAP.get(target_region, target_region)}）GWR样本特征：")
    print(f"   样本数量：{sample_count}条 | SOCD变异系数：{socd_cv:.2f} | 平均邻近距离：{avg_distance:.2f} km")

    if avg_distance < 5:
        base_min, base_max = 40, 90
    elif 5 <= avg_distance < 10:
        base_min, base_max = 45, 95
    else:
        base_min, base_max = 50, 100

    if sample_count < 100:
        base_min -= 5
        base_max += 10
    elif sample_count > 1000:
        base_min += 5
        base_max -= 5

    if socd_cv > 0.8:
        base_min -= 10
        base_max -= 10
    elif socd_cv < 0.5:
        base_min += 5
        base_max += 5

    # fixed=False 下 bw 是邻居数量，而不是距离阈值。
    # 宽带宽敏感性版本：使用样本数量比例作为搜索范围，避免 GWR 过度局部化。
    bw_min = max(n_features + 2, int(0.15 * sample_count))
    bw_max = min(sample_count - 2, int(0.75 * sample_count))

    if bw_max <= bw_min:
        bw_min = max(n_features + 2, 10, min(sample_count // 4, sample_count - 3))
        bw_max = max(bw_min + 1, min(sample_count - 2, sample_count // 2))

    print(f"🎯 宽带宽敏感性搜索范围：{bw_min} ~ {bw_max}")
    return int(bw_min), int(bw_max)


def preprocess_data(df_raw, config):
    """
    分区数据预处理：
    - 筛选单一地理分区；
    - 字段筛选；
    - 缺失填充；
    - 高相关冗余特征剔除；
    - PCA合并植被/遥感指数；
    - VIF控制；
    - 标准化。
    """
    df_raw = df_raw.rename(columns={k: v for k, v in COL_NAME_MAP.items() if k in df_raw.columns})
    initial_count = len(df_raw)

    if 'Region' not in df_raw.columns:
        raise ValueError("输入数据缺少 Region 字段，无法按地理分区训练。")

    target_region = config['target_region']
    df_region_raw = df_raw[df_raw['Region'] == target_region].copy()
    print(f"📌 筛选 Region={target_region}（{REGION_EN_MAP.get(target_region, target_region)}）样本："
          f"{initial_count}条 → 筛选后 {len(df_region_raw)} 条")

    if len(df_region_raw) < 30:
        raise ValueError(f"Region={target_region} 样本数过少：{len(df_region_raw)}，跳过建模。")

    core_columns = [
        'FID', 'Samples', 'profile_id', 'latitude', 'longitude',
        'upper_dept', 'lower_dept', 'True_Soil_Depth', 'Depth', 'SOCD__kg_m',
        'BGREEN', 'BNIR', 'BRED', 'DEM', 'NDVI', 'SAVI', 'OSAVI', 'EVI',
        'LSWI', 'MSAVI', 'NDWI', 'TWI', 'MAT', 'MAP', 'PET', 'AI',
        'Sand', 'Silt', 'Clay', 'slope_deg', '土地覆盖类型',
        'pH', 'Albedo', 'plan_cur', 'Region', 'CLCD'
    ]
    core_columns = [col for col in core_columns if col in df_region_raw.columns]
    df = df_region_raw[core_columns].copy()

    # 目标变量
    df.rename(columns={'SOCD__kg_m': 'SOCD_kg_m2'}, inplace=True)
    df = df.dropna(subset=['latitude', 'longitude', 'SOCD_kg_m2'])

    # 缺失值填充：分区内按 CLCD 中位数填充，再整体中位数兜底。
    exclude_cols = ['latitude', 'longitude', 'SOCD_kg_m2', '土地覆盖类型', 'Region', 'CLCD']
    fill_cols = [col for col in df.columns if col not in exclude_cols]
    for col in fill_cols:
        if df[col].dtype.kind in 'biufc':
            if 'CLCD' in df.columns:
                df[col] = df.groupby('CLCD')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    # 高相关冗余特征剔除：保留与 SOCD 相关性更高的变量。
    feat_cols_to_check = [
        'DEM', 'TWI', 'slope_deg', 'plan_cur', 'MAT', 'MAP', 'PET', 'AI',
        'Sand', 'Silt', 'Clay', 'pH', 'LSWI', 'Albedo', 'True_Soil_Depth'
    ]
    feat_cols_to_check = [col for col in feat_cols_to_check if col in df.columns]
    to_drop = []

    if len(feat_cols_to_check) > 1:
        corr_matrix = df[feat_cols_to_check].corr().abs()
        socd_corr = df[feat_cols_to_check].corrwith(df['SOCD_kg_m2']).abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        for col1, col2 in high_corr_pairs:
            to_drop.append(col2 if socd_corr[col1] >= socd_corr[col2] else col1)
        to_drop = list(set(to_drop))
        if to_drop:
            print(f"🔍 自动剔除冗余特征（相关系数>0.9）：{to_drop}")
            df = df.drop(columns=to_drop)
        else:
            print("✅ 无高度冗余特征")
    else:
        print("⚠️ 特征数量不足，跳过冗余特征筛选")

    # PCA 合并植被/遥感指数
    veg_cols = ['BGREEN', 'BNIR', 'BRED', 'NDVI', 'SAVI', 'OSAVI', 'EVI', 'MSAVI', 'NDWI']
    veg_cols = [col for col in veg_cols if col in df.columns]
    if veg_cols:
        for col in veg_cols:
            df[col] = df[col].fillna(df[col].median())
        pca = PCA(n_components=2, random_state=config['random_state'])
        pca_result = pca.fit_transform(df[veg_cols])
        df['Vegetation_Factor1'] = pca_result[:, 0]
        df['Vegetation_Factor2'] = pca_result[:, 1]
        print(f"🌿 植被指数PCA总解释方差比：{sum(pca.explained_variance_ratio_):.4f}")
        df = df.drop(columns=veg_cols)

    indep_cols = [
        'DEM', 'TWI', 'slope_deg', 'plan_cur', 'MAT', 'MAP', 'PET', 'AI',
        'Sand', 'Silt', 'Clay', 'pH', 'Vegetation_Factor1', 'Vegetation_Factor2',
        'LSWI', 'Albedo', 'True_Soil_Depth'
    ]
    indep_cols = [col for col in indep_cols if col in df.columns and col not in to_drop]

    # VIF 检验：GWR 对局部共线性敏感，因此保留 VIF 控制。
    if len(indep_cols) > 1 and len(df) > len(indep_cols) + 2:
        vif_data = df[indep_cols].copy()
        vif_data = sm.add_constant(vif_data)
        try:
            vif_df = pd.DataFrame({
                'Variable': vif_data.columns,
                'VIF': [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
            })
            vif_df = vif_df[vif_df['Variable'] != 'const']
            high_vif = vif_df[vif_df['VIF'] > config['vif_threshold']]['Variable'].tolist()
            if high_vif:
                print(f"⚠️ 移除极强共线性变量（VIF>{config['vif_threshold']}）：{high_vif}")
                df = df.drop(columns=high_vif)
                indep_cols = [col for col in indep_cols if col not in high_vif]
            vif_df.to_csv(os.path.join(config['output_path'], 'GWR_VIF检验结果.csv'), index=False, encoding='utf_8_sig')
        except Exception as e:
            print(f"⚠️ VIF计算失败，跳过VIF筛选。原因：{repr(e)}")
    else:
        print("⚠️ 样本数或特征数不足，跳过 VIF 筛选。")

    # 与现有 RF/SVR/CNN 代码保持一致：在分区全体建模样本上标准化后再空间划分。
    scaler = StandardScaler()
    df[indep_cols] = scaler.fit_transform(df[indep_cols])

    df = df.reset_index(drop=True)
    print(f"✅ Region={target_region} 预处理完成：有效样本 {len(df)} 条 | GWR建模自变量 {len(indep_cols)} 个")
    print(f"✅ GWR最终自变量：{indep_cols}")

    return df, indep_cols, scaler


def spatial_split(df, config):
    """与 RF/SVR/CNN 一致：0.09° 空间网格独立划分训练/验证/测试。"""
    df = df.copy()
    grid_size = config['grid_size']
    df['lat_grid'] = (df['latitude'] // grid_size).astype(int)
    df['lon_grid'] = (df['longitude'] // grid_size).astype(int)
    df['grid_id'] = df['lat_grid'].astype(str) + '_' + df['lon_grid'].astype(str)

    grid_ids = df['grid_id'].unique()
    if len(grid_ids) < 3:
        raise ValueError(f"空间网格数量过少：{len(grid_ids)}，无法划分训练/验证/测试。")

    train_val_grids, test_grids = train_test_split(
        grid_ids,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    train_grids, val_grids = train_test_split(
        train_val_grids,
        test_size=config['val_size'] / (1 - config['test_size']),
        random_state=config['random_state']
    )

    df_train = df[df['grid_id'].isin(train_grids)].copy()
    df_val = df[df['grid_id'].isin(val_grids)].copy()
    df_test = df[df['grid_id'].isin(test_grids)].copy()

    split_stats = pd.DataFrame({
        'Region': [config['target_region']] * 3,
        'Dataset': ['Train', 'Validation', 'Test'],
        'Sample_Count': [len(df_train), len(df_val), len(df_test)],
        'Ratio': [len(df_train) / len(df), len(df_val) / len(df), len(df_test) / len(df)],
        'SOCD_Mean': [df_train['SOCD_kg_m2'].mean(), df_val['SOCD_kg_m2'].mean(), df_test['SOCD_kg_m2'].mean()],
        'SOCD_Std': [df_train['SOCD_kg_m2'].std(), df_val['SOCD_kg_m2'].std(), df_test['SOCD_kg_m2'].std()],
        'Grid_Count': [df_train['grid_id'].nunique(), df_val['grid_id'].nunique(), df_test['grid_id'].nunique()]
    })
    split_stats.to_csv(os.path.join(config['output_path'], f'Region_{config["target_region"]}_GWR_spatial_split_stats.csv'),
                       index=False, encoding='utf_8_sig')
    print('\n📊 空间独立数据集划分完成：')
    print(split_stats)
    return df_train, df_val, df_test


def adaptive_gwr_predict_train_to_test(coords_train, X_train, y_train, coords_test, X_test, bw, kernel='bisquare'):
    """
    使用训练集局部回归参数预测测试集/验证集。
    关键点：
    - fixed=False 时，bw 表示邻居数量，而不是距离阈值；
    - 每个待预测点只用训练样本估计局部回归系数；
    - 不使用待预测点真实 SOCD 参与拟合，从而形成独立预测。
    """
    coords_train = np.asarray(coords_train, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    coords_test = np.asarray(coords_test, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    n_train = len(coords_train)
    n_test = len(coords_test)
    p = X_train.shape[1]

    k = int(round(float(bw)))
    k = max(k, p + 2)
    k = min(k, n_train)

    X_train_i = np.hstack([np.ones((n_train, 1)), X_train])
    X_test_i = np.hstack([np.ones((n_test, 1)), X_test])
    y_pred = np.zeros(n_test)

    for i in range(n_test):
        dist = np.sqrt(((coords_train - coords_test[i]) ** 2).sum(axis=1))
        h = np.partition(dist, k - 1)[k - 1] + 1e-12

        if kernel == 'bisquare':
            weights = (1 - (dist / h) ** 2) ** 2
            weights[dist > h] = 0
        else:
            weights = np.exp(-(dist / h) ** 2)

        if np.count_nonzero(weights > 0) < p + 2:
            nearest_idx = np.argsort(dist)[:p + 2]
            weights[:] = 0
            weights[nearest_idx] = 1

        sw = np.sqrt(weights).reshape(-1, 1)
        Xw = X_train_i * sw
        yw = y_train * sw.ravel()
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        y_pred[i] = X_test_i[i] @ beta

    return y_pred


def gwr_predict_with_mgwr_or_manual(coords_train, y_train, X_train, coords_test, X_test, bw, config, train_results=None):
    """优先尝试 mgwr.predict；若失败，自动使用手动 adaptive GWR 预测。"""
    try:
        print('🔍 尝试使用 mgwr.predict 进行独立样本预测...')
        model = GWR(coords_train, y_train, X_train, bw=bw, kernel=config['kernel'], fixed=config['fixed'])
        if train_results is None:
            train_results = model.fit()
        pred_results = model.predict(
            coords_test,
            X_test,
            exog_scale=train_results.scale,
            exog_resid=train_results.resid_response
        )
        y_pred = pred_results.predictions.ravel()
        print('✅ mgwr.predict 独立样本预测成功')
        return y_pred, 'mgwr.predict'
    except Exception as e:
        print(f'⚠️ mgwr.predict 失败，切换为手动自适应 GWR 预测。原因：{repr(e)}')
        y_pred = adaptive_gwr_predict_train_to_test(
            coords_train, X_train, y_train, coords_test, X_test, bw, kernel=config['kernel']
        )
        print('✅ 手动自适应 GWR 独立样本预测完成')
        return y_pred, 'manual_adaptive_gwr'


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = round(r2_score(y_true, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
    mae = round(mean_absolute_error(y_true, y_pred), 4)
    rpd = round(np.std(y_true, ddof=1) / rmse, 4) if rmse != 0 else np.nan
    return r2, rmse, mae, rpd


def to_log_target(y_raw, eps):
    """SOCD 原始尺度 -> log(SOCD + eps)。用于 GWR 内部建模。"""
    y_raw = np.asarray(y_raw, dtype=float).reshape(-1, 1)
    if np.any(y_raw < 0):
        raise ValueError('SOCD存在负值，无法进行 log(SOCD + eps) 建模。')
    return np.log(y_raw + eps)


def from_log_target(y_log, eps):
    """log(SOCD + eps) -> SOCD 原始尺度。仅做反变换，避免极端负值。"""
    y_log = np.asarray(y_log, dtype=float).ravel()
    y_raw = np.exp(y_log) - eps
    # 理论下界约为 -eps；这里仅消除数值误差导致的微小负值，不是后验暴力截断。
    return np.maximum(y_raw, 0.0)


def plot_pred_vs_obs(y_true, y_pred, metrics, save_path):
    test_r2, test_rmse, test_mae, test_rpd = metrics
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        y_true_sorted, y_pred_sorted, z_sorted = y_true[idx], y_pred[idx], z[idx]
        sns.scatterplot(x=y_true_sorted, y=y_pred_sorted, c=z_sorted, cmap='viridis', alpha=0.6, s=40, edgecolor='none')
    except Exception:
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=40, edgecolor='none')

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
    plt.plot(y_true, slope * y_true + intercept, 'r-', lw=2)

    plt.xlabel('Lab-measured SOCD (kg/m²)', fontsize=17)
    plt.ylabel('Predicted SOCD (kg/m²)', fontsize=17)

    ax = plt.gca()
    x_axis_min, x_axis_max = ax.get_xlim()
    y_axis_min, y_axis_max = ax.get_ylim()
    x_pos = x_axis_min + (x_axis_max - x_axis_min) * 0.05
    y_pos = y_axis_min + (y_axis_max - y_axis_min) * 0.95

    plt.text(
        x=x_pos, y=y_pos,
        s=f'R²={test_r2:.4f}\nRMSE={test_rmse:.4f} (kg/m²)\nMAE={test_mae:.4f} (kg/m²)\nRPD={test_rpd:.4f}',
        fontsize=18,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        fontfamily='Times New Roman',
        ha='left', va='top', linespacing=1.5
    )


    plt.xticks(fontsize=16, fontfamily='Times New Roman')
    plt.yticks(fontsize=16, fontfamily='Times New Roman')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']


def plot_error_hist(errors, metrics, save_path):
    _, rmse, mae, _ = metrics
    errors = np.asarray(errors).ravel()

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})
    sns.histplot(errors, bins=30, kde=True, edgecolor='black')
    plt.axvline(x=0, linestyle='--', linewidth=2, label='Zero Error')
    plt.text(
        x=np.nanmax(errors) * 0.65 if np.nanmax(errors) > 0 else 0,
        y=plt.gca().get_ylim()[1] * 0.80,
        s=f'RMSE = {rmse:.4f} kg/m²\nMAE = {mae:.4f} kg/m²',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontfamily='Times New Roman'
    )
    plt.xlabel('Prediction Error (kg/m²)', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.legend(prop={'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']


def calculate_permutation_importance_gwr(coords_train, X_train, y_train_model, coords_eval, X_eval, y_eval_raw, feature_names, bw, config):
    """
    基于独立测试样本的 GWR 置换重要性：基准测试集R² - 打乱后测试集R²。
    log-GWR 情况下：GWR 内部预测 log(SOCD + eps)，随后反变换到原始 SOCD 尺度，再计算 R²。
    """
    n_eval = len(X_eval)
    sample_size = min(config['perm_sample_size'], n_eval)
    rng = np.random.RandomState(config['random_state'])
    sample_idx = rng.choice(n_eval, sample_size, replace=False)

    X_base = X_eval[sample_idx].copy()
    y_base_raw = np.asarray(y_eval_raw).ravel()[sample_idx].copy()
    coords_base = coords_eval[sample_idx].copy()
    eps = float(config.get('log_eps', 1e-6))

    y_pred_base_model = adaptive_gwr_predict_train_to_test(
        coords_train, X_train, y_train_model, coords_base, X_base, bw, kernel=config['kernel']
    )
    y_pred_base = from_log_target(y_pred_base_model, eps) if config.get('use_log_target', True) else y_pred_base_model
    base_r2 = r2_score(y_base_raw.ravel(), y_pred_base.ravel())

    importances = np.zeros((config['perm_repeats'], len(feature_names)))
    for i, feat in enumerate(feature_names):
        for n in range(config['perm_repeats']):
            X_perm = X_base.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            y_pred_perm_model = adaptive_gwr_predict_train_to_test(
                coords_train, X_train, y_train_model, coords_base, X_perm, bw, kernel=config['kernel']
            )
            y_pred_perm = from_log_target(y_pred_perm_model, eps) if config.get('use_log_target', True) else y_pred_perm_model
            perm_r2 = r2_score(y_base_raw.ravel(), y_pred_perm.ravel())
            importances[n, i] = base_r2 - perm_r2

    perm_imp_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Permutation_Importance': importances.mean(axis=0),
        'Importance_Std': importances.std(axis=0)
    }).sort_values('Permutation_Importance', ascending=False)

    return perm_imp_df, base_r2


def plot_permutation_importance(perm_imp_df, save_path):
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    perm_imp_final = perm_imp_df[perm_imp_df['Permutation_Importance'] > 0.0005].head(15)
    if perm_imp_final.empty:
        perm_imp_final = perm_imp_df.head(15)

    colors = ['#1f77b4' if i < 3 else '#6baed6' for i in range(len(perm_imp_final))]
    bars = sns.barplot(x='Permutation_Importance', y='Feature_Name', data=perm_imp_final, palette=colors)
    for bar in bars.patches:
        w = bar.get_width()
        bars.text(w + 0.0001, bar.get_y() + bar.get_height() / 2,
                  f'{w:.4f}', va='center', fontsize=10, fontfamily='Times New Roman')

    plt.xlabel('Permutation Importance (ΔR²)', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    plt.xticks(fontsize=11, fontfamily='Times New Roman')
    plt.yticks(fontsize=11, fontfamily='Times New Roman')
    if perm_imp_final['Permutation_Importance'].max() > 0:
        plt.xlim(0, perm_imp_final['Permutation_Importance'].max() * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']


def save_prediction_distribution_check(y_train, y_val, y_test, y_test_pred, config):
    summary = pd.DataFrame([
        {
            'Region': config['target_region'],
            'Dataset': 'Train_y',
            'Mean': float(np.mean(y_train)),
            'Std': float(np.std(y_train, ddof=1)),
            'Min': float(np.min(y_train)),
            'Max': float(np.max(y_train))
        },
        {
            'Region': config['target_region'],
            'Dataset': 'Validation_y',
            'Mean': float(np.mean(y_val)),
            'Std': float(np.std(y_val, ddof=1)),
            'Min': float(np.min(y_val)),
            'Max': float(np.max(y_val))
        },
        {
            'Region': config['target_region'],
            'Dataset': 'Test_y_true',
            'Mean': float(np.mean(y_test)),
            'Std': float(np.std(y_test, ddof=1)),
            'Min': float(np.min(y_test)),
            'Max': float(np.max(y_test))
        },
        {
            'Region': config['target_region'],
            'Dataset': 'Test_y_pred',
            'Mean': float(np.mean(y_test_pred)),
            'Std': float(np.std(y_test_pred, ddof=1)),
            'Min': float(np.min(y_test_pred)),
            'Max': float(np.max(y_test_pred))
        }
    ])
    summary.to_csv(os.path.join(config['output_path'], f'Region_{config["target_region"]}_GWR_TEST_prediction_distribution_check.csv'),
                   index=False, encoding='utf_8_sig')


# ======================== 2. 单分区 GWR 主流程 ========================
def train_and_evaluate_gwr_region(df, indep_cols, config):
    target_region = config['target_region']
    region_name = REGION_EN_MAP.get(target_region, target_region)

    # 1) 空间独立划分
    df_train, df_val, df_test = spatial_split(df, config)

    # 2) 加投影坐标：仅用于 GWR 空间距离权重计算
    df = add_projected_coords(df)
    df_train = add_projected_coords(df_train)
    df_val = add_projected_coords(df_val)
    df_test = add_projected_coords(df_test)

    coords_train = df_train[['x_proj', 'y_proj']].values
    X_train = df_train[indep_cols].values
    y_train_raw = df_train['SOCD_kg_m2'].values.reshape(-1, 1)

    coords_val = df_val[['x_proj', 'y_proj']].values
    X_val = df_val[indep_cols].values
    y_val_raw = df_val['SOCD_kg_m2'].values.reshape(-1, 1)

    coords_test = df_test[['x_proj', 'y_proj']].values
    X_test = df_test[indep_cols].values
    y_test_raw = df_test['SOCD_kg_m2'].values.reshape(-1, 1)

    # log-GWR：模型内部使用 log(SOCD + eps)，评价和输出仍回到 SOCD 原始尺度。
    eps = float(config.get('log_eps', 1e-6))
    if config.get('use_log_target', True):
        y_train = to_log_target(y_train_raw, eps)
        y_val_model = to_log_target(y_val_raw, eps)
        y_test_model = to_log_target(y_test_raw, eps)
        target_transform = f'log(SOCD + {eps:g})'
    else:
        y_train = y_train_raw
        y_val_model = y_val_raw
        y_test_model = y_test_raw
        target_transform = 'raw SOCD'

    # 3) 训练集 GWR：用于独立测试集预测
    bw_min, bw_max = auto_calc_bw_range(df_train, target_region, n_features=len(indep_cols))
    print('\n🔍 基于训练集搜索 GWR 最优带宽...')
    bw_selector = Sel_BW(coords_train, y_train, X_train, kernel=config['kernel'], fixed=config['fixed'])
    optimal_bw = bw_selector.search(search_method='golden_section', bw_min=bw_min, bw_max=bw_max)
    print(f'✅ 训练集最优带宽：{optimal_bw:.2f}')

    print('\n🚀 基于训练集拟合 GWR 模型...')
    gwr_train_model = GWR(coords_train, y_train, X_train, bw=optimal_bw, kernel=config['kernel'], fixed=config['fixed'])
    gwr_train_results = gwr_train_model.fit()
    print(f'✅ 训练集 GWR 拟合完成（目标变量：{target_transform}）')

    # 验证集预测：仅作为检查，不进入主表。预测后反变换回 SOCD 原始尺度。
    y_val_pred_model, val_pred_method = gwr_predict_with_mgwr_or_manual(
        coords_train, y_train, X_train, coords_val, X_val, optimal_bw, config, train_results=gwr_train_results
    )
    y_val_pred = from_log_target(y_val_pred_model, eps) if config.get('use_log_target', True) else y_val_pred_model
    val_metrics = calc_metrics(y_val_raw.ravel(), y_val_pred)

    # 测试集预测：这是分区四模型比较的核心精度。预测后反变换回 SOCD 原始尺度。
    y_test_pred_model, test_pred_method = gwr_predict_with_mgwr_or_manual(
        coords_train, y_train, X_train, coords_test, X_test, optimal_bw, config, train_results=gwr_train_results
    )
    y_test_pred = from_log_target(y_test_pred_model, eps) if config.get('use_log_target', True) else y_test_pred_model
    test_metrics = calc_metrics(y_test_raw.ravel(), y_test_pred)
    test_r2, test_rmse, test_mae, test_rpd = test_metrics

    print(f'\n📈 Region={target_region}（{region_name}）GWR空间独立测试集精度（用于分区四模型统一比较）：')
    print(f'   R²：{test_r2:.4f}')
    print(f'   RMSE：{test_rmse:.4f} kg/m²')
    print(f'   MAE：{test_mae:.4f} kg/m²')
    print(f'   RPD：{test_rpd:.4f}')
    print(f'   测试集预测方法：{test_pred_method}')

    # 保存统一比较主表：推荐用于四模型主表
    compare_metrics = pd.DataFrame([{
        '模型': 'GWR',
        'Region': target_region,
        'Region_Name': region_name,
        '评价口径': '空间独立测试集',
        '目标变换': target_transform,
        'R²': test_r2,
        'RMSE (kg/m2)': test_rmse,
        'MAE (kg/m2)': test_mae,
        'RPD': test_rpd,
        '最优带宽': optimal_bw,
        '训练样本数': len(df_train),
        '验证样本数': len(df_val),
        '测试样本数': len(df_test),
        '自变量数': len(indep_cols),
        '测试集预测方法': test_pred_method
    }])
    compare_metrics.to_csv(os.path.join(config['output_path'], f'Region_{target_region}_GWR_空间独立测试集_四模型对比指标.csv'),
                           index=False, encoding='utf_8_sig')

    # 兼容旧文件名：但内容已经是正确的测试集精度，不再是全样本 fitted 精度
    compare_metrics.rename(columns={'评价口径': '数据'}).to_csv(
        os.path.join(config['output_path'], 'GWR_RF精度对比表.csv'),
        index=False, encoding='utf_8_sig'
    )

    diagnostics = pd.DataFrame([{
        '模型类型': f'Region={target_region} GWR',
        'Region_Name': region_name,
        '评价口径': '空间独立测试集预测',
        '目标变换': target_transform,
        '训练集最优带宽': optimal_bw,
        'Validation_R²': val_metrics[0],
        'Validation_RMSE (kg/m²)': val_metrics[1],
        'Validation_MAE (kg/m²)': val_metrics[2],
        'Validation_RPD': val_metrics[3],
        'Test_R²': test_r2,
        'Test_RMSE (kg/m²)': test_rmse,
        'Test_MAE (kg/m²)': test_mae,
        'Test_RPD': test_rpd,
        '训练样本数': len(df_train),
        '验证样本数': len(df_val),
        '测试样本数': len(df_test),
        '自变量数': len(indep_cols),
        '验证集预测方法': val_pred_method,
        '测试集预测方法': test_pred_method
    }])
    diagnostics.to_csv(os.path.join(config['output_path'], 'GWR_空间独立测试集_模型诊断.csv'),
                       index=False, encoding='utf_8_sig')
    diagnostics.to_csv(os.path.join(config['output_path'], 'GWR模型诊断_优化版.csv'),
                       index=False, encoding='utf_8_sig')

    # 测试集预测结果表
    test_result = pd.DataFrame({
        '纬度': df_test['latitude'].values,
        '经度': df_test['longitude'].values,
        'Region': target_region,
        'Region_Name': region_name,
        '实测SOCD(kg/m2)': y_test_raw.ravel(),
        '预测SOCD(kg/m2)': y_test_pred,
        '误差(kg/m2)': y_test_pred - y_test_raw.ravel(),
        '数据集类型': 'Test',
        'grid_id': df_test['grid_id'].values
    })
    test_result.to_csv(os.path.join(config['output_path'], f'Region_{target_region}_GWR_TEST_predictions.csv'),
                       index=False, encoding='utf-8-sig')

    # 预测分布检查
    save_prediction_distribution_check(y_train_raw.ravel(), y_val_raw.ravel(), y_test_raw.ravel(), y_test_pred, config)

    # 测试集散点图和误差图
    plot_pred_vs_obs(
        y_test_raw.ravel(), y_test_pred, test_metrics,
        os.path.join(config['output_path'], f'Region_{target_region}_GWR_TEST_pred_vs_obs.png'),
    )
    plot_error_hist(
        y_test_raw.ravel() - y_test_pred, test_metrics,
        os.path.join(config['output_path'], f'Region_{target_region}_GWR_TEST_error_histogram.png')
    )

    # 4) 测试集置换重要性：跨模型统一解释口径
    print('\n🔍 计算 GWR 测试集置换重要性（训练集拟合 -> 测试集打乱）...')
    perm_imp_df, base_r2 = calculate_permutation_importance_gwr(
        coords_train, X_train, y_train,
        coords_test, X_test, y_test_raw,
        indep_cols, optimal_bw, config
    )
    perm_imp_df.to_csv(os.path.join(config['output_path'], f'Region_{target_region}_GWR_TEST_permutation_importance.csv'),
                       index=False, encoding='utf_8_sig')
    plot_permutation_importance(
        perm_imp_df,
        os.path.join(config['output_path'], f'Region_{target_region}_GWR_TEST_permutation_importance.png')
    )
    print(f'✅ GWR测试集置换重要性完成，基准R²={base_r2:.4f}')

    # 5) 全样本 GWR 拟合：仅用于局部解释，不进入分区四模型主精度表
    print('\n🔍 开始分区全样本 GWR 拟合，用于局部R²和局部系数解释（不作为四模型主精度）...')
    coords_all = df[['x_proj', 'y_proj']].values
    X_all = df[indep_cols].values
    y_all_raw = df['SOCD_kg_m2'].values.reshape(-1, 1)
    y_all = to_log_target(y_all_raw, eps) if config.get('use_log_target', True) else y_all_raw

    bw_min_all, bw_max_all = auto_calc_bw_range(df, target_region, n_features=len(indep_cols))
    bw_selector_all = Sel_BW(coords_all, y_all, X_all, kernel=config['kernel'], fixed=config['fixed'])
    optimal_bw_all = bw_selector_all.search(search_method='golden_section', bw_min=bw_min_all, bw_max=bw_max_all)
    print(f'✅ 分区全样本解释用 GWR 最优带宽：{optimal_bw_all:.2f}')

    gwr_full_model = GWR(coords_all, y_all, X_all, bw=optimal_bw_all, kernel=config['kernel'], fixed=config['fixed'])
    gwr_full_results = gwr_full_model.fit()
    y_all_fit_model = gwr_full_results.fitted_values.ravel() if hasattr(gwr_full_results, 'fitted_values') else gwr_full_results.predy.ravel()
    y_all_fit = from_log_target(y_all_fit_model, eps) if config.get('use_log_target', True) else y_all_fit_model
    fullfit_metrics = calc_metrics(y_all_raw.ravel(), y_all_fit)

    pd.DataFrame([{
        '模型类型': f'Region={target_region} GWR',
        'Region_Name': region_name,
        '评价口径': '分区全样本拟合，仅用于局部解释，不进入四模型预测精度主表',
        '目标变换': target_transform,
        '全样本最优带宽': optimal_bw_all,
        'FullFit_R²': fullfit_metrics[0],
        'FullFit_RMSE (kg/m²)': fullfit_metrics[1],
        'FullFit_MAE (kg/m²)': fullfit_metrics[2],
        'FullFit_RPD': fullfit_metrics[3],
        '样本数': len(df),
        '自变量数': len(indep_cols)
    }]).to_csv(os.path.join(config['output_path'], f'Region_{target_region}_GWR_FULLFIT_局部解释诊断.csv'),
               index=False, encoding='utf_8_sig')

    # 局部R²、局部系数和图件
    df['局部R²'] = gwr_full_results.localR2.ravel()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
        crs='EPSG:4326'
    )

    # 局部R²空间分布图
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(gdf['longitude'], gdf['latitude'], c=gdf['局部R²'], cmap='RdYlGn', s=50, alpha=0.8, edgecolors='k')
    plt.colorbar(scatter, label='Local R²')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title(f'GWR Local R² Spatial Distribution: {target_region} ({region_name})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], f'Region_{target_region}_GWR_FULLFIT_local_R2.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 全样本拟合散点图：单独保存，避免和测试集图混淆
    plot_pred_vs_obs(
        y_all_raw.ravel(), y_all_fit, fullfit_metrics,
        os.path.join(config['output_path'], f'Region_{target_region}_GWR_FULLFIT_pred_vs_obs_仅解释用.png'),
    )

    # 特征相关性热力图（保留原有功能）
    core_cols = [
        'latitude', 'longitude', 'SOCD_kg_m2', 'MAT', 'MAP', 'PET', 'AI',
        'DEM', 'TWI', 'slope_deg', 'plan_cur', 'pH', 'Sand', 'Silt', 'Clay',
        'True_Soil_Depth', 'Vegetation_Factor1', 'Vegetation_Factor2'
    ]
    core_cols = [col for col in core_cols if col in df.columns]
    corr_df = df[core_cols].corr()

    plt.figure(figsize=(14, 12))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})
    sns.heatmap(
        corr_df, annot=True, fmt='.2f', cmap='RdBu_r',
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )
    plt.title(f'Correlation Heatmap (Region={target_region} - {region_name})',
              fontsize=14, pad=20, fontfamily='Times New Roman')
    plt.xticks(rotation=45, ha='right', fontfamily='Times New Roman')
    plt.yticks(rotation=0, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], f'Region_{target_region}_GWR_feature_corr_heatmap.png'),
                dpi=300)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 原始局部系数贡献重要性图（解释用，不是主预测精度）
    feat_importance = []
    for feat in indep_cols:
        coefs = gwr_full_results.params[:, indep_cols.index(feat) + 1].ravel()
        std_coef = coefs * (df[feat].std() / np.std(y_all.ravel(), ddof=1))
        feat_importance.append({'Feature_Name': feat, 'Importance': np.mean(np.abs(std_coef))})

    feat_imp_df = pd.DataFrame(feat_importance).sort_values('Importance', ascending=False).head(15)
    feat_imp_df.to_csv(os.path.join(config['output_path'], f'Region_{target_region}_GWR_FULLFIT_local_coef_importance.csv'),
                       index=False, encoding='utf_8_sig')

    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})
    colors = ['#1f77b4' if i < 3 else '#6baed6' for i in range(len(feat_imp_df))]
    bars = sns.barplot(x='Importance', y='Feature_Name', data=feat_imp_df, palette=colors)
    for bar in bars.patches:
        width = bar.get_width()
        bars.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                  f'{width:.3f}', va='center', fontsize=10, fontfamily='Times New Roman')
    plt.xlabel('Feature Importance', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    plt.xticks(fontsize=11, fontfamily='Times New Roman')
    plt.yticks(fontsize=11, fontfamily='Times New Roman')
    if feat_imp_df['Importance'].max() > 0:
        plt.xlim(0, feat_imp_df['Importance'].max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], f'Region_{target_region}_GWR_FULLFIT_feature_importance.png'),
                dpi=300)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 全量样本结果表（解释用）
    all_result = pd.DataFrame({
        '纬度': df['latitude'].values,
        '经度': df['longitude'].values,
        'Region': target_region,
        'Region_Name': region_name,
        '实测SOCD(kg/m2)': y_all_raw.ravel(),
        '预测SOCD(kg/m2)': y_all_fit,
        '误差(kg/m2)': y_all_fit - y_all_raw.ravel(),
        '局部R²': df['局部R²'].values,
        '数据集类型': 'All_FULLFIT'
    })
    all_result.to_csv(os.path.join(config['output_path'], f'Region_{target_region}_GWR_FULLFIT_All_Sample_Results.csv'),
                      index=False, encoding='utf-8-sig')

    # 局部参数表
    params_df = pd.DataFrame(gwr_full_results.params, columns=['Intercept'] + indep_cols)
    aux_cols = ['FID', 'Samples', 'profile_id', 'latitude', 'longitude', 'SOCD_kg_m2', '土地覆盖类型', 'Region', 'CLCD']
    aux_cols = [col for col in aux_cols if col in df.columns]
    params_df = pd.concat([params_df, df[aux_cols].reset_index(drop=True)], axis=1)
    params_df['Predicted_SOCD_FULLFIT'] = y_all_fit
    params_df['Local_R²'] = df['局部R²'].values

    feat_std = df[indep_cols].std().values
    socd_std = np.std(y_all.ravel(), ddof=1)
    for i, feat in enumerate(indep_cols):
        std_coef = gwr_full_results.params[:, i + 1] * (feat_std[i] / socd_std)
        params_df[f'{feat}_Standardized_Coefficient'] = std_coef
    params_df.to_csv(os.path.join(config['output_path'], 'GWR_FULLFIT_局部参数表.csv'),
                     index=False, encoding='utf_8_sig')

    # 特征贡献统计表
    contribution_stats = []
    for feat in indep_cols:
        coefs = gwr_full_results.params[:, indep_cols.index(feat) + 1].ravel()
        se = gwr_full_results.bse[:, indep_cols.index(feat) + 1].ravel()
        with np.errstate(divide='ignore', invalid='ignore'):
            t_val = np.abs(coefs / se)
        contribution_stats.append({
            'Feature_Name': feat,
            'Local_Coefficient_Mean': round(np.mean(coefs), 4),
            'Significant_Sample_Ratio(%)': round(np.mean(t_val > 1.96) * 100, 2),
            'Positive_Coefficient_Ratio(%)': round(np.mean(coefs > 0) * 100, 2),
            'Negative_Coefficient_Ratio(%)': round(np.mean(coefs < 0) * 100, 2)
        })
    pd.DataFrame(contribution_stats).to_csv(os.path.join(config['output_path'], 'GWR_FULLFIT_特征贡献统计表.csv'),
                                            index=False, encoding='utf_8_sig')

    # 全特征局部系数空间图
    print('\n🔍 绘制分区全样本 GWR 局部系数空间分布图...')
    for feat in indep_cols:
        coefs = gwr_full_results.params[:, indep_cols.index(feat) + 1].ravel()
        plt.figure(figsize=(12, 10))
        plt.scatter(df['longitude'], df['latitude'], c=coefs, cmap='RdBu_r', s=50, alpha=0.8, edgecolors='k')
        plt.colorbar(label=f'{feat} Local Coefficient')
        plt.xlabel('Longitude (°)')
        plt.ylabel('Latitude (°)')
        plt.title(f'GWR Local Coefficient Distribution: {feat} | {target_region} ({region_name})')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_path'], f'Region_{target_region}_GWR_FULLFIT_{feat}_local_coefficient.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 分区SOCD制图数据（解释用，与旧功能兼容）
    mapping_data = pd.DataFrame({
        'Longitude': df['longitude'].values.round(6),
        'Latitude': df['latitude'].values.round(6),
        'Region_Type': target_region,
        'Region_Name': region_name,
        'Predicted_SOCD_kg_m2': y_all_fit.round(4),
        'Model_Name': 'GWR',
        'Local_R²': df['局部R²'].values.round(4)
    })
    mapping_path = os.path.join(config['output_path'], f'GWR_Region_{target_region}_FULLFIT_SOCD_Mapping_Data.csv')
    mapping_data.to_csv(mapping_path, index=False, encoding='utf-8-sig')

    # 汇总返回
    result_summary = {
        'Region': target_region,
        'Region_Name': region_name,
        'Target_Transform': target_transform,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
        'Test_RPD': test_rpd,
        'FullFit_R2': fullfit_metrics[0],
        'FullFit_RMSE': fullfit_metrics[1],
        'FullFit_MAE': fullfit_metrics[2],
        'FullFit_RPD': fullfit_metrics[3],
        'Train_BW': optimal_bw,
        'FullFit_BW': optimal_bw_all,
        'Train_N': len(df_train),
        'Val_N': len(df_val),
        'Test_N': len(df_test),
        'All_N': len(df),
        'Feature_N': len(indep_cols),
        'Test_Pred_Method': test_pred_method
    }

    print('\n🎉 分区 GWR 空间独立测试集评价版运行完成！')
    print(f'📁 输出目录：{config["output_path"]}')
    print(f'✅ 四模型分区比较请使用：Region_{target_region}_GWR_空间独立测试集_四模型对比指标.csv')
    print(f'✅ GWR测试集核心指标：R²={test_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | RPD={test_rpd:.4f}')
    print('⚠️ GWR_FULLFIT_* 文件只用于局部R²和局部系数解释，不用于四模型预测精度主表。')

    return result_summary


# ======================== 3. 单分区和批量运行 ========================
def run_one_region(region=None):
    if region is not None:
        CONFIG['target_region'] = region
        CONFIG['output_path'] = os.path.join(
            CONFIG['output_base'],
            f"Region_{region}_{REGION_EN_MAP.get(region, region)}"
        )

    ensure_output_dir(CONFIG)

    print("📥 读取CSV格式原始数据...")
    df_raw = pd.read_csv(CONFIG['data_path'], encoding='gbk')

    df_prep, indep_cols, scaler = preprocess_data(df_raw, CONFIG)
    result_summary = train_and_evaluate_gwr_region(df_prep, indep_cols, CONFIG)

    print(f"\n🎉 单一地理分区（Region={CONFIG['target_region']}）GWR模型训练完成！")
    return result_summary


def batch_run_all_regions():
    print("🚀 开始批量运行所有6个地理分区 GWR...")
    all_summaries = []

    for region in ALL_REGIONS:
        print(f"\n{'=' * 70}")
        print(f"正在处理分区：{region} - {REGION_EN_MAP[region]}")
        print('=' * 70)

        try:
            summary = run_one_region(region)
            all_summaries.append(summary)
        except Exception as e:
            print(f"❌ Region={region} 运行失败：{repr(e)}")
            all_summaries.append({
                'Region': region,
                'Region_Name': REGION_EN_MAP.get(region, region),
                'Error': repr(e)
            })

    summary_df = pd.DataFrame(all_summaries)
    os.makedirs(CONFIG['output_base'], exist_ok=True)
    summary_path = os.path.join(CONFIG['output_base'], 'GWR_按地理分区_空间独立测试集_汇总.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf_8_sig')

    print("\n✅ 所有地理分区处理完成！")
    print(f"📊 汇总表已保存：{summary_path}")
    return summary_df


if __name__ == "__main__":
    # 单分区运行：
    # run_one_region('dongbei')

    # 批量运行全部6个地理分区：
    batch_run_all_regions()
