# ======================== 1. 导入必需库（完整无遗漏） ========================
# ======================== 1. 导入必需库（完整无遗漏） ========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
from scipy import stats  # 修复：完整导入stats模块（关键）
from scipy.stats import pearsonr, gaussian_kde  # 新增：gaussian_kde用于散点图密度排序
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import (
    train_test_split, GridSearchCV, KFold
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import geopandas as gpd  # 新增：用于全量数据制图
from shapely.geometry import Point  # 新增：用于全量数据制图

# 全局设置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
np.random.seed(42)

# 多核并行优化
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

# 映射字典（与RF/GWR完全一致）
CLCD_EN_MAP = {
    1: 'Cropland',
    2: 'Forest',
    3: 'Shrubland',
    4: 'Grassland',
    5: 'Water',
    6: 'Snow/Ice',
    7: 'Barren land',
    8: 'Impervious',
    9: 'Wetland'
}
COL_NAME_MAP = {'真实土层厚度': 'True_Soil_Depth', '植被综合因子1': 'Vegetation_Factor1',
                '植被综合因子2': 'Vegetation_Factor2'}

# ======================== 2. 核心参数配置（适配所有CLCD类型） ========================
CONFIG = {
    # Set data_path to the local training CSV used in the study.
    "data_path": r"PATH_TO_TRAINING_DATA.csv",
    "output_path": os.path.join(os.path.dirname(__file__), "outputs", "svr_gms") + os.sep,
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "svr_params": {
        "kernel": "rbf",
        "max_iter": 100000,
        "verbose": 0
    },
    "param_grid": {
        "C": [50, 100],
        "gamma": ["scale"],
        "epsilon": [0.3]
    }
}


# ======================== 新增：统一格式的置换重要性（与RF/GWR逻辑100%一致） ========================
def calculate_permutation_importance_unified(model, X, y, feature_names, n_repeats=3, random_state=42):
    """SVR统一置换重要性：与RF/GWR定义一致（基准R²-打乱后R²）"""
    # 计算基准精度（R²，与RF/GWR统一）
    y_pred_base = model.predict(X)
    base_r2 = r2_score(y.ravel(), y_pred_base)

    # 初始化重要性数组
    importances = np.zeros((n_repeats, len(feature_names)))
    rng = np.random.RandomState(random_state)

    # 逐特征打乱计算（与RF/GWR完全同逻辑）
    for i, feat in enumerate(feature_names):
        for n in range(n_repeats):
            # 复制并仅打乱当前特征
            X_perm = X.copy()
            if isinstance(X_perm, pd.DataFrame):
                X_perm[feat] = rng.permutation(X_perm[feat].values)
            else:
                X_perm[:, i] = rng.permutation(X_perm[:, i])

            # 计算打乱后的预测精度
            y_pred_perm = model.predict(X_perm)
            perm_r2 = r2_score(y.ravel(), y_pred_perm)

            # 重要性=基准R² - 打乱后R²（与RF/GWR定义一致）
            importances[n, i] = base_r2 - perm_r2

    # 生成结果表（格式与RF/GWR完全对齐）
    perm_imp_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Permutation_Importance': importances.mean(axis=0),
        'Importance_Std': importances.std(axis=0)
    }).sort_values('Permutation_Importance', ascending=False)

    return perm_imp_df, base_r2


# ======================== 3. 数据预处理 ========================
def preprocess_data(df_raw, config):
    # 中文列名→英文（与RF/GWR一致）
    df_raw = df_raw.rename(columns={k: v for k, v in COL_NAME_MAP.items() if k in df_raw.columns})

    # 筛选核心字段
    candidate_features = [
        'True_Soil_Depth', 'DEM', 'TWI', 'plan_cur', 'slope_deg', 'slope_aspect_deg',
        'MAT', 'MAP', 'PET', 'AI', 'pH', 'Sand', 'Silt', 'Clay', 'Albedo',
        'BGREEN', 'BNIR', 'BRED', 'NDVI', 'SAVI', 'OSAVI', 'EVI', 'LSWI', 'MSAVI', 'NDWI'
    ]
    core_columns = candidate_features + ['latitude', 'longitude', 'SOCD__kg_m', 'CLCD', 'Region', 'Year']
    core_columns = [col for col in core_columns if col in df_raw.columns]
    df = df_raw[core_columns].copy()

    # ==============================================
    # 【已删除】CLCD 地类筛选，直接使用全部样本
    # ==============================================
    initial_count = len(df)
    print(f"📌 全样本：{initial_count}条")

    # 目标变量校准
    df.rename(columns={'SOCD__kg_m': 'SOCD_kg_m2'}, inplace=True)
    socd_mean = df['SOCD_kg_m2'].mean()
    socd_std = df['SOCD_kg_m2'].std()
    df = df[(df['SOCD_kg_m2'] >= 0) & (df['SOCD_kg_m2'] <= socd_mean + 2 * socd_std)].copy()
    print(f"📏 目标变量校准：SOCD均值={socd_mean:.4f}，标准差={socd_std:.4f}，保留0~{socd_mean + 2 * socd_std:.4f}范围样本")

    # 缺失值处理
    df = df.dropna(subset=['SOCD_kg_m2'])
    missing_stats = df.isnull().sum()[df.isnull().sum() > 0]
    if len(missing_stats) > 0:
        print(f"📊 缺失值填充前：{missing_stats}")
    else:
        print(f"✅ 无缺失值，无需填充")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col != 'SOCD_kg_m2':
            df[col] = df[col].fillna(df[col].median())

    # 植被PCA优化
    veg_cols = ['BGREEN', 'BNIR', 'BRED', 'NDVI', 'SAVI', 'OSAVI', 'EVI', 'LSWI', 'MSAVI', 'NDWI']
    veg_cols = [col for col in veg_cols if col in df.columns]
    if veg_cols:
        pca = PCA(n_components=2, random_state=config['random_state'])
        pca_result = pca.fit_transform(df[veg_cols])
        df['Vegetation_Factor1'] = pca_result[:, 0]
        df['Vegetation_Factor2'] = pca_result[:, 1]
        print(
            f"🌿 植被PCA解释方差比：PC1={pca.explained_variance_ratio_[0]:.4f}，PC2={pca.explained_variance_ratio_[1]:.4f}，总计={sum(pca.explained_variance_ratio_):.4f}")
        candidate_features = [col for col in candidate_features if col not in veg_cols] + ['Vegetation_Factor1',
                                                                                           'Vegetation_Factor2']

    # VIF筛选
    def vif_filter(df, features, max_vif=20):
        vif_cols = features.copy()
        while True:
            temp_df = df[vif_cols].fillna(df[vif_cols].median())
            temp_df = sm.add_constant(temp_df)
            vif_list = []
            for i in range(1, temp_df.shape[1]):
                vif = variance_inflation_factor(temp_df.values, i)
                vif_list.append({'特征': temp_df.columns[i], 'VIF': round(vif, 2)})
            if not vif_list:
                break
            vif_df = pd.DataFrame(vif_list).sort_values('VIF', ascending=False)
            max_vif_row = vif_df.iloc[0]
            if max_vif_row['VIF'] <= max_vif or len(vif_cols) <= 5:
                break
            drop_col = max_vif_row['特征']
            vif_cols.remove(drop_col)
            print(f"⚠️ 剔除高共线性特征：{drop_col}（VIF={max_vif_row['VIF']}）")
        final_vif_df = pd.DataFrame([
            {'特征': col, 'VIF': round(
                variance_inflation_factor(sm.add_constant(df[vif_cols]).fillna(df[vif_cols].median()).values, i + 1),
                2)}
            for i, col in enumerate(vif_cols)
        ]).sort_values('VIF', ascending=False)
        print(f"📋 最终特征VIF值：\n{final_vif_df}")
        return vif_cols

    final_features = vif_filter(df, candidate_features)
    print(f"✅ 最终特征数：{len(final_features)} | 特征列表：{final_features}")

    # 标准化
    scaler = StandardScaler()
    df[final_features] = scaler.fit_transform(df[final_features])
    std_stats = pd.DataFrame({
        '特征': final_features,
        '均值': df[final_features].mean().round(4),
        '标准差': df[final_features].std().round(4)
    })
    print(f"📊 特征标准化后统计量：\n{std_stats.head(10)}")

    # 区域哑变量
    #final_features += list(region_dummies.columns)

    # 与 RF/GWR 保持一致：删除 Region 和 CLCD，不参与建模
    df = df.drop(columns=['Region', 'CLCD'], errors='ignore')

    return df, scaler, final_features


# ======================== 4. 空间分层抽样 ========================
def spatial_split(df, config):
    df['lat_grid'] = (df['latitude'] // 0.09).astype(int)
    df['lon_grid'] = (df['longitude'] // 0.09).astype(int)
    df['grid_id'] = df['lat_grid'].astype(str) + '_' + df['lon_grid'].astype(str)
    total_grids = len(df['grid_id'].unique())
    print(f"🌍 空间网格总数：{total_grids}")

    # 拆分数据
    grid_ids = df['grid_id'].unique()
    train_val_grids, test_grids = train_test_split(grid_ids, test_size=config['test_size'],
                                                   random_state=config['random_state'])
    df_test = df[df['grid_id'].isin(test_grids)].copy()
    df_train_val = df[df['grid_id'].isin(train_val_grids)].copy()

    train_grids, val_grids = train_test_split(train_val_grids, test_size=config['val_size'] / (1 - config['test_size']),
                                              random_state=config['random_state'])
    df_train = df[df['grid_id'].isin(train_grids)].copy()
    df_val = df[df['grid_id'].isin(val_grids)].copy()

    # 数据集统计
    dataset_stats = pd.DataFrame({
        '数据集': ['训练集', '验证集', '测试集'],
        '样本数': [len(df_train), len(df_val), len(df_test)],
        '占比': [f"{len(df_train) / len(df) * 100:.1f}%", f"{len(df_val) / len(df) * 100:.1f}%",
                 f"{len(df_test) / len(df) * 100:.1f}%"],
        'SOCD均值': [df_train['SOCD_kg_m2'].mean().round(4), df_val['SOCD_kg_m2'].mean().round(4),
                     df_test['SOCD_kg_m2'].mean().round(4)],
        'SOCD标准差': [df_train['SOCD_kg_m2'].std().round(4), df_val['SOCD_kg_m2'].std().round(4),
                       df_test['SOCD_kg_m2'].std().round(4)]
    })
    print(f"\n📈 各数据集统计：\n{dataset_stats}")

    # 提取特征标签
    feature_cols = [col for col in df.columns if
                    col not in ['latitude', 'longitude', 'SOCD_kg_m2', 'lat_grid', 'lon_grid', 'grid_id', 'CLCD',
                                'Region', 'Year']]
    X_train, y_train = df_train[feature_cols], df_train['SOCD_kg_m2']
    X_val, y_val = df_val[feature_cols], df_val['SOCD_kg_m2']
    X_test, y_test = df_test[feature_cols], df_test['SOCD_kg_m2']

    print(f"\n📊 数据拆分：训练集{len(X_train)} | 验证集{len(X_val)} | 测试集{len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df


# ======================== 5. 模型训练 ========================
def train_svr(X_train, y_train, X_val, y_val, config):
    print(f"\n🚀 开始训练SVR模型（全国全样本）...")
    start_time = time.time()

    kf = KFold(n_splits=5, shuffle=True, random_state=config['random_state'])
    grid_search = GridSearchCV(
        estimator=SVR(**config['svr_params']),
        param_grid=config['param_grid'],
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=20,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    # 参数搜索结果
    cv_results = pd.DataFrame(grid_search.cv_results_)[
        ['param_C', 'param_gamma', 'param_epsilon', 'mean_test_score', 'rank_test_score']]
    cv_results['mean_test_score'] = -cv_results['mean_test_score']
    cv_results['mean_test_score'] = cv_results['mean_test_score'].round(4)
    print(f"\n🔍 参数搜索结果：\n{cv_results}")

    best_svr = grid_search.best_estimator_
    print(f"✅ 最优参数：{grid_search.best_params_}，最优CV得分（MSE）：{grid_search.best_score_ * -1:.4f}")

    # 验证集评估
    y_val_pred = best_svr.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_corr, val_p = pearsonr(y_val, y_val_pred)

    print(f"⏱️ 训练耗时：{time.time() - start_time:.2f}秒")
    print(
        f"📈 验证集：R²={val_r2:.4f} | RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | Pearson相关系数={val_corr:.4f}（p值={val_p:.4e}）")
    return best_svr, grid_search.best_params_


# ======================== 6. 模型评估 ========================
def evaluate_svr(svr_model, best_params, X_test, y_test, feature_cols, config, df_prep, df_all):
    y_test_pred = svr_model.predict(X_test)

    # 核心指标
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_corr, test_p = pearsonr(y_test, y_test_pred)
    test_relative_error = (np.abs(y_test - y_test_pred) / y_test).mean() * 100
    test_rpd = round(np.std(y_test, ddof=1) / test_rmse, 4)

    test_metrics = {
        'R²': round(test_r2, 4),
        'RMSE (kg/m²)': round(test_rmse, 4),
        'MAE (kg/m²)': round(test_mae, 4),
        'Pearson相关系数': round(test_corr, 4),
        'Pearson p值': f"{test_p:.4e}",
        '平均相对误差 (%)': round(test_relative_error, 4),
        'RPD': test_rpd,  # 新增：RPD指标
        '数据类型': '全国全样本'
    }

    print(f"\n📊 测试集最终结果（全国全样本）：")
    for k, v in test_metrics.items():
        print(f"   {k}：{v}")

    # 散点图 可视化1：预测vs实测（优化后：与RF格式完全统一）
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    # 密度计算+排序（完全保留原代码）
    slope, intercept, r_val, _, _ = stats.linregress(y_test, y_test_pred)
    xy = np.vstack([y_test, y_test_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_test_sorted, y_test_pred_sorted, z_sorted = y_test.iloc[idx], y_test_pred[idx], z[idx]
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())

    # 散点绘制（完全保留原参数）
    sns.scatterplot(
        x=y_test_sorted, y=y_test_pred_sorted,
        c=z_sorted, cmap='viridis',
        alpha=0.6, s=40, edgecolor='none'
    )

    # 参考线（完全保留原参数）
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
    plt.plot(y_test, slope * y_test + intercept, 'r-', lw=2)

    # 坐标轴标注【对齐RF：正体+上标²，无斜体】
    plt.xlabel('Lab-measured SOCD (kg/m²)', fontsize=17)
    plt.ylabel('Predicted SOCD (kg/m²)', fontsize=17)

    # 文本位置（完全保留原逻辑）
    ax = plt.gca()
    x_axis_min, x_axis_max = ax.get_xlim()
    y_axis_min, y_axis_max = ax.get_ylim()
    x_pos = x_axis_min + (x_axis_max - x_axis_min) * 0.05
    y_pos = y_axis_min + (y_axis_max - y_axis_min) * 0.95

    # 指标文本【对齐RF：R²上标+单位+行间距1.5】
    plt.text(
        x=x_pos, y=y_pos,
        s=f'R²={test_r2:.4f}\nRMSE={test_rmse:.4f} (kg/m²)\nMAE={test_mae:.4f} (kg/m²)\nRPD={test_rpd:.4f}',
        fontsize=18, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontfamily='Times New Roman',
        ha='left', va='top', linespacing=1.5
    )

    # 刻度+布局（完全对齐RF）
    plt.xticks(fontsize=16, fontfamily='Times New Roman')
    plt.yticks(fontsize=16, fontfamily='Times New Roman')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存（保留原路径/命名）
    plt.savefig(f"{config['output_path']}全国全样本_SVR_pred_vs_obs.png", dpi=400)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 可视化2：误差分布（保留原有逻辑，不修改）
    plt.figure(figsize=(12, 6))
    errors = y_test - y_test_pred
    sns.histplot(errors, bins=30, kde=True, color='lightcoral', edgecolor='black', alpha=0.7, label='误差分布')
    plt.axvline(x=0, color='red', linestyle='--', lw=2, label='0误差线')
    plt.axvline(x=errors.mean(), color='blue', linestyle='-', lw=2, label=f'误差均值={errors.mean():.4f}')
    plt.axvline(x=errors.median(), color='green', linestyle=':', lw=2, label=f'误差中位数={errors.median():.4f}')
    plt.text(
        x=errors.max() * 0.7, y=plt.gca().get_ylim()[1] * 0.8,
        s=f'RMSE={test_rmse:.4f}\nMAE={test_mae:.4f}\n平均相对误差={test_relative_error:.2f}%\n误差标准差={errors.std():.4f}',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.xlabel('预测误差 (kg/m²)', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.title(f'全国全样本 SVR预测误差分布', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config['output_path']}全国全样本_SVR_error_hist.png", dpi=300)
    plt.close()

    # 可视化3：特征重要性（保留原有逻辑，不修改）
    print(f"\n🔢 计算RBF核特征排列重要性（前15个特征）...")
    perm_imp = permutation_importance(
        svr_model, X_test, y_test,
        n_repeats=10, random_state=config['random_state'], n_jobs=20
    )
    feat_imp = pd.DataFrame({
        '特征名称': feature_cols,
        '重要性': perm_imp.importances_mean,
        '重要性标准差': perm_imp.importances_std
    }).sort_values('重要性', ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='重要性', y='特征名称', data=feat_imp, xerr=feat_imp['重要性标准差'], color='skyblue', capsize=0.1)
    plt.xlabel('特征重要性', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.title(f'全国全样本 SVR前15特征重要性', fontsize=14)
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{config['output_path']}全国全样本_SVR_feat_imp.png", dpi=300)
    plt.close()

    # 新增：统一格式的置换重要性图（与RF/GWR样式完全一致）
    print(f"\n🔍 计算统一格式置换重要性（跨模型对比用）...")
    # 抽样计算（避免样本过多耗时，与GWR逻辑一致）
    sample_size = min(500, len(X_test))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_test_sample = X_test.iloc[sample_idx] if isinstance(X_test, pd.DataFrame) else X_test[sample_idx]
    y_test_sample = y_test.iloc[sample_idx] if isinstance(y_test, pd.Series) else y_test[sample_idx]

    # 调用统一置换重要性函数
    perm_imp_df, base_r2 = calculate_permutation_importance_unified(
        svr_model, X_test_sample, y_test_sample, feature_cols,
        n_repeats=3, random_state=config['random_state']
    )

    # 保存置换重要性结果（文件名与RF/GWR一致）
    perm_imp_df.to_csv(f"{config['output_path']}全国全样本_SVR_permutation_importance.csv",
                       index=False, encoding='utf_8_sig')
    print(f"✅ 统一置换重要性结果已保存（与RF/GWR格式一致）")

    # 绘制统一格式置换重要性图（复刻RF/GWR样式）
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    # 过滤低贡献特征（与RF/GWR一致：重要性>0.0005）
    perm_imp_filtered = perm_imp_df[perm_imp_df['Permutation_Importance'] > 0.0005]
    if len(perm_imp_filtered) > 15:
        perm_imp_final = perm_imp_filtered.head(15)
    else:
        perm_imp_final = perm_imp_filtered

    # 颜色、样式与RF/GWR完全一致
    colors = ['#1f77b4' if i < 3 else '#6baed6' for i in range(len(perm_imp_final))]
    bars = sns.barplot(x='Permutation_Importance', y='Feature_Name', data=perm_imp_final, palette=colors)

    # 4位小数标注（与RF/GWR一致）
    for bar in bars.patches:
        width = bar.get_width()
        bars.text(
            width + 0.0001, bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}', va='center', fontsize=10, fontfamily='Times New Roman'
        )

    # 坐标轴与RF/GWR一致
    plt.xlabel('Permutation Importance (ΔR²)', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    plt.xticks(fontsize=11, fontfamily='Times New Roman')
    plt.yticks(fontsize=11, fontfamily='Times New Roman')
    plt.xlim(0, perm_imp_final['Permutation_Importance'].max() * 1.15)
    plt.tight_layout()

    # 保存（文件名与RF/GWR一致）
    plt.savefig(f"{config['output_path']}全国全样本_SVR_permutation_importance.png", dpi=300)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(f"✅ 统一置换重要性图已保存（与RF/GWR样式一致）")

    # 新增：全量数据建模（适配全国SOCD制图，格式与GWR一致）
    print("\n🔍 生成全国SOCD制图数据...")
    # 提取全量数据特征（与建模特征一致）
    X_all = df_prep[feature_cols].values
    # 全量预测
    y_all_pred = svr_model.predict(X_all)

    # 构建制图数据（列名、格式与GWR完全一致）
    national_mapping_data = pd.DataFrame({
        'Longitude': df_prep['longitude'].values.round(6),
        'Latitude': df_prep['latitude'].values.round(6),
        'CLCD_Type': 'All',
        'Land_Cover': '全国全样本',
        'Predicted_SOCD_kg_m2': y_all_pred.round(4),
        'Model_Name': 'SVR'
    })

    # 保存制图数据（文件名与GWR一致）
    mapping_path = f"{config['output_path']}SVR_全国全样本_National_SOCD_Mapping_Data.csv"
    national_mapping_data.to_csv(mapping_path, index=False, encoding='utf-8-sig')
    print(f"✅ 全国SOCD制图数据已保存：{mapping_path}")

    # 新增：全量样本结果表（与GWR格式一致）
    all_result = pd.DataFrame({
        '纬度': df_prep['latitude'].values,
        '经度': df_prep['longitude'].values,
        '实测SOCD(kg/m2)': df_prep['SOCD_kg_m2'].values,
        '预测SOCD(kg/m2)': y_all_pred,
        '误差(kg/m2)': y_all_pred - df_prep['SOCD_kg_m2'].values,
        '数据集类型': 'All'
    })
    all_result.to_csv(f"{config['output_path']}全国全样本_SVR_All_Sample_Results.csv", index=False,
                      encoding='utf-8-sig')
    print(f"✅ 全量样本结果表已保存（与GWR格式一致）")

    # 保存文件（保留原有逻辑，不修改）
    # 1. 模型指标
    metrics_df = pd.DataFrame(test_metrics, index=[0])
    metrics_df.to_csv(f"{config['output_path']}全国全样本_SVR_metrics.csv", index=False,
                      encoding='utf_8_sig')

    # 2. 特征重要性
    feat_imp.to_csv(f"{config['output_path']}全国全样本_SVR_feat_imp.csv", index=False,
                    encoding='utf_8_sig')

    # 3. 测试集预测结果
    test_result_df = pd.DataFrame({
        '纬度': df_all.loc[y_test.index, 'latitude'].values,
        '经度': df_all.loc[y_test.index, 'longitude'].values,
        '实测SOCD': y_test.values,
        '预测SOCD': y_test_pred,
        '误差': errors,
        '相对误差(%)': (errors / y_test.values) * 100
    })
    test_result_df.to_csv(f"{config['output_path']}全国全样本_SVR_test_predictions.csv", index=False,
                          encoding='utf_8_sig')

    # 4. 模型配置（修复grid_search引用问题）
    config_df = pd.DataFrame({
        '参数名称': ['数据类型', '测试集比例', '验证集比例', '随机种子', '核函数', '最优C', '最优gamma', '最优epsilon'],
        '参数值': [
            '全国全样本', config['test_size'], config['val_size'], config['random_state'],
            config['svr_params']['kernel'],
            best_params.get('C', 'N/A'),
            best_params.get('gamma', 'N/A'),
            best_params.get('epsilon', 'N/A')
        ]
    })
    config_df.to_csv(f"{config['output_path']}全国全样本_SVR_config.csv", index=False,
                     encoding='utf_8_sig')

    return test_metrics


# ======================== 7. 主函数 ========================
def main():
    # 创建输出目录
    if not os.path.exists(CONFIG['output_path']):
        os.makedirs(CONFIG['output_path'])
        print(f"📁 创建输出目录：{CONFIG['output_path']}")

    # 读取数据
    print("📥 读取原始数据...")
    df_raw = pd.read_csv(CONFIG['data_path'], encoding='gbk')
    print(f"📊 原始数据形状：{df_raw.shape}，字段数：{len(df_raw.columns)}")
    print(f"🔍 原始数据前5行字段：{df_raw.columns[:5].tolist()}")

    # 预处理
    df_prep, scaler, feat_cols = preprocess_data(df_raw, CONFIG)

    # 抽样
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df_all = spatial_split(df_prep, CONFIG)

    # 训练
    svr_model, best_params = train_svr(X_train, y_train, X_val, y_val, CONFIG)

    # 评估
    test_metrics = evaluate_svr(svr_model, best_params, X_test, y_test, feature_cols, CONFIG, df_prep, df_all)

    # 最终输出（更新文件清单，添加新增文件）
    print(f"\n🎉 运行完成！所有结果已保存至：{CONFIG['output_path']}")
    print(f"📋 输出文件清单：")
    print(f"   1. 全国全样本_SVR_metrics.csv（模型指标）")
    print(f"   2. 全国全样本_SVR_feat_imp.csv（特征重要性）")
    print(f"   3. 全国全样本_SVR_test_predictions.csv（测试集预测结果）")
    print(f"   4. 全国全样本_SVR_config.csv（模型配置）")
    print(f"   5. 全国全样本_SVR_pred_vs_obs.png（预测vs实测图，优化后）")
    print(f"   6. 全国全样本_SVR_error_hist.png（误差分布图）")
    print(f"   7. 全国全样本_SVR_feat_imp.png（特征重要性图）")
    print(f"   8. 全国全样本_SVR_permutation_importance.csv（统一置换重要性结果）")
    print(f"   9. 全国全样本_SVR_permutation_importance.png（统一置换重要性图）")
    print(f"  10. SVR_全国全样本_National_SOCD_Mapping_Data.csv（全国SOCD制图数据）")
    print(f"  11. 全国全样本_SVR_All_Sample_Results.csv（全量样本结果表）")
    print(
        f"\n📌 核心指标汇总：R²={test_metrics['R²']} | RMSE={test_metrics['RMSE (kg/m²)']} | MAE={test_metrics['MAE (kg/m²)']} | RPD={test_metrics['RPD']}")


if __name__ == "__main__":
    main()
