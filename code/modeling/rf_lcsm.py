# ======================== 1. 导入必需库 ========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import warnings
import os
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

# 基础字体配置（先重置，避免冲突）
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')

# ======================== 2. 核心参数配置 ========================
# 多核优化
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

# 地类编码→英文映射
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
COL_NAME_MAP = {
    '真实土层厚度': 'True_Soil_Depth',
    '植被综合因子1': 'Vegetation_Factor1',
    '植被综合因子2': 'Vegetation_Factor2'
}

CONFIG = {
    # Set data_path to the local training CSV used in the study.
    "data_path": r"PATH_TO_TRAINING_DATA.csv",
    "output_path": os.path.join(os.path.dirname(__file__), "outputs", "rf_lcsm") + os.sep,
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "target_clcd": 7,
    "rf_params": {
        "n_estimators": 800,
        "max_depth": 20,
        "min_samples_split": 3,
        "oob_score": True,
        "random_state": 42,
        "n_jobs": 20,
        "verbose": 0
    }
}


# ======================== 3. 数据预处理（修复中文列名） ========================
def preprocess_data(df_raw, config):
    # 第一步：映射中文列名为英文（不改动原表格）
    df_raw = df_raw.rename(columns={k: v for k, v in COL_NAME_MAP.items() if k in df_raw.columns})

    # 核心特征筛选（全英文列名）
    core_columns = [
        'latitude', 'longitude', 'SOCD__kg_m',
        'MAT', 'MAP', 'PET', 'AI',
        'DEM', 'TWI', 'slope_deg', 'plan_cur',
        'NDVI', 'LSWI', 'NDWI', 'Albedo',
        'pH', 'Sand', 'Silt', 'Clay', 'True_Soil_Depth',
        'Region', 'CLCD'
    ]
    core_columns = [col for col in core_columns if col in df_raw.columns]
    df = df_raw[core_columns].copy()

    # 筛选单一地类
    if df['CLCD'].dtype == 'object':
        df['CLCD'] = pd.to_numeric(df['CLCD'], errors='coerce')
    initial_sample_count = len(df)
    df = df[df['CLCD'] == config['target_clcd']].copy()
    print(f"   原始样本数：{initial_sample_count} → 筛选后样本数：{len(df)}")
    if len(df) < 200:
        print(f"⚠️ 警告：当前地类样本数仅{len(df)}，可能导致模型过拟合")

    # SOCD校准
    socd_mean = df['SOCD__kg_m'].mean()
    socd_std = df['SOCD__kg_m'].std()
    df = df[(df['SOCD__kg_m'] >= 0) & (df['SOCD__kg_m'] <= socd_mean + 2 * socd_std)]
    df.rename(columns={'SOCD__kg_m': 'SOCD_kg_m2'}, inplace=True)

    # 缺失值处理
    df = df.dropna(subset=['SOCD_kg_m2'])
    fill_cols = [col for col in df.columns if col not in ['latitude', 'longitude', 'SOCD_kg_m2', 'Region', 'CLCD']]
    for col in fill_cols:
        df[col] = df.groupby(['Region'])[col].transform(lambda x: x.fillna(x.median()))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 冗余特征剔除
    feat_cols_to_check = [
        'MAT', 'MAP', 'PET', 'AI',
        'DEM', 'TWI', 'slope_deg', 'plan_cur',
        'NDVI', 'LSWI', 'NDWI', 'Albedo',
        'pH', 'Sand', 'Silt', 'Clay', 'True_Soil_Depth'
    ]
    feat_cols_to_check = [col for col in feat_cols_to_check if col in df.columns]

    if len(feat_cols_to_check) > 1:
        corr_matrix = df[feat_cols_to_check].corr().abs()
        socd_corr = df[feat_cols_to_check].corrwith(df['SOCD_kg_m2']).abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        to_drop = []
        for (col1, col2) in high_corr_pairs:
            to_drop.append(col2 if socd_corr[col1] >= socd_corr[col2] else col1)
        to_drop = list(set(to_drop))
        if to_drop:
            print(f"🔍 自动剔除冗余特征（相关系数>0.9）：{to_drop}")
            df = df.drop(columns=to_drop)
            feat_cols_to_check = [col for col in feat_cols_to_check if col not in to_drop]
        else:
            print("✅ 无高度冗余特征")
    else:
        print("⚠️ 特征数量不足，跳过冗余特征筛选")

    # 植被指数PCA
    veg_cols = ['NDVI', 'LSWI', 'NDWI']
    veg_cols = [col for col in veg_cols if col in df.columns]
    if veg_cols:
        for col in veg_cols:
            df[col] = df.groupby(['Region'])[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(df[veg_cols])
        df['Vegetation_Factor1'] = pca_result[:, 0]
        df['Vegetation_Factor2'] = pca_result[:, 1]
        print(f"🌿 植被指数PCA总解释方差比：{sum(pca.explained_variance_ratio_):.4f}")
        df = df.drop(columns=veg_cols)
        feat_cols_to_check = [col for col in feat_cols_to_check if col not in veg_cols] + ['Vegetation_Factor1',
                                                                                           'Vegetation_Factor2']

    # 特征标准化
    scaler = StandardScaler()
    df[feat_cols_to_check] = scaler.fit_transform(df[feat_cols_to_check])

    # 生成Region虚拟变量
    df = df.drop(['Region', 'CLCD'], axis=1)

    final_feat_cols = feat_cols_to_check
    print(f"✅ 数据预处理完成！可用样本数：{len(df)}")
    #print(f"✅ 最终保留特征数：{len(final_feat_cols)}（数值特征{len(feat_cols_to_check)} + Region虚拟变量{len(region_dummies.columns)}）")
    print(f"✅ 最终保留特征数：{len(final_feat_cols)}（纯环境驱动特征）")

    return df, scaler, final_feat_cols


# ======================== 4. 空间分层抽样 ========================
def spatial_split(df, config):
    df['lat_grid'] = (df['latitude'] // 0.09).astype(int)
    df['lon_grid'] = (df['longitude'] // 0.09).astype(int)
    df['grid_id'] = df['lat_grid'].astype(str) + '_' + df['lon_grid'].astype(str)

    grid_ids = df['grid_id'].unique()
    train_val_grids, test_grids = train_test_split(
        grid_ids, test_size=config['test_size'], random_state=config['random_state']
    )
    df_test = df[df['grid_id'].isin(test_grids)].copy()
    df_train_val = df[df['grid_id'].isin(train_val_grids)].copy()

    train_grids, val_grids = train_test_split(
        train_val_grids, test_size=config['val_size'] / (1 - config['test_size']), random_state=config['random_state']
    )
    df_train = df[df['grid_id'].isin(train_grids)].copy()
    df_val = df[df['grid_id'].isin(val_grids)].copy()

    feature_cols = [col for col in df.columns if
                    col not in ['latitude', 'longitude', 'SOCD_kg_m2', 'lat_grid', 'lon_grid', 'grid_id']]
    X_train, y_train = df_train[feature_cols], df_train['SOCD_kg_m2']
    X_val, y_val = df_val[feature_cols], df_val['SOCD_kg_m2']
    X_test, y_test = df_test[feature_cols], df_test['SOCD_kg_m2']

    print(f"\n📊 数据集拆分完成（单一地类CLCD={config['target_clcd']}）：")
    print(f"   训练集：{len(X_train)} 样本 | 验证集：{len(X_val)} 样本 | 测试集：{len(X_test)} 样本")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df_train, df_val, df_test


# ======================== 5. RF模型训练 ========================
def train_rf(X_train, y_train, X_val, y_val, config):
    rf = RandomForestRegressor(**config['rf_params'])
    rf.fit(X_train, y_train)

    y_val_pred = rf.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    print(f"✅ RF模型训练完成！")
    print(f"   验证集 R²：{val_r2:.4f}")
    print(f"   验证集 RMSE：{val_rmse:.4f} kg/m²")
    print(f"   验证集 MAE：{val_mae:.4f} kg/m²")
    print(f"   袋外得分（OOB）：{rf.oob_score_:.4f}")
    return rf


# ======================== 新增：置换重要性计算函数（仅新增，不修改原有） ========================
def calculate_permutation_importance(model, X, y, feature_names, n_repeats=10, random_state=42):
    """计算置换重要性（跨模型统一对比用）"""
    # 计算基准精度
    y_pred_base = model.predict(X)
    base_r2 = r2_score(y, y_pred_base)

    # 初始化重要性数组
    importances = np.zeros((n_repeats, len(feature_names)))
    rng = np.random.RandomState(random_state)

    # 逐特征打乱计算
    for i, feat in enumerate(feature_names):
        for n in range(n_repeats):
            X_perm = X.copy()
            if isinstance(X_perm, pd.DataFrame):
                X_perm[feat] = rng.permutation(X_perm[feat].values)
            else:
                X_perm[:, i] = rng.permutation(X_perm[:, i])

            # 计算打乱后的精度
            y_pred_perm = model.predict(X_perm)
            perm_r2 = r2_score(y, y_pred_perm)

            # 重要性=基准精度-打乱后精度
            importances[n, i] = base_r2 - perm_r2

    # 生成结果表
    perm_imp_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Permutation_Importance': importances.mean(axis=0),
        'Importance_Std': importances.std(axis=0)
    }).sort_values('Permutation_Importance', ascending=False)

    return perm_imp_df, base_r2


# ======================== 6. 模型评估（修复字体+乱码） ========================
def evaluate_rf(rf, X_test, y_test, feature_cols, config, df_preprocessed, df_train, df_val, df_test):
    y_test_pred = rf.predict(X_test)
    X_train_feat = df_train[[col for col in df_train.columns if
                             col not in ['latitude', 'longitude', 'SOCD_kg_m2', 'lat_grid', 'lon_grid', 'grid_id']]]
    X_val_feat = df_val[[col for col in df_val.columns if
                         col not in ['latitude', 'longitude', 'SOCD_kg_m2', 'lat_grid', 'lon_grid', 'grid_id']]]
    y_train_pred = rf.predict(X_train_feat)
    y_val_pred = rf.predict(X_val_feat)

    # 计算精度指标（含RPD）
    test_r2 = round(r2_score(y_test, y_test_pred), 4)
    test_rmse = round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4)
    test_mae = round(mean_absolute_error(y_test, y_test_pred), 4)
    test_rpd = round(np.std(y_test, ddof=1) / test_rmse, 4)
    test_metrics = {
        'R²': test_r2,
        'RMSE (kg/m²)': test_rmse,
        'MAE (kg/m²)': test_mae,
        'CLCD类型': config['target_clcd']
    }

    print(f"\n📈 单一地类（CLCD={config['target_clcd']}）RF模型最终精度（测试集）：")
    for k, v in test_metrics.items():
        print(f"   {k}：{v}")
    print(f"   RPD：{test_rpd:.4f}")

    # ========== 1. 散点图（四大指标固定在左上角） ==========
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})
    slope, intercept, r_val, _, _ = stats.linregress(y_test, y_test_pred)
    xy = np.vstack([y_test, y_test_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_test_sorted, y_test_pred_sorted, z_sorted = y_test.iloc[idx], y_test_pred[idx], z[idx]
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())

    sns.scatterplot(x=y_test_sorted, y=y_test_pred_sorted, c=z_sorted, cmap='viridis', alpha=0.6, s=40, edgecolor='none')
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
    plt.plot(y_test, slope * y_test + intercept, 'r-', lw=2)

    ax = plt.gca()
    x_axis_min, x_axis_max = ax.get_xlim()
    y_axis_min, y_axis_max = ax.get_ylim()
    x_pos = x_axis_min + (x_axis_max - x_axis_min) * 0.05
    y_pos = y_axis_min + (y_axis_max - y_axis_min) * 0.95

    plt.text(
        x=x_pos, y=y_pos,
        s=f'R²={test_r2:.4f}\nRMSE={test_rmse:.4f} (kg/m²)\nMAE={test_mae:.4f} (kg/m²)\nRPD={test_rpd:.4f}',
        fontsize=18, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontfamily='Times New Roman',
        ha='left', va='top', linespacing=1.5
    )

    plt.xlabel('Lab-measured SOCD (kg/m²)', fontsize=17)
    plt.ylabel('Predicted SOCD (kg/m²)', fontsize=17)
    plt.xticks(fontsize=16, fontfamily='Times New Roman')
    plt.yticks(fontsize=16, fontfamily='Times New Roman')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_pred_vs_obs.png", dpi=400)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(f"✅ 散点图已保存（优化后）：{config['output_path']}CLCD_{config['target_clcd']}_RF_pred_vs_obs.png")

    # ========== 2. 误差直方图（英文+新罗马字体） ==========
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    errors = y_test - y_test_pred
    sns.histplot(errors, bins=20, kde=False, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.text(
        x=errors.max() * 0.7, y=plt.gca().get_ylim()[1] * 0.8,
        s=f'RMSE = {test_metrics["RMSE (kg/m²)"]} kg/m²\nMAE = {test_metrics["MAE (kg/m²)"]} kg/m²',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontfamily='Times New Roman'
    )
    plt.title(f'CLCD={config["target_clcd"]} SOCD Prediction Error Distribution (Observed - Predicted)', fontsize=12)
    plt.xlabel('Prediction Error (kg/m²)', fontsize=11)
    plt.ylabel('Sample Count', fontsize=11)
    plt.legend(prop={'family': 'Times New Roman'})
    plt.xticks(fontfamily='Times New Roman')
    plt.yticks(fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_error_histogram.png", dpi=300)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # ========== 3. 特征相关性热力图（修复乱码：全英文+新罗马字体） ==========
    core_numeric_cols = [
        'latitude', 'longitude', 'SOCD_kg_m2',
        'MAT', 'MAP', 'PET', 'AI',
        'DEM', 'TWI', 'slope_deg', 'plan_cur',
        'pH', 'Sand', 'Silt', 'Clay', 'True_Soil_Depth',
        'Vegetation_Factor1', 'Vegetation_Factor2'
    ]
    core_numeric_cols = [col for col in core_numeric_cols if col in df_preprocessed.columns]
    corr_df = df_preprocessed[core_numeric_cols].corr()

    plt.figure(figsize=(14, 12))
    # 强制新罗马字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    sns.heatmap(
        corr_df, annot=True, fmt='.2f', cmap='RdBu_r',
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient'}  # 色条标签改为英文
    )
    # 标题改为英文
    plt.title(
        f'Correlation Heatmap Between Features and SOCD (CLCD={config["target_clcd"]} - {CLCD_EN_MAP[config["target_clcd"]]})',
        fontsize=14, pad=20, fontfamily='Times New Roman')
    plt.xticks(rotation=45, ha='right', fontfamily='Times New Roman')
    plt.yticks(rotation=0, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_feature_corr_heatmap.png", dpi=300)
    plt.close()
    # 恢复中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # ========== 4. 特征重要性图（新罗马+全英文） ==========
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    feat_imp = pd.DataFrame({
        'Feature_Name': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    colors = ['#1f77b4' if i < 3 else '#6baed6' for i in range(len(feat_imp))]
    bars = sns.barplot(x='Importance', y='Feature_Name', data=feat_imp, palette=colors)

    for bar in bars.patches:
        width = bar.get_width()
        bars.text(
            width + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}', va='center', fontsize=10, fontfamily='Times New Roman'
        )

    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    # plt.title(
    #    fontsize=14, pad=15)
    plt.xticks(fontsize=13, fontfamily='Times New Roman')
    plt.yticks(fontsize=13, fontfamily='Times New Roman')
    plt.xlim(0, feat_imp['Importance'].max() * 1.15)
    plt.tight_layout()
    plt.savefig(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_feature_importance.png", dpi=400)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(f"✅ 特征重要性图已保存（优化后）：{config['output_path']}CLCD_{config['target_clcd']}_RF_feature_importance.png")

    # ========== 5. 全国SOCD制图数据（修复特征不匹配） ==========
    X_all = df_preprocessed[[col for col in df_preprocessed.columns if
                             col not in ['latitude', 'longitude', 'SOCD_kg_m2', 'grid_id', 'lat_grid', 'lon_grid']]]
    y_all_pred = rf.predict(X_all)

    national_mapping_data = pd.DataFrame({
        'Longitude': df_preprocessed['longitude'].values.round(6),
        'Latitude': df_preprocessed['latitude'].values.round(6),
        'CLCD_Type': config['target_clcd'],
        'Land_Cover': CLCD_EN_MAP[config['target_clcd']],
        'Predicted_SOCD_kg_m2': y_all_pred.round(4),
        'Model_Name': 'RF'
    })
    mapping_path = f"{config['output_path']}RF_CLCD{config['target_clcd']}_National_SOCD_Mapping_Data.csv"
    national_mapping_data.to_csv(mapping_path, index=False, encoding='gbk')
    print(f"✅ 全国SOCD制图数据已保存：{mapping_path}")

    # ========== 6. 原有CSV输出（保留） ==========
    train_result = pd.DataFrame({
        '纬度': df_train['latitude'].values,
        '经度': df_train['longitude'].values,
        '实测SOCD(kg/m2)': df_train['SOCD_kg_m2'].values,
        '预测SOCD(kg/m2)': y_train_pred,
        '误差(kg/m2)': y_train_pred - df_train['SOCD_kg_m2'].values,
        '数据集类型': 'Train'
    })
    val_result = pd.DataFrame({
        '纬度': df_val['latitude'].values,
        '经度': df_val['longitude'].values,
        '实测SOCD(kg/m2)': df_val['SOCD_kg_m2'].values,
        '预测SOCD(kg/m2)': y_val_pred,
        '误差(kg/m2)': y_val_pred - df_val['SOCD_kg_m2'].values,
        '数据集类型': 'Validation'
    })
    test_result = pd.DataFrame({
        '纬度': df_test['latitude'].values,
        '经度': df_test['longitude'].values,
        '实测SOCD(kg/m2)': df_test['SOCD_kg_m2'].values,
        '预测SOCD(kg/m2)': y_test_pred,
        '误差(kg/m2)': y_test_pred - df_test['SOCD_kg_m2'].values,
        '数据集类型': 'Test'
    })
    all_result = pd.concat([train_result, val_result, test_result], ignore_index=True)
    all_result.to_csv(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_All_Sample_Results.csv", index=False,
                      encoding='gbk')
    print(f"✅ 新增样本CSV输出：{config['output_path']}CLCD_{config['target_clcd']}_RF_All_Sample_Results.csv")

    # 精度指标保存（补充RPD）
    metrics_with_rpd = test_metrics.copy()
    metrics_with_rpd['RPD'] = test_rpd
    metrics_df = pd.DataFrame(metrics_with_rpd, index=[0])
    metrics_df.to_csv(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_metrics.csv", index=False,
                      encoding='utf_8_sig')
    feat_imp.to_csv(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_feature_importance.csv", index=False,
                    encoding='utf_8_sig')
    corr_df.to_csv(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_feature_correlation.csv", index=True,
                   encoding='utf_8_sig')

    # ========== 新增：计算并保存置换重要性（跨模型统一对比用） ==========
    print("\n🔍 计算置换重要性（跨模型统一对比用）...")
    perm_imp_df, base_r2 = calculate_permutation_importance(
        rf, X_test, y_test, feature_cols
    )
    # 保存置换重要性结果
    perm_imp_df.to_csv(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_permutation_importance.csv",
                       index=False, encoding='utf_8_sig')
    print(f"✅ 置换重要性结果已保存：{config['output_path']}CLCD_{config['target_clcd']}_RF_permutation_importance.csv")

    # ========== 新增：绘制置换重要性图（保持原有出图风格） ==========
    # ========== 绘制置换重要性图（提升小数位数，精准显示） ==========
    # ========== 绘制置换重要性图（仅保留核心特征，剔除低贡献值） ==========
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    # 核心过滤逻辑：只保留重要性 > 0.0005 的特征（彻底剔除0.0000/0.0003这类极小值）
    # 阈值可根据你的需求微调，0.0005是兼顾“精简”和“保留有效特征”的最优值
    perm_imp_filtered = perm_imp_df[perm_imp_df['Permutation_Importance'] > 0.0005]

    # 若过滤后特征数太多，仍保留前15个；若太少，全部保留（避免空图）
    if len(perm_imp_filtered) > 15:
        perm_imp_final = perm_imp_filtered.head(15)
    else:
        perm_imp_final = perm_imp_filtered

    # 绘制（无0.0000/0.0003这类无意义数值）
    colors = ['#1f77b4' if i < 3 else '#6baed6' for i in range(len(perm_imp_final))]
    bars = sns.barplot(x='Permutation_Importance', y='Feature_Name', data=perm_imp_final, palette=colors)

    # 标注保留4位小数（精准显示核心特征数值）
    for bar in bars.patches:
        width = bar.get_width()
        bars.text(
            width + 0.0001,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}',
            va='center', fontsize=15, fontfamily='Times New Roman'
        )

    plt.xlabel('Permutation Importance (ΔR²)', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.xticks(fontsize=14, fontfamily='Times New Roman')
    plt.yticks(fontsize=14, fontfamily='Times New Roman')
    plt.xlim(0, perm_imp_final['Permutation_Importance'].max() * 1.15)
    plt.tight_layout()
    plt.savefig(f"{config['output_path']}CLCD_{config['target_clcd']}_RF_permutation_importance.png", dpi=400)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(f"✅ 置换重要性图已保存：{config['output_path']}CLCD_{config['target_clcd']}_RF_permutation_importance.png")

    return test_metrics, feat_imp, test_rpd


# ======================== 7. 主函数（修复test_rpd未定义） ========================
def main():
    if not os.path.exists(CONFIG['output_path']):
        os.makedirs(CONFIG['output_path'])

    print("📥 读取CSV格式原始数据...")
    df_raw = pd.read_csv(CONFIG['data_path'], encoding='gbk')

    df_prep, scaler, all_feat_cols = preprocess_data(df_raw, CONFIG)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df_train, df_val, df_test = spatial_split(df_prep,
                                                                                                            CONFIG)

    rf_model = train_rf(X_train, y_train, X_val, y_val, CONFIG)

    test_metrics, feat_imp, test_rpd = evaluate_rf(rf_model, X_test, y_test, feature_cols, CONFIG, df_prep, df_train,
                                                   df_val, df_test)

    print(f"\n🎉 单一地类（CLCD={CONFIG['target_clcd']}）RF模型训练完成！")
    print(f"📁 所有结果已保存至：{CONFIG['output_path']}")
    print(f"\n📌 核心对比指标：")
    print(
        f"   R²：{test_metrics['R²']} | RMSE：{test_metrics['RMSE (kg/m²)']} kg/m² | MAE：{test_metrics['MAE (kg/m²)']} kg/m² | RPD：{test_rpd:.4f}")


if __name__ == "__main__":
    main()
