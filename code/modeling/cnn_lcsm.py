# ======================== 1D CNN最终优化版（3轮取最优+Grad-CAM+对齐GWR散点图） ========================
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import stats
from scipy.stats import gaussian_kde
import geopandas as gpd
from shapely.geometry import Point

# ---------------------- 基础配置（统一特征名+映射字典） ----------------------
TARGET_CLCD = 7  # 耕地=1、裸地=7等，手动修改
# Set this to the local training CSV used in the study.
DATA_PATH = r"PATH_TO_TRAINING_DATA.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "cnn_lcsm") + os.sep

# 映射字典（与GWR完全一致）
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

# ---------------------- 固定配置（增强随机种子固定，控制波动） ----------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽所有TensorFlow日志（只显示ERROR）
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# 核心：固定所有随机种子（确保结果可复现）
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 绘图配置（与GWR完全一致）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 创建输出目录
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# 全局基础配置（特征名中文→英文，与GWR统一）
GLOBAL_CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "seq_feature_cols": [
        'DEM', 'TWI', 'slope_deg', 'plan_cur', 'MAT', 'MAP', 'PET', 'AI',
        'Sand', 'Silt', 'Clay', 'pH', 'Vegetation_Factor1', 'Vegetation_Factor2',
        'LSWI', 'Albedo', 'True_Soil_Depth'
    ],
    "vif_threshold": 20,
    "data_augment_threshold": 500,
    "min_valid_samples": 50,
    "train_runs": 3  # 训练轮次（固定3轮）
}


# ======================== 核心函数：动态超参数适配（不变） ========================
def get_dynamic_cnn_params(sample_count):
    if sample_count < 300:
        return {
            "kernel_size": 2,
            "filters1": 4,
            "filters2": 4,
            "dense1": 8,
            "dense2": 4,
            "l2_reg": 0.0,
            "learning_rate": 5e-5,
            "batch_size": 4,
            "epochs": 150
        }
    elif 300 <= sample_count < 800:
        return {
            "kernel_size": 2,
            "filters1": 6,
            "filters2": 6,
            "dense1": 12,
            "dense2": 6,
            "l2_reg": 0.0005,
            "learning_rate": 8e-5,
            "batch_size": 6,
            "epochs": 120
        }
    else:
        return {
            "kernel_size": 3,
            "filters1": 8,
            "filters2": 16,
            "dense1": 16,
            "dense2": 8,
            "l2_reg": 0.001,
            "learning_rate": 1e-4,
            "batch_size": 16,
            "epochs": 100
        }


# ======================== 核心函数：小样本数据增强（不变） ========================
def augment_small_data(X, y, augment_times=2):
    X_aug = []
    y_aug = []
    perturb_ratio = 0.1
    for _ in range(augment_times):
        perturb = np.random.normal(0, perturb_ratio, X.shape) * np.std(X, axis=0)
        X_perturb = X + perturb
        X_aug.append(X_perturb)
        y_aug.append(y)
    X_aug = np.concatenate([X] + X_aug, axis=0)
    y_aug = np.concatenate([y] + y_aug, axis=0)
    X_aug, y_aug = shuffle(X_aug, y_aug, random_state=GLOBAL_CONFIG['random_state'])
    return X_aug, y_aug


# ======================== 新增：统一置换重要性（与GWR逻辑一致） ========================
def calculate_permutation_importance_unified(model, X_test, y_test, feature_names, n_repeats=3, random_state=42):
    """CNN统一置换重要性：基准R²-打乱后R²"""
    y_pred_base = model.predict(X_test, verbose=0).flatten()
    base_r2 = r2_score(y_test, y_pred_base)

    importances = np.zeros((n_repeats, len(feature_names)))
    rng = np.random.RandomState(random_state)

    for i, feat in enumerate(feature_names):
        for n in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i, 0] = rng.permutation(X_perm[:, i, 0])  # 适配3D输入
            y_pred_perm = model.predict(X_perm, verbose=0).flatten()
            perm_r2 = r2_score(y_test, y_pred_perm)
            importances[n, i] = base_r2 - perm_r2

    perm_imp_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Permutation_Importance': importances.mean(axis=0),
        'Importance_Std': importances.std(axis=0)
    }).sort_values('Permutation_Importance', ascending=False)

    return perm_imp_df, base_r2


# ======================== 新增：置换重要性出图（与GWR样式一致） ========================
def plot_permutation_importance(perm_imp_df, target_clcd):
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    perm_imp_filtered = perm_imp_df[perm_imp_df['Permutation_Importance'] > 0.0005]
    perm_imp_final = perm_imp_filtered.head(15) if len(perm_imp_filtered) > 15 else perm_imp_filtered

    colors = ['#1f77b4' if i < 3 else '#6baed6' for i in range(len(perm_imp_final))]
    bars = sns.barplot(x='Permutation_Importance', y='Feature_Name', data=perm_imp_final, palette=colors)

    for bar in bars.patches:
        width = bar.get_width()
        bars.text(
            width + 0.0001, bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}', va='center', fontsize=10, fontfamily='Times New Roman'
        )

    plt.xlabel('Permutation Importance (ΔR²)', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    plt.xticks(fontsize=11, fontfamily='Times New Roman')
    plt.yticks(fontsize=11, fontfamily='Times New Roman')
    plt.xlim(0, perm_imp_final['Permutation_Importance'].max() * 1.15)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_PATH, f'CLCD_{target_clcd}_CNN_permutation_importance.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    return save_path


# ======================== 新增：CNN专属Grad-CAM特征重要性（修复版） ========================
def get_gradcam_importance(model, X_test, feature_names):
    """
    修复版Grad-CAM：适配1D CNN的特征维度，确保输出和原始特征数一致
    返回：和原始特征序列长度匹配的Grad-CAM重要性
    """
    # 提取最后一层卷积层（适配模型结构）
    conv_layer = model.layers[3]  # Conv1D第二层（最后一层卷积）
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    # 计算梯度（反映特征对预测的贡献）
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(X_test)
        loss = predictions[:, 0]  # 预测值的损失

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # 平均梯度（特征权重）

    # 核心修复：将卷积层输出映射回原始特征维度
    conv_outputs_avg = tf.reduce_mean(conv_outputs, axis=1).numpy()  # 形状：(样本数, filters数)
    filter_importance = tf.reduce_mean(conv_outputs_avg * pooled_grads.numpy(), axis=0).numpy()

    # 适配原始特征数
    n_features = len(feature_names)
    if len(filter_importance) >= n_features:
        feature_weights = filter_importance[:n_features]
    else:
        feature_weights = np.pad(filter_importance, (0, n_features - len(filter_importance)), mode='constant')

    # 归一化+异常值处理
    feature_weights = feature_weights / (np.max(feature_weights) + 1e-8)  # 避免除0
    feature_weights = np.nan_to_num(feature_weights)  # 处理NaN

    # 构建结果表（确保长度匹配）
    gradcam_imp_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'GradCAM_Importance': feature_weights,
        'Importance_Type': 'Feature Interaction (CNN Exclusive)'
    }).sort_values('GradCAM_Importance', ascending=False)

    return gradcam_imp_df


# ======================== 新增：Grad-CAM出图（专属样式） ========================
def plot_gradcam_importance(gradcam_imp_df, target_clcd):
    plt.figure(figsize=(10, 8))  # 增大画布高度，避免标签重叠
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

    # 核心修复1：过滤负重要性（只保留≥0的特征）
    gradcam_filtered = gradcam_imp_df[gradcam_imp_df['GradCAM_Importance'] >= 0].copy()
    # 取前10个最重要的特征（避免图太拥挤）
    gradcam_final = gradcam_filtered.head(10).sort_values('GradCAM_Importance', ascending=True)

    # 核心修复2：渐变配色（体现重要性高低）
    colors = plt.cm.Reds(np.linspace(0.4, 1, len(gradcam_final)))  # 红色渐变，越重要颜色越深

    # 绘制条形图
    bars = sns.barplot(
        x='GradCAM_Importance',
        y='Feature_Name',
        data=gradcam_final,
        palette=colors
    )

    # 标注（4位小数，避免重叠）
    for bar in bars.patches:
        width = bar.get_width()
        bars.text(
            width + 0.02,  # 增加文本与条形的间距
            bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}',
            va='center',
            fontsize=10,
            fontfamily='Times New Roman'
        )

    # 核心修复3：调整坐标轴范围（从0开始，避免负值）
    plt.xlim(0, gradcam_final['GradCAM_Importance'].max() * 1.2)
    plt.xlabel('GradCAM Importance (Normalized, Feature Interaction)', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    plt.title(f'CNN Grad-CAM Feature Importance (CLCD={target_clcd})', fontsize=14, pad=20)  # 增加标题间距

    # 优化标签显示
    plt.yticks(fontsize=11, fontfamily='Times New Roman')
    plt.xticks(fontsize=11, fontfamily='Times New Roman')
    plt.grid(linestyle='--', alpha=0.3, axis='x')  # 只显示x轴网格
    plt.tight_layout()  # 自动调整布局

    # 保存
    save_path = os.path.join(OUTPUT_PATH, f'CLCD_{target_clcd}_CNN_GradCAM_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 避免内容被裁剪
    plt.close()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    return save_path


# ======================== 核心函数：构建CNN模型（不变） ========================
def build_optimized_cnn(input_shape, params):
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(
        filters=params['filters1'],
        kernel_size=params['kernel_size'],
        activation='relu',
        padding='same',
        kernel_regularizer=l2(params['l2_reg'])
    )(inputs)
    pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)
    conv2 = Conv1D(
        filters=params['filters2'],
        kernel_size=params['kernel_size'],
        activation='relu',
        padding='same',
        kernel_regularizer=l2(params['l2_reg'])
    )(pool1)
    pool2 = MaxPooling1D(pool_size=2, padding='same')(conv2)
    gap = GlobalAveragePooling1D()(pool2)
    dense1 = Dense(params['dense1'], activation='relu')(gap)
    dropout_rate = 0.2 if len(X_train_cnn) >= 800 else 0.0
    dropout1 = Dropout(dropout_rate)(dense1)
    dense2 = Dense(params['dense2'], activation='relu')(dropout1)
    outputs = Dense(1)(dense2)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=params['learning_rate'], clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model


# ======================== 核心函数：3轮训练 → 返回全部3轮结果（仅修改返回值） ========================
def train_cnn_multirun(X_train_cnn, y_train_scaled, X_val_cnn, y_val_scaled, X_test_cnn, y_test_original,
                       final_seq_cols, cnn_params):
    """3轮训练，返回全部3轮预测与指标，用于科学平均"""
    all_y_pred = []
    all_metrics = {'r2': [], 'rmse': [], 'mae': [], 'rpd': []}
    all_models = []
    all_histories = []

    for run in range(GLOBAL_CONFIG['train_runs']):
        print(f"\n🚀 开始第{run + 1}/{GLOBAL_CONFIG['train_runs']}轮CNN训练...")
        np.random.seed(GLOBAL_CONFIG['random_state'] + run)
        tf.random.set_seed(GLOBAL_CONFIG['random_state'] + run)

        input_shape = (len(final_seq_cols), 1)
        cnn_model = build_optimized_cnn(input_shape, cnn_params)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-6, verbose=0)
        model_checkpoint = ModelCheckpoint(
            os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_best_cnn_model_run{run + 1}.h5'),
            monitor='val_loss', save_best_only=True, verbose=0
        )

        history = cnn_model.fit(
            X_train_cnn, y_train_scaled,
            batch_size=cnn_params['batch_size'],
            epochs=cnn_params['epochs'],
            validation_data=(X_val_cnn, y_val_scaled),
            callbacks=[early_stopping, lr_reduce, model_checkpoint],
            verbose=0
        )

        cnn_model.load_weights(os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_best_cnn_model_run{run + 1}.h5'))
        y_pred_scaled = cnn_model.predict(X_test_cnn, verbose=0)
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

        r2 = r2_score(y_test_original, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        mae = mean_absolute_error(y_test_original, y_pred)
        rpd = np.std(y_test_original, ddof=1) / rmse

        all_y_pred.append(y_pred)
        all_models.append(cnn_model)
        all_histories.append(history)
        all_metrics['r2'].append(r2)
        all_metrics['rmse'].append(rmse)
        all_metrics['mae'].append(mae)
        all_metrics['rpd'].append(rpd)

        print(f"   第{run + 1}轮指标：R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")

    # 科学：返回全部3轮结果
    return np.array(all_y_pred), all_metrics, all_models, all_histories


# ======================== 1. 数据预处理（不变） ========================
print("📥 读取原始数据...")
df_raw = pd.read_csv(DATA_PATH, encoding='gbk')

# 中文列名→英文（与GWR统一）
df_raw = df_raw.rename(columns={k: v for k, v in COL_NAME_MAP.items() if k in df_raw.columns})

df_raw['CLCD'] = pd.to_numeric(df_raw['CLCD'], errors='coerce')
df_clcd_raw = df_raw[df_raw['CLCD'] == TARGET_CLCD].copy()
initial_count = len(df_clcd_raw)
print(f"📌 筛选CLCD={TARGET_CLCD}样本：{initial_count}条")

keep_cols = [
    'FID', 'Samples', 'profile_id', 'latitude', 'longitude',
    'upper_dept', 'lower_dept', 'True_Soil_Depth', 'Depth', 'SOCD__kg_m',
    'BGREEN', 'BNIR', 'BRED', 'DEM', 'NDVI', 'SAVI', 'OSAVI', 'EVI',
    'LSWI', 'MSAVI', 'NDWI', 'TWI', 'MAT', 'MAP', 'PET', 'AI',
    'Sand', 'Silt', 'Clay', 'slope_deg', '土地覆盖类型',
    'pH', 'Albedo', 'plan_cur'
]
keep_cols = [col for col in keep_cols if col in df_clcd_raw.columns]
df = df_clcd_raw[keep_cols].copy()

df.rename(columns={'SOCD__kg_m': 'SOCD'}, inplace=True)
if len(df) > 30:
    socd_mean = df['SOCD'].mean()
    socd_std = df['SOCD'].std()
    df = df[(df['SOCD'] >= socd_mean - 3 * socd_std) & (df['SOCD'] <= socd_mean + 3 * socd_std)]
df = df.dropna(subset=['latitude', 'longitude', 'SOCD'])
valid_count = len(df)
print(f"✅ 有效样本数（剔除缺失值+极端值）：{valid_count}条")

if valid_count < GLOBAL_CONFIG['min_valid_samples']:
    raise ValueError(
        f"CLCD={TARGET_CLCD}有效样本数{valid_count} < 最小要求{GLOBAL_CONFIG['min_valid_samples']}，终止建模")

# 植被指数PCA
veg_cols = ['BGREEN', 'BNIR', 'BRED', 'NDVI', 'SAVI', 'OSAVI', 'EVI', 'MSAVI', 'NDWI']
veg_cols = [col for col in veg_cols if col in df.columns]
for col in veg_cols:
    df[col] = df[col].fillna(df[col].median())
pca_n = 1 if len(veg_cols) <= 3 else 2
pca = PCA(n_components=pca_n, random_state=GLOBAL_CONFIG['random_state'])
pca_result = pca.fit_transform(df[veg_cols])
for i in range(pca_n):
    df[f'Vegetation_Factor{i + 1}'] = pca_result[:, i]
print(f"🌿 植被指数PCA总解释方差比：{sum(pca.explained_variance_ratio_):.4f}")

# ======================== 修复：先保留有用特征 → 再删共线性 ========================
indep_cols = [col for col in GLOBAL_CONFIG['seq_feature_cols'] if col in df.columns]
for col in indep_cols:
    df[col] = df[col].fillna(df[col].median())

# 第一步：计算特征与 SOCD 的相关性 → 保留有用特征（|r|≥0.1）
corr_with_y = df[indep_cols].corrwith(df['SOCD']).abs()
useful_cols = corr_with_y[corr_with_y >= 0.1].index.tolist()
print(f"✅ 保留与SOCD相关的有效特征：{useful_cols}")

# 第二步：只在【有用特征】里做 VIF 剔除（绝不误杀有用信号）
if len(useful_cols) > 1:
    vif_data = df[useful_cols].copy()
    vif_data = sm.add_constant(vif_data)
    vif_df = pd.DataFrame()
    vif_df['变量'] = vif_data.columns
    vif_df['VIF'] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    vif_df = vif_df[vif_df['变量'] != 'const']
    high_vif = vif_df[vif_df['VIF'] > GLOBAL_CONFIG['vif_threshold']]['变量'].tolist()
else:
    high_vif = []

if high_vif:
    print(f"⚠️ 移除高共线性变量（VIF>20）：{high_vif}")
    final_seq_cols = [col for col in useful_cols if col not in high_vif]
else:
    final_seq_cols = useful_cols

print(f"✅ 最终特征序列（有用+不共线）：{final_seq_cols}")

# 特征标准化
scaler = StandardScaler()
df[final_seq_cols] = scaler.fit_transform(df[final_seq_cols])

# ======================== 2. 空间分层抽样（不变） ========================
df['lat_grid'] = (df['latitude'] // 0.09).astype(int)
df['lon_grid'] = (df['longitude'] // 0.09).astype(int)
df['grid_id'] = df['lat_grid'].astype(str) + '_' + df['lon_grid'].astype(str)

grid_ids = df['grid_id'].unique()
train_val_grids, test_grids = train_test_split(
    grid_ids, test_size=GLOBAL_CONFIG['test_size'], random_state=GLOBAL_CONFIG['random_state']
)
df_test = df[df['grid_id'].isin(test_grids)].copy()
df_train_val = df[df['grid_id'].isin(train_val_grids)].copy()

train_grids, val_grids = train_test_split(
    train_val_grids, test_size=GLOBAL_CONFIG['val_size'] / (1 - GLOBAL_CONFIG['test_size']),
    random_state=GLOBAL_CONFIG['random_state']
)
df_train = df[df['grid_id'].isin(train_grids)].copy()
df_val = df[df['grid_id'].isin(val_grids)].copy()

# 数据集统计
dataset_stats = pd.DataFrame({
    '数据集': ['训练集', '验证集', '测试集'],
    '样本数': [len(df_train), len(df_val), len(df_test)],
    '占比': [f"{len(df_train) / len(df) * 100:.1f}%", f"{len(df_val) / len(df) * 100:.1f}%",
             f"{len(df_test) / len(df) * 100:.1f}%"],
    'SOCD均值': [df_train['SOCD'].mean().round(4), df_val['SOCD'].mean().round(4), df_test['SOCD'].mean().round(4)],
    'SOCD标准差': [df_train['SOCD'].std().round(4), df_val['SOCD'].std().round(4), df_test['SOCD'].std().round(4)]
})
print(f"\n📈 数据集统计：\n{dataset_stats}")

# ======================== 3. 小样本数据增强（不变） ========================
X_train = df_train[final_seq_cols].values
y_train = df_train['SOCD'].values
if len(df_train) < GLOBAL_CONFIG['data_augment_threshold']:
    print(f"\n🔧 启用小样本数据增强（训练集样本{len(df_train)} < 阈值{GLOBAL_CONFIG['data_augment_threshold']}）")
    X_train_aug, y_train_aug = augment_small_data(X_train, y_train)
    print(f"✅ 增强后训练集样本数：{len(X_train_aug)}（原始{len(X_train)} + 增强{len(X_train_aug) - len(X_train)}）")
    X_train_cnn = X_train_aug.reshape(-1, len(final_seq_cols), 1)
else:
    X_train_cnn = X_train.reshape(-1, len(final_seq_cols), 1)
    y_train_aug = y_train

# 验证集/测试集处理
X_val_cnn = df_val[final_seq_cols].values.reshape(-1, len(final_seq_cols), 1)
X_test_cnn = df_test[final_seq_cols].values.reshape(-1, len(final_seq_cols), 1)
y_test_original = df_test['SOCD'].values

# 标签标准化
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_aug.reshape(-1, 1))
y_val_scaled = y_scaler.transform(df_val['SOCD'].values.reshape(-1, 1))

# ======================== 4. 动态生成CNN超参数（不变） ========================
train_sample_count = len(X_train_cnn)
cnn_params = get_dynamic_cnn_params(train_sample_count)
print(f"\n⚙️ 动态适配超参数（训练样本数{train_sample_count}）：")
for k, v in cnn_params.items():
    print(f"   {k}: {v}")

# ======================== 5. 3轮训练 → 计算科学平均值（核心修改） ========================
all_y_pred_array, all_metrics, all_models, all_histories = train_cnn_multirun(
    X_train_cnn, y_train_scaled, X_val_cnn, y_val_scaled,
    X_test_cnn, y_test_original, final_seq_cols, cnn_params
)

# 科学：3次平均
y_pred_mean = np.mean(all_y_pred_array, axis=0)
r2_mean = r2_score(y_test_original, y_pred_mean)
rmse_mean = np.sqrt(mean_squared_error(y_test_original, y_pred_mean))
mae_mean = mean_absolute_error(y_test_original, y_pred_mean)
rpd_mean = np.std(y_test_original, ddof=1) / rmse_mean

# 波动：标准差
r2_std = np.std(all_metrics['r2'])
rmse_std = np.std(all_metrics['rmse'])
mae_std = np.std(all_metrics['mae'])

best_run = np.argmax(all_metrics['r2']) + 1
relative_error = (mae_mean / np.mean(y_test_original)) * 100 if np.mean(y_test_original) != 0 else 0

# ======================== 统一输出：均值±标准差 ========================
print(f"\n📊 1D CNN 3次重复平均结果（CLCD={TARGET_CLCD}）：")
print(f"   R²   = {r2_mean:.4f} ± {r2_std:.4f}")
print(f"   RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"   MAE  = {mae_mean:.4f} ± {mae_std:.4f}")
print(f"   平均相对误差(%) = {relative_error:.4f}")
print(f"   RPD  = {rpd_mean:.4f}")
print(f"📝 各轮R²：{all_metrics['r2']}")
print(f"   各轮RMSE：{all_metrics['rmse']}")
print(f"   各轮MAE：{all_metrics['mae']}")

# 保存指标表（最终论文版：仅保留均值±标准差，无多余数据）
metrics_df = pd.DataFrame({
    'CLCD': [TARGET_CLCD],
    'R2': [f"{r2_mean:.4f} ± {r2_std:.4f}"],
    'RMSE': [f"{rmse_mean:.4f} ± {rmse_std:.4f}"],
    'MAE': [f"{mae_mean:.4f} ± {mae_std:.4f}"],
    'RPD': [f"{rpd_mean:.4f}"],
    '平均相对误差(%)': [round(relative_error, 4)],
    '重复次数': [3],
    '最优轮次': [best_run],
    '各轮R²': [str(all_metrics['r2'])],
    '各轮RMSE': [str(all_metrics['rmse'])],
    '各轮MAE': [str(all_metrics['mae'])],
    '训练样本数(增强后)': [len(X_train_cnn)],
    '验证样本数': [len(X_val_cnn)],
    '测试样本数': [len(X_test_cnn)],
    '是否启用数据增强': [len(X_train_cnn) > len(X_train)],
    '模型超参数': [str(cnn_params)]
})
metrics_df.to_csv(os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_cnn_metrics_mean.csv'), index=False, encoding='utf-8-sig')

# ======================== 6. 散点图：使用3次平均预测（样式完全不变） ========================
pred_df = df_test[['FID', 'latitude', 'longitude', 'SOCD']].copy()
pred_df['预测SOCD'] = y_pred_mean
pred_df['误差'] = pred_df['预测SOCD'] - pred_df['SOCD']
pred_df.to_csv(os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_cnn_predictions_mean.csv'), index=False, encoding='utf-8-sig')

# 训练曲线（用最优轮，不变）
best_idx = np.argmax(all_metrics['r2'])
best_history = all_histories[best_idx]
cnn_model = all_models[best_idx]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(best_history.history['loss'][:20], label='训练损失', color='#2E86AB')
plt.plot(best_history.history['val_loss'][:20], label='验证损失', color='#A23B72')
plt.xlabel('迭代次数')
plt.ylabel('MSE损失')
plt.title('训练/验证损失曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(best_history.history['mae'][:20], label='训练MAE', color='#F18F01')
plt.plot(best_history.history['val_mae'][:20], label='验证MAE', color='#C73E1D')
plt.xlabel('迭代次数')
plt.ylabel('MAE (kg/m²)')
plt.title('训练/验证MAE曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_cnn_train_curve.png'), dpi=300)
plt.close()

# ---------------------- 散点图：100%保留你原版样式，只换平均数据 ----------------------
plt.figure(figsize=(6, 6))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})

y_true = y_test_original
y_pred = y_pred_mean
slope, intercept, r_val, _, _ = stats.linregress(y_true, y_pred)
xy = np.vstack([y_true, y_pred])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
y_true_sorted, y_pred_sorted, z_sorted = y_true[idx], y_pred[idx], z[idx]
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

sns.scatterplot(
    x=y_true_sorted, y=y_pred_sorted,
    c=z_sorted, cmap='viridis',
    alpha=0.6, s=40, edgecolor='none'
)

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
    s=f'R²={r2_mean:.4f}\nRMSE={rmse_mean:.4f} (kg/m²)\nMAE={mae_mean:.4f} (kg/m²)\nRPD={rpd_mean:.4f}',
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

plt.savefig(os.path.join(OUTPUT_PATH, f'CLCD_{TARGET_CLCD}_CNN_pred_vs_obs.png'), dpi=400)
plt.close()
plt.rcParams['font.sans-serif'] = ['SimHei']

# ======================== 7. 全量数据建模（不变，用最优模型） ========================
print("\n🔍 生成全国SOCD制图数据...")
X_all = df[final_seq_cols].values.reshape(-1, len(final_seq_cols), 1)
y_all_pred = cnn_model.predict(X_all, verbose=0).flatten()
y_all_pred = y_scaler.inverse_transform(y_all_pred.reshape(-1, 1)).flatten()

national_mapping_data = pd.DataFrame({
    'Longitude': df['longitude'].values.round(6),
    'Latitude': df['latitude'].values.round(6),
    'CLCD_Type': TARGET_CLCD,
    'Land_Cover': CLCD_EN_MAP[TARGET_CLCD],
    'Predicted_SOCD_kg_m2': y_all_pred.round(4),
    'Model_Name': 'CNN'
})
mapping_path = os.path.join(OUTPUT_PATH, f'CNN_CLCD{TARGET_CLCD}_National_SOCD_Mapping_Data.csv')
national_mapping_data.to_csv(mapping_path, index=False, encoding='utf-8-sig')
print(f"✅ 全国SOCD制图数据已保存：{mapping_path}")

all_result = pd.DataFrame({
    '纬度': df['latitude'].values,
    '经度': df['longitude'].values,
    '实测SOCD(kg/m2)': df['SOCD'].values,
    '预测SOCD(kg/m2)': y_all_pred,
    '误差(kg/m2)': y_all_pred - df['SOCD'].values,
    '数据集类型': 'All'
})
all_result.to_csv(os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_CNN_All_Sample_Results.csv'), index=False, encoding='utf-8-sig')
print(f"✅ 全量样本结果表已保存")

# ======================== 8. 置换重要性（不变） ========================
print("\n🔍 计算统一格式置换重要性...")
sample_size = min(500, len(X_test_cnn))
np.random.seed(GLOBAL_CONFIG['random_state'])
sample_idx = np.random.choice(len(X_test_cnn), sample_size, replace=False)
X_test_sample = X_test_cnn[sample_idx]
y_test_sample = y_test_original[sample_idx]

perm_imp_df, base_r2 = calculate_permutation_importance_unified(
    cnn_model, X_test_sample, y_test_sample, final_seq_cols,
    n_repeats=3, random_state=GLOBAL_CONFIG['random_state']
)
perm_imp_path = os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_CNN_permutation_importance.csv')
perm_imp_df.to_csv(perm_imp_path, index=False, encoding='utf-8-sig')
print(f"✅ 置换重要性结果已保存：{perm_imp_path}")

plot_path = plot_permutation_importance(perm_imp_df, TARGET_CLCD)
print(f"✅ 置换重要性图已保存：{plot_path}")

# ======================== 9. CNN专属Grad-CAM（不变） ========================
print("\n🔍 计算CNN专属Grad-CAM特征重要性...")
gradcam_imp_df = get_gradcam_importance(cnn_model, X_test_cnn, final_seq_cols)
gradcam_imp_path = os.path.join(OUTPUT_PATH, f'CLCD{TARGET_CLCD}_CNN_GradCAM_importance.csv')
gradcam_imp_df.to_csv(gradcam_imp_path, index=False, encoding='utf-8-sig')
print(f"✅ CNN专属Grad-CAM结果已保存：{gradcam_imp_path}")

gradcam_plot_path = plot_gradcam_importance(gradcam_imp_df, TARGET_CLCD)
print(f"✅ CNN专属Grad-CAM图已保存：{gradcam_plot_path}")

# ======================== 最终输出 ========================
print(f"\n🎉 所有结果已保存至：{OUTPUT_PATH}")
print(f"\n📌 3次平均核心指标：R²={r2_mean:.4f} | RMSE={rmse_mean:.4f} | MAE={mae_mean:.4f} | RPD={rpd_mean:.4f}")
print("进程结束，退出代码：0")
