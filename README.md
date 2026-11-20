# Soil Organic Carbon Density Mapping in China

**Research Data, Code and Results**

## Research Overview

This project maps soil organic carbon density (SOCD) at 0–30 cm across terrestrial China. The study used 5,838 SOCD samples to compare four models—Random Forest (RF), Geographically Weighted Regression (GWR), Support Vector Regression (SVR), and a 1D Convolutional Neural Network (1D-CNN)—under three modeling strategies. RF was used for the national 30 m SOCD prediction.

The strategies are Global Modeling Strategy (GMS), Land-Cover-Specific Modeling (LCSM), and Region-Specific Modeling (RSM).

## Code

| Directory | Contents |
|---|---|
| [code/modeling/](code/modeling/) | 12 scripts for the four models and three modeling strategies, including model evaluation. |
| [code/gee/](code/gee/) | Earth Engine classifier training and validation, and tiled national 30 m prediction. |
| [code/processing/](code/processing/) | Processing and mosaicking of the 30 m raster tiles. |
| [code/analysis/](code/analysis/) | SOCD statistical analyses, product comparisons, and figures. |

## Results

| File | Contents |
|---|---|
| [model_performance.csv](results/model_performance.csv) | Model performance across the four models and three strategies. |
| [feature_importance_zero_filled.csv](results/feature_importance_zero_filled.csv) | Feature-importance results; zero-filled entries are retained. |
| [socd_500m_national_statistics.csv](results/socd_500m_national_statistics.csv) | National valid-pixel statistics from the 500 m analysis stage. |
| [socd_500m_land_cover_statistics.csv](results/socd_500m_land_cover_statistics.csv) | SOCD statistics by land-cover class at 500 m. |
| [socd_500m_region_statistics.csv](results/socd_500m_region_statistics.csv) | SOCD statistics by region at 500 m. |
| [product_agreement_metrics.csv](results/product_agreement_metrics.csv) | Agreement metrics for the three SOCD products. |
| [product_distribution_statistics.csv](results/product_distribution_statistics.csv) | Distribution statistics for the three SOCD products. |

The national, land-cover, and regional summary tables above belong to the 500 m analysis stage; they are separate from the final 30 m raster product.

## 30 m SOCD Dataset

| Property | Description |
|---|---|
| Study area | Terrestrial China |
| Soil depth | 0–30 cm |
| Spatial resolution | 30 m |
| Unit | kg m⁻² |
| Format | GeoTIFF |

The main data are SOCD GeoTIFF rasters. Optional OVR files are external pyramids for improved display performance in GIS. Download the dataset from the [data-v1 GitHub Release](https://github.com/TitaniumND5/SOCD-30m/releases/tag/data-v1).

The repository root also contains [Download-SOCD.ps1](Download-SOCD.ps1), a volume download and verification helper, and [SHA256SUMS.csv](SHA256SUMS.csv), the checksum list for the release assets.

## Notes

- Set the relevant input-data paths before running research scripts.
- Some GEE scripts use Earth Engine Assets from the original research project and require access to those assets.
- This repository provides the research code and main results; it does not include all original input data.

---

# 中国土壤有机碳密度反演与制图

**研究数据、代码与结果**

## 研究概述

本项目针对中国陆地区域开展 0–30 cm 土壤有机碳密度（SOCD）制图。研究使用 5,838 个 SOCD 样本，对比随机森林（RF）、地理加权回归（GWR）、支持向量回归（SVR）和一维卷积神经网络（1D-CNN）四种模型在三种建模策略下的表现，并使用 RF 生成全国 30 m SOCD 预测成果。

三种策略分别为全局建模（GMS）、土地覆盖类型专项建模（LCSM）和分区建模（RSM）。

## 研究代码

| 目录 | 内容 |
|---|---|
| [code/modeling/](code/modeling/) | 四种模型与三种建模策略的建模和评价代码，共 12 个脚本。 |
| [code/gee/](code/gee/) | Earth Engine 分类器训练、验证及全国 30 m 分块预测代码。 |
| [code/processing/](code/processing/) | 30 m 栅格分块处理与镶嵌代码。 |
| [code/analysis/](code/analysis/) | SOCD 统计分析、产品比较及绘图代码。 |

## 研究结果

| 文件 | 内容 |
|---|---|
| [model_performance.csv](results/model_performance.csv) | 四种模型在三种建模策略下的性能评价。 |
| [feature_importance_zero_filled.csv](results/feature_importance_zero_filled.csv) | 变量重要性结果，保留零填充值。 |
| [socd_500m_national_statistics.csv](results/socd_500m_national_statistics.csv) | 500 m 分析阶段的全国有效像元统计。 |
| [socd_500m_land_cover_statistics.csv](results/socd_500m_land_cover_statistics.csv) | 500 m 分析阶段按土地覆盖类型统计的 SOCD。 |
| [socd_500m_region_statistics.csv](results/socd_500m_region_statistics.csv) | 500 m 分析阶段按地理分区统计的 SOCD。 |
| [product_agreement_metrics.csv](results/product_agreement_metrics.csv) | 三种 SOCD 产品的一致性指标。 |
| [product_distribution_statistics.csv](results/product_distribution_statistics.csv) | 三种 SOCD 产品的分布统计。 |

全国、土地覆盖和分区汇总表属于 500 m 分析阶段，与最终 30 m 栅格产品的分辨率不同。

## 全国 30 m SOCD 数据

| 属性 | 说明 |
|---|---|
| 研究区域 | 中国陆地区域 |
| 土层深度 | 0–30 cm |
| 空间分辨率 | 30 m |
| 数据单位 | kg m⁻² |
| 数据格式 | GeoTIFF |

主数据为 SOCD GeoTIFF 栅格；可选 OVR 文件为外部金字塔，用于提升 GIS 显示效率。数据可从 [data-v1 GitHub Release](https://github.com/TitaniumND5/SOCD-30m/releases/tag/data-v1) 下载。

仓库根目录还提供 [Download-SOCD.ps1](Download-SOCD.ps1) 分卷下载与校验辅助脚本，以及 [SHA256SUMS.csv](SHA256SUMS.csv) Release 附件校验清单。

## 使用说明

- 运行研究脚本前，请设置相应的输入数据路径。
- 部分 GEE 脚本引用原研究项目的 Earth Engine Assets，需要具备相应访问权限。
- 本仓库提供研究代码和主要结果，不包含全部原始输入数据。
