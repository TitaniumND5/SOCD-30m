# 全国 30 m SOCD 栅格成果

本仓库发布全国 30 m 土壤有机碳密度（SOCD）栅格成果。

## 数据说明

- 深度：0–30 cm
- 单位：kg m⁻²
- 格式：GeoTIFF（Float32，NoData = -9999）
- 主数据：完整栅格文件 `SOCD_RF_NATIONAL_30m_v3_mosaic.tif`
- 可选文件：`SOCD_RF_NATIONAL_30m_v3_mosaic.tif.ovr` 外部金字塔，用于加快地图浏览；不影响主数据分析

## 下载

打开 [SOCD 30m Data Release](https://github.com/TitaniumND5/SOCD-30m/releases/tag/data-v1)，在 **Assets** 中下载分卷。GitHub 自动生成的 **Source code (zip)** 不包含栅格数据。

- 仅需主栅格：下载全部 `SOCD_30m_TIFF.7z.001`、`.002` 等分卷。
- 需要更流畅地浏览：同时下载全部 `SOCD_30m_OVR.7z.001`、`.002` 等分卷。
- 同一组的分卷放在同一目录中，文件名和编号保持不变。

也可以用 Windows PowerShell 批量下载并校验：

```powershell
# 仅下载 TIFF
.\Download-SOCD.ps1 -Set TIFF

# 下载 TIFF 与 OVR
.\Download-SOCD.ps1 -Set All
```

下载脚本使用 Windows 自带的 `curl.exe`，默认将文件保存到当前目录；也可指定 `-OutputDirectory`。不需要管理员权限、GitHub CLI 或 Python。校验值见仓库内的 `SHA256SUMS.csv`。

## 解压与使用

1. 确保需要的那一组分卷都已下载，且都在同一个目录。
2. 使用 7-Zip 右键打开该组的 `.7z.001` 文件并解压；只需对 `.001` 操作一次。
3. 获得 `SOCD_RF_NATIONAL_30m_v3_mosaic.tif`。若下载了 OVR，也会获得对应的 `.tif.ovr`。
4. 将 TIFF 与 OVR 放在同一目录并保持原文件名，再用支持 BigTIFF 的 GIS 软件打开 TIFF。

两组分卷彼此独立，可以只解压 TIFF，也可以分别解压 TIFF 和 OVR。
