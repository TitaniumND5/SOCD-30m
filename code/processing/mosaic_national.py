"""Lossless national SOCD mosaic; originals remain read-only. No reprojection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import numpy as np
import rasterio
from rasterio.windows import Window

# Configure these locations before use; source GeoTIFFs are not included in this package.
INPUT = Path(r'PATH_TO_30M_TILE_DIRECTORY')
WORK = Path(__file__).resolve().parent
OUT = Path(r'PATH_TO_MOSAIC_OUTPUT_DIRECTORY')
BASE = 'SOCD_RF_NATIONAL_30m_v3_mosaic'
GDAL_BIN = Path(r'PATH_TO_GDAL_BIN_DIRECTORY')
EXPECTED = {f'R{r:02}C{c:02}' for r in range(1, 5) for c in range(1, 6)} - {
    'R01C01', 'R01C02', 'R01C03'
}
PATTERN = re.compile(r'^SOCD_RF_NATIONAL_30m_v3_(R\d{2}C\d{2}(?:_[ABCD])*)(?:-\d+-\d+)?\.tif$')


def save_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def inventory():
    records = []
    excluded = []
    reference = None
    for path in sorted(INPUT.rglob('*.tif')):
        match = PATTERN.fullmatch(path.name)
        if not match or '镶嵌后结果' in path.relative_to(INPUT).parts:
            excluded.append(str(path))
            continue
        with rasterio.open(path) as src:
            if reference is None:
                reference = (src.crs, src.transform)
            crs, transform = reference
            assert src.crs == crs, f'CRS mismatch: {path}'
            assert src.count == 1 and src.dtypes == ('float32',), f'Type mismatch: {path}'
            assert src.nodata == -9999, f'NoData mismatch: {path}'
            assert src.transform.a == 30 and src.transform.e == -30, f'Resolution mismatch: {path}'
            assert src.transform.b == src.transform.d == 0, f'Rotated grid: {path}'
            for offset in ((src.transform.c - transform.c) / 30,
                           (src.transform.f - transform.f) / 30):
                assert abs(offset - round(offset)) < 1e-7, f'Grid misalignment: {path}'
            records.append(dict(path=str(path), tile_id=match.group(1),
                                width=src.width, height=src.height,
                                bounds=list(src.bounds), transform=list(src.transform)[:6],
                                bytes=path.stat().st_size))
    assert records, 'No source TIFFs'
    ids = {r['tile_id'] for r in records}
    missing = EXPECTED - {v.split('_')[0] for v in ids}
    assert not missing, f'Missing base tiles: {sorted(missing)}'
    for tile in ids:
        assert not any(other.startswith(tile + '_') for other in ids), f'Parent/child duplicate: {tile}'
    if 'R04C05' not in ids:
        assert {f'R04C05_{c}' for c in 'ABCD'} <= ids, 'Missing R04C05 child'
    overlaps = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            x = min(a['bounds'][2], b['bounds'][2]) - max(a['bounds'][0], b['bounds'][0])
            y = min(a['bounds'][3], b['bounds'][3]) - max(a['bounds'][1], b['bounds'][1])
            if x > 1e-6 and y > 1e-6:
                assert min(x, y) <= 30.000001, f'Unexpected wide overlap: {a["path"]}, {b["path"]}'
                left = max(a['bounds'][0], b['bounds'][0])
                top = min(a['bounds'][3], b['bounds'][3])
                with rasterio.open(a['path']) as sa, rasterio.open(b['path']) as sb:
                    arrays = []
                    for ds in (sa, sb):
                        arrays.append(ds.read(1, window=Window(
                            round((left - ds.transform.c) / 30),
                            round((ds.transform.f - top) / 30), round(x / 30), round(y / 30))))
                aa, bb = arrays
                common = (aa != -9999) & (bb != -9999) & np.isfinite(aa) & np.isfinite(bb)
                differences = np.abs(aa[common].astype('float64') - bb[common])
                overlaps.append(dict(first=a['path'], second=b['path'], pixels=aa.size,
                                     both_valid=int(common.sum()),
                                     different_valid_pixels=int(np.count_nonzero(differences)),
                                     max_abs_difference=float(differences.max()) if differences.size else 0))
    bounds = [min(r['bounds'][0] for r in records), min(r['bounds'][1] for r in records),
              max(r['bounds'][2] for r in records), max(r['bounds'][3] for r in records)]
    width = round((bounds[2] - bounds[0]) / 30)
    height = round((bounds[3] - bounds[1]) / 30)
    summary = dict(input_dir=str(INPUT), output_dir=str(OUT), input_count=len(records),
                   expected_empty_base_tiles=['R01C01', 'R01C02', 'R01C03'],
                   source_tile_ids=sorted(ids), missing_base_tiles=sorted(missing),
                   excluded_files=excluded, overlaps=overlaps,
                   overlap_policy='Sorted file order; last valid source wins in one-pixel task-edge overlaps.',
                   crs=reference[0].to_wkt(),
                   resolution=[30, 30], nodata=-9999, dtype='float32', bounds=bounds,
                   width=width, height=height, source_bytes=sum(r['bytes'] for r in records),
                   uncompressed_bytes=width * height * 4,
                   free_disk_bytes=shutil.disk_usage(INPUT).free)
    return records, summary


def run_gdal(name, args):
    env = os.environ.copy()
    env['PATH'] = str(GDAL_BIN) + os.pathsep + env.get('PATH', '')
    env['GDAL_FILENAME_IS_UTF8'] = 'YES'
    env['GDAL_CACHEMAX'] = '512'
    cmd = [str(GDAL_BIN / (name + '.exe')), *map(str, args)]
    print('Running:', name, flush=True)
    with (WORK / (name + '.log')).open('x', encoding='utf-8') as log:
        log.write(json.dumps(cmd, ensure_ascii=False) + '\n')
        log.flush()
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def verify(records, summary, path):
    rng = random.Random(20261013)
    checked = valid = 0
    sampled_min = math.inf
    sampled_max = -math.inf
    with rasterio.open(path) as dst:
        assert (dst.width, dst.height) == (summary['width'], summary['height'])
        assert dst.res == (30, 30) and dst.nodata == -9999 and dst.dtypes == ('float32',)
        assert dst.crs == rasterio.crs.CRS.from_wkt(summary['crs'])
        for record in records:
            with rasterio.open(record['path']) as src:
                dc = round((src.transform.c - dst.transform.c) / 30)
                dr = round((dst.transform.f - src.transform.f) / 30)
                positions = [(2, 2), (max(2, src.width - 66), max(2, src.height - 66)),
                             (max(0, src.width // 2 - 32), max(0, src.height // 2 - 32))]
                positions += [(rng.randrange(2, max(3, src.width - 65)),
                               rng.randrange(2, max(3, src.height - 65))) for _ in range(5)]
                for col, row in positions:
                    w = min(64, src.width - col)
                    h = min(64, src.height - row)
                    a = src.read(1, window=Window(col, row, w, h))
                    b = dst.read(1, window=Window(col + dc, row + dr, w, h))
                    assert np.array_equal(a, b, equal_nan=True), f'Pixel mismatch: {record["path"]}'
                    checked += a.size
                    values = a[(a != -9999) & np.isfinite(a)]
                    valid += values.size
                    if values.size:
                        sampled_min = min(sampled_min, float(values.min()))
                        sampled_max = max(sampled_max, float(values.max()))
    return dict(exact_sampled_pixels=checked, valid_sampled_pixels=valid,
                sampled_min=sampled_min if valid else None,
                sampled_max=sampled_max if valid else None, exact_match=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse this run\'s manifest and VRT; TIFF must not exist.')
    args = parser.parse_args()
    records, summary = inventory()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.build and not args.resume:
        return
    OUT.mkdir(exist_ok=True)
    vrt = OUT / (BASE + '.vrt')
    tif = OUT / (BASE + '.tif')
    outputs = [tif, OUT / 'mosaic_qc.json', OUT / 'README.md', WORK / 'gdal_translate.log']
    if not args.resume:
        outputs += [vrt, OUT / 'input_manifest.csv', WORK / 'gdalbuildvrt.log', WORK / 'source_files.txt']
    for path in outputs:
        assert not path.exists(), f'Refuse to overwrite: {path}'
    # Compressed output is expected near source size. Use standard TIFF blocks for compatibility.
    assert summary['free_disk_bytes'] > summary['source_bytes'] * 1.5 + 10 * 2**30, 'Insufficient disk headroom'
    if not args.resume:
        with (OUT / 'input_manifest.csv').open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        with (WORK / 'source_files.txt').open('x', encoding='utf-8') as handle:
            handle.write('\n'.join(r['path'] for r in records) + '\n')
    else:
        with (OUT / 'input_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
            assert [r['path'] for r in csv.DictReader(handle)] == [r['path'] for r in records]
        assert vrt.exists(), 'Resume requires VRT'
    started = time.time()
    if not args.resume:
        run_gdal('gdalbuildvrt', ['-resolution', 'user', '-tr', '30', '30', '-srcnodata', '-9999',
                                '-vrtnodata', '-9999', '-input_file_list', WORK / 'source_files.txt', vrt])
    print('VRT exact-value verification:', verify(records, summary, vrt), flush=True)
    run_gdal('gdal_translate', ['--config', 'CHECK_DISK_FREE_SPACE', 'FALSE', '-of', 'GTiff',
                               '-co', 'COMPRESS=LZW', '-co', 'PREDICTOR=2', '-co', 'TILED=YES',
                               '-co', 'BLOCKXSIZE=256', '-co', 'BLOCKYSIZE=256', '-co', 'BIGTIFF=YES',
                               '-co', 'NUM_THREADS=4',
                               '-a_nodata', '-9999', vrt, tif])
    qc = verify(records, summary, tif)
    summary.update(qc)
    summary['elapsed_seconds'] = time.time() - started
    summary['output_bytes'] = tif.stat().st_size
    save_json(OUT / 'mosaic_qc.json', summary)
    readme = f'''# 全国 SOCD 30 m 镶嵌\n\n输入：{INPUT}\n\n主结果：{tif.name}\n\n- {len(records)} 个原始 GeoTIFF，17 个基础块（R04C05 使用 A–D）。\n- R01C01–R01C03 按用户确认的无陆地交集处理。\n- CLCD Albers、30 m、Float32、NoData=-9999；不重投影、不插值、不平滑、不修改值。\n- LZW 无损压缩、256×256 内部分块、BigTIFF；空白块采用 GDAL sparse 存储。建议使用 GDAL/QGIS/ArcGIS 读取。\n- 保留 VRT（依赖原始输入路径）、输入清单和 QC JSON。原始文件及已有单块镶嵌未修改。\n- 精确抽样比对 {qc['exact_sampled_pixels']:,} 个像元（其中有效 {qc['valid_sampled_pixels']:,} 个），全部一致。\n- 本轮按用户要求仅做快速完整性检查；不宣称已完成逐像元全国覆盖或全部下载文件完整性审计。\n- 脚本：{Path(__file__).resolve()}\n- 过程日志：{WORK}\n'''
    readme = readme.replace('空白块采用 GDAL sparse 存储。建议使用 GDAL/QGIS/ArcGIS 读取。',
                            '采用标准 TIFF 块存储（非 sparse）；适用于支持 BigTIFF 的 GIS 软件。')
    readme += f"\n相邻任务存在 {len(summary['overlaps'])} 处一像元重叠，共同有效边缘像元均逐值比较；差异像元数 {sum(o['different_valid_pixels'] for o in summary['overlaps'])}。拼接重叠处按清单顺序后一个有效值优先。\n"
    with (OUT / 'README.md').open('x', encoding='utf-8') as handle:
        handle.write(readme)
    print('COMPLETE', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
