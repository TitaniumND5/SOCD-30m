// ============================================================
// National 0-30 cm SOCD: final candidate, v3 — tiled 30 m production
// using the frozen SOCD_RF_v3_classifier Earth Engine Asset.
//
// This is an Asset-classifier variant of the pilot-verified tiled script.
// It preserves the frozen feature construction, task tiling and export logic.
// ============================================================

// ----------------------- Run controls -----------------------
// Normal operation: set EXPORT_ALL_TILES=true to create all 20 base tasks,
// or set it false and list one or more IDs in EXPORT_TILE_IDS for a pilot.
// The Earth Engine asset IDs below are from the original project and require access
// to that project; they are retained to document the production dependency chain.
var EXPORT_ALL_TILES = false;
var EXPORT_TILE_IDS = ['R02C03'];
var EXPORT_FOLDER = 'SOCD_Paper_tiled_30m_v3';

// Failure recovery.  Leave null for base tiles.  For example, setting this
// to 'R02C03' creates only R02C03_A ... R02C03_D.  A child may itself be
// subdivided later (for example 'R02C03_A') using the same 2 x 2 logic.
var SUBDIVIDE_PARENT_TILE = null;

var EXPORT_GRID_ROWS = 4;
var EXPORT_GRID_COLS = 5;
var EXPORT_SCALE = 30;
var TILE_SCALE = 16;
var PREDICTION_DEPTH_CM = 30;
var START_DATE = '2012-01-01';
var END_DATE = '2024-01-01';
var YEARS = 12;

var RF_CLASSIFIER_ASSET =
  'projects/qingdao-coastline-changes/assets/SOCD_RF_v3_classifier';

var CHINA = ee.FeatureCollection(
  'projects/qingdao-coastline-changes/assets/China_Boundary_Standard'
).geometry();

// Keep the server-side projection for spatial geometry.  Its client-side WKT
// is used only as Export's CRS string; CLCD does not expose a stable EPSG code.
var TARGET_PROJECTION = ee.Image(
  'projects/qingdao-coastline-changes/assets/CLCD_v01_2020_albert'
).projection();
var TARGET_CRS = TARGET_PROJECTION.getInfo().wkt;

var RF_FEATURES = [
  'MAT', 'MAP', 'PET', 'DEM', 'TWI', 'slope_deg', 'plan_cur',
  'Albedo', 'Sand', 'Silt', 'Clay', 'True_Soil_Depth',
  'Vegetation_Factor1', 'Vegetation_Factor2'
];

function buildTerrain() {
  // Raw SRTM: do not smooth, clip, or force a projection before derivatives.
  var dem = ee.Image('USGS/SRTMGL1_003').rename('DEM'); // m
  var terrain = ee.Algorithms.Terrain(dem);
  var slope = terrain.select('slope').rename('slope_deg'); // degrees
  var gradient = dem.gradient();
  var dx = gradient.select('x');
  var dy = gradient.select('y');
  var d2x = dx.gradient().select('x');
  var d2y = dy.gradient().select('y');
  var dxdy = dx.gradient().select('y');
  var planCur = d2x.multiply(dy.pow(2))
    .subtract(dxdy.multiply(2).multiply(dx).multiply(dy))
    .add(d2y.multiply(dx.pow(2)))
    .rename('plan_cur');

  // Retained source-sampling TWI definition; it is not replaced by MERIT TWI.
  var flowAccum = ee.Terrain.aspect(dem).multiply(0).add(1).focal_max({
    radius: 5, units: 'pixels'
  });
  var twi = flowAccum.divide(slope.multiply(Math.PI / 180).tan().add(0.0001))
    .add(1).log().rename('TWI');
  return ee.Image.cat([dem, twi, slope, planCur]);
}

function landsatPrep(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(8).eq(0)   // cloud
    .and(qa.bitwiseAnd(16).eq(0))     // cloud shadow
    .and(qa.bitwiseAnd(32).eq(0));    // snow
  var refl = image.updateMask(mask)
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            ['BLUE', 'BGREEN', 'BRED', 'BNIR', 'SWIR1', 'SWIR2'])
    .multiply(0.0000275).add(-0.2);
  var ndvi = refl.normalizedDifference(['BNIR', 'BRED']).rename('NDVI');
  var lswi = refl.normalizedDifference(['BNIR', 'SWIR1']).rename('LSWI');
  var ndwi = refl.normalizedDifference(['BGREEN', 'BNIR']).rename('NDWI');
  var albedo = refl.expression(
    '0.356*B + 0.130*R + 0.373*N + 0.085*S1 + 0.072*S2 - 0.018', {
      B: refl.select('BLUE'), R: refl.select('BRED'), N: refl.select('BNIR'),
      S1: refl.select('SWIR1'), S2: refl.select('SWIR2')
    }).rename('Albedo');
  return refl.addBands([ndvi, lswi, ndwi, albedo]);
}

// The region argument only removes Landsat scenes that cannot contribute to
// the requested area.  All time, QA, cloud-cover and median definitions stay
// exactly the same as the national v3 baseline.
function buildLandsat(filterRegion) {
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(filterRegion)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(landsatPrep)
    .median();
}

function buildPCA(landsat) {
  // Python PCA fitted directly to unscaled NDVI, LSWI, NDWI in this order.
  var mean = ee.Image.constant([
    0.3778213194319611, 0.03947964550548265, -0.3997132175921266
  ]);
  var centered = landsat.select(['NDVI', 'LSWI', 'NDWI']).subtract(mean);
  var pca1 = centered.multiply([
    0.7110950942155324, 0.46056306554342974, -0.531248933777604
  ]).reduce(ee.Reducer.sum()).rename('Vegetation_Factor1');
  var pca2 = centered.multiply([
    0.6944756144667882, -0.5780468042657547, 0.4284454609272047
  ]).reduce(ee.Reducer.sum()).rename('Vegetation_Factor2');
  return ee.Image.cat([pca1, pca2]);
}

function buildClimate() {
  var tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
    .filterDate(START_DATE, END_DATE)
    // Continuous climate fields: apply bilinear resampling before reductions.
    // This is the only spatial interpolation in the frozen v3 workflow.
    .map(function(image) {
      return image.resample('bilinear');
    });
  var mat = tc.select(['tmmn', 'tmmx']).mean().reduce(ee.Reducer.mean())
    .multiply(0.1).rename('MAT'); // degree C
  var map = tc.select('pr').mean().rename('MAP'); // mm/month
  var pet = tc.select('pet').sum().divide(YEARS).multiply(0.1)
    .rename('PET'); // mm/year
  return ee.Image.cat([mat, map, pet]);
}

function depthWeighted(id, outputName) {
  var image = ee.Image(id);
  return image.select('b0').multiply(5 / 30)
    .add(image.select('b10').multiply(10 / 30))
    .add(image.select('b30').multiply(15 / 30))
    .rename(outputName);
}

function buildSoil() {
  var sand = depthWeighted(
    'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02', 'Sand'
  );
  var physicalClay = depthWeighted(
    'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02', 'physical_Clay'
  );
  var silt = ee.Image(100).subtract(sand).subtract(physicalClay).rename('Silt');

  // IMPORTANT:
  // The original modelling dataset field named "Clay" is nearly
  // identical to pH-H2O × 10 and is not physical clay fraction.
  // To preserve the established model feature semantics,
  // the OpenLandMap pH-H2O encoded value is used and renamed "Clay".
  // OpenLandMap physical clay is used only to derive Silt.
  var clayModelProxy = depthWeighted(
    'OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02', 'Clay'
  ); // raw pH x 10 encoding: do NOT multiply by 0.1
  return ee.Image.cat([sand, silt, clayModelProxy]);
}

function assembleFeatureStack(landsat, region) {
  var featureImage = ee.Image.cat([
    buildClimate(), buildTerrain(), landsat.select('Albedo'), buildSoil(), buildPCA(landsat)
  ]).select([
    'MAT', 'MAP', 'PET', 'DEM', 'TWI', 'slope_deg', 'plan_cur', 'Albedo',
    'Sand', 'Silt', 'Clay', 'Vegetation_Factor1', 'Vegetation_Factor2'
  ]).toFloat();
  return featureImage.clip(region);
}

function buildPredictionFeatureStack(landsatQueryRegion, clipRegion) {
  return assembleFeatureStack(buildLandsat(landsatQueryRegion), clipRegion);
}

function predictSOCD(featureImage, classifier, region) {
  var predictionDepth = ee.Image.constant(PREDICTION_DEPTH_CM)
    .rename('True_Soil_Depth');
  return featureImage.addBands(predictionDepth).select(RF_FEATURES)
    .classify(classifier).rename('SOCD').max(0).toFloat().clip(region);
}

// --------------------- Albers pixel-grid tiling ---------------------
// Project the China boundary first, then align its bounds to integer 30 m
// coordinate edges.  getInfo() is intentionally used only for task-grid
// construction, as the export CRS WKT is already obtained client-side above.
var CHINA_PROJECTED = CHINA.transform(TARGET_PROJECTION, 1);
var CHINA_PROJECTED_BOUNDS = CHINA_PROJECTED.bounds(1, TARGET_PROJECTION);
var chinaBoundsRing = CHINA_PROJECTED_BOUNDS.coordinates().getInfo()[0];
var rawXMin = Math.min.apply(null, chinaBoundsRing.map(function(coord) { return coord[0]; }));
var rawXMax = Math.max.apply(null, chinaBoundsRing.map(function(coord) { return coord[0]; }));
var rawYMin = Math.min.apply(null, chinaBoundsRing.map(function(coord) { return coord[1]; }));
var rawYMax = Math.max.apply(null, chinaBoundsRing.map(function(coord) { return coord[1]; }));

var ALIGNED_XMIN = Math.floor(rawXMin / EXPORT_SCALE) * EXPORT_SCALE;
var ALIGNED_XMAX = Math.ceil(rawXMax / EXPORT_SCALE) * EXPORT_SCALE;
var ALIGNED_YMIN = Math.floor(rawYMin / EXPORT_SCALE) * EXPORT_SCALE;
var ALIGNED_YMAX = Math.ceil(rawYMax / EXPORT_SCALE) * EXPORT_SCALE;
var TOTAL_COLS = Math.round((ALIGNED_XMAX - ALIGNED_XMIN) / EXPORT_SCALE);
var TOTAL_ROWS = Math.round((ALIGNED_YMAX - ALIGNED_YMIN) / EXPORT_SCALE);

if (TOTAL_COLS <= 0 || TOTAL_ROWS <= 0) {
  throw new Error('Invalid aligned China bounding box; no 30 m pixels were generated.');
}

function pad2(number) {
  return number < 10 ? '0' + number : String(number);
}

function makeTile(tileId, rowIndex, colIndex, colStart, colEnd, rowStartNorth, rowEndNorth) {
  var xMin = ALIGNED_XMIN + colStart * EXPORT_SCALE;
  var xMax = ALIGNED_XMIN + colEnd * EXPORT_SCALE;
  var yMax = ALIGNED_YMAX - rowStartNorth * EXPORT_SCALE;
  var yMin = ALIGNED_YMAX - rowEndNorth * EXPORT_SCALE;
  return {
    id: tileId,
    row: rowIndex,
    col: colIndex,
    colStart: colStart,
    colEnd: colEnd,
    rowStartNorth: rowStartNorth,
    rowEndNorth: rowEndNorth,
    pixelWidth: colEnd - colStart,
    pixelHeight: rowEndNorth - rowStartNorth,
    xMin: xMin,
    xMax: xMax,
    yMin: yMin,
    yMax: yMax,
    geometry: ee.Geometry.Rectangle(
      [xMin, yMin, xMax, yMax], TARGET_PROJECTION, false
    )
  };
}

function splitTile2x2(parent) {
  var colMid = parent.colStart + Math.floor(parent.pixelWidth / 2);
  var rowMidNorth = parent.rowStartNorth + Math.floor(parent.pixelHeight / 2);
  if (colMid === parent.colStart || colMid === parent.colEnd ||
      rowMidNorth === parent.rowStartNorth || rowMidNorth === parent.rowEndNorth) {
    throw new Error('Tile ' + parent.id + ' is too small to subdivide on the 30 m grid.');
  }
  return {
    A: makeTile(parent.id + '_A', parent.row, parent.col,
      parent.colStart, colMid, parent.rowStartNorth, rowMidNorth), // north-west
    B: makeTile(parent.id + '_B', parent.row, parent.col,
      colMid, parent.colEnd, parent.rowStartNorth, rowMidNorth),   // north-east
    C: makeTile(parent.id + '_C', parent.row, parent.col,
      parent.colStart, colMid, rowMidNorth, parent.rowEndNorth),   // south-west
    D: makeTile(parent.id + '_D', parent.row, parent.col,
      colMid, parent.colEnd, rowMidNorth, parent.rowEndNorth)      // south-east
  };
}

var BASE_TILES = [];
for (var row = 0; row < EXPORT_GRID_ROWS; row++) {
  var rowStartNorth = Math.floor(row * TOTAL_ROWS / EXPORT_GRID_ROWS);
  var rowEndNorth = Math.floor((row + 1) * TOTAL_ROWS / EXPORT_GRID_ROWS);
  for (var col = 0; col < EXPORT_GRID_COLS; col++) {
    var colStart = Math.floor(col * TOTAL_COLS / EXPORT_GRID_COLS);
    var colEnd = Math.floor((col + 1) * TOTAL_COLS / EXPORT_GRID_COLS);
    var tileId = 'R' + pad2(row + 1) + 'C' + pad2(col + 1);
    BASE_TILES.push(makeTile(
      tileId, row + 1, col + 1, colStart, colEnd, rowStartNorth, rowEndNorth
    ));
  }
}

function findTileById(tileId) {
  var idParts = tileId.split('_');
  var baseId = idParts[0];
  var tile = null;
  BASE_TILES.forEach(function(candidate) {
    if (candidate.id === baseId) tile = candidate;
  });
  if (tile === null) {
    throw new Error('Unknown base tile ID: ' + baseId);
  }
  for (var partIndex = 1; partIndex < idParts.length; partIndex++) {
    var childCode = idParts[partIndex];
    var children = splitTile2x2(tile);
    if (!children.hasOwnProperty(childCode)) {
      throw new Error('Unknown subdivision child ID in ' + tileId + ': ' + childCode);
    }
    tile = children[childCode];
  }
  return tile;
}

function tileMetadata(tile) {
  return {
    tile_id: tile.id,
    row_north_to_south: tile.row,
    col_west_to_east: tile.col,
    pixel_width: tile.pixelWidth,
    pixel_height: tile.pixelHeight,
    xmin: tile.xMin,
    ymin: tile.yMin,
    xmax: tile.xMax,
    ymax: tile.yMax
  };
}

var BASE_TILE_GRID = ee.FeatureCollection(BASE_TILES.map(function(tile) {
  return ee.Feature(tile.geometry, tileMetadata(tile));
}));

var CANDIDATE_TILES;
if (SUBDIVIDE_PARENT_TILE !== null) {
  var parentTile = findTileById(SUBDIVIDE_PARENT_TILE);
  var parentChildren = splitTile2x2(parentTile);
  CANDIDATE_TILES = [parentChildren.A, parentChildren.B, parentChildren.C, parentChildren.D];
} else if (EXPORT_ALL_TILES) {
  CANDIDATE_TILES = BASE_TILES.slice();
} else {
  CANDIDATE_TILES = EXPORT_TILE_IDS.map(function(tileId) {
    return findTileById(tileId);
  });
}

if (CANDIDATE_TILES.length === 0) {
  throw new Error('No export tiles selected.');
}

// Calculate all candidate overlaps before any tile-specific Landsat query,
// feature-stack construction or Export task.  Area is evaluated in the CLCD
// Albers target projection.  This covers both base tiles and any recursively
// generated child tiles.
var CANDIDATE_TILES_WITH_OVERLAP = CANDIDATE_TILES.map(function(tile) {
  tile.clipRegion = tile.geometry.intersection(CHINA, 1, TARGET_PROJECTION);
  tile.overlapAreaM2 = tile.clipRegion.area(1, TARGET_PROJECTION);
  return tile;
});
var candidateTileOverlapFeatures = ee.FeatureCollection(
  CANDIDATE_TILES_WITH_OVERLAP.map(function(tile) {
    return ee.Feature(tile.geometry, tileMetadata(tile)).set(
      'overlap_area_m2', tile.overlapAreaM2
    );
  })
);
var nonEmptyTileIds = candidateTileOverlapFeatures
  .filter(ee.Filter.gt('overlap_area_m2', 0))
  .aggregate_array('tile_id')
  .getInfo();
var EXPORT_TILES = CANDIDATE_TILES_WITH_OVERLAP.filter(function(tile) {
  return nonEmptyTileIds.indexOf(tile.id) !== -1;
});

if (EXPORT_TILES.length === 0) {
  throw new Error('All selected tiles have zero overlap with China.');
}

// ---------------------------- Main ----------------------------
Map.centerObject(CHINA, 4);
Map.addLayer(CHINA, {color: 'FFFFFF'}, 'China boundary', true);
Map.addLayer(BASE_TILE_GRID.style({color: 'FF00FF', fillColor: '00000000', width: 1}), {},
  '30 m export tile grid (4 rows x 5 columns)', true);

print('Mode:', {
  EXPORT_SCALE: EXPORT_SCALE,
  TILE_SCALE: TILE_SCALE,
  EXPORT_GRID_ROWS: EXPORT_GRID_ROWS,
  EXPORT_GRID_COLS: EXPORT_GRID_COLS,
  EXPORT_FOLDER: EXPORT_FOLDER,
  EXPORT_ALL_TILES: EXPORT_ALL_TILES,
  EXPORT_TILE_IDS: EXPORT_TILE_IDS,
  SUBDIVIDE_PARENT_TILE: SUBDIVIDE_PARENT_TILE
});
print('CLCD-derived client target CRS:', TARGET_CRS);
print('Target projection:', TARGET_PROJECTION);
print('Target nominal scale:', TARGET_PROJECTION.nominalScale());
print('RF classifier Asset ID:', RF_CLASSIFIER_ASSET);
print('Projected China bounds before 30 m alignment:', {
  xmin: rawXMin, ymin: rawYMin, xmax: rawXMax, ymax: rawYMax
});
print('Aligned 30 m China task grid:', {
  xmin: ALIGNED_XMIN, ymin: ALIGNED_YMIN,
  xmax: ALIGNED_XMAX, ymax: ALIGNED_YMAX,
  total_cols: TOTAL_COLS, total_rows: TOTAL_ROWS
});
print('All base-tile metadata:', BASE_TILES.map(tileMetadata));
print('Candidate tile overlap metadata (m2):', candidateTileOverlapFeatures);
print('Selected export-tile metadata:', EXPORT_TILES.map(tileMetadata));

// The loaded immutable classifier is shared by every selected tile.  No
// nationwide training stack, sampleRegions call, or RF training graph remains
// in this production script.
var classifier = ee.Classifier.load(RF_CLASSIFIER_ASSET);

function exportTile(tile) {
  // tile.clipRegion is only the feature-stack and prediction clip region.
  // Landsat deliberately queries the simpler rectangular tile geometry.
  // The export region remains the exact rectangular 30 m task tile so adjacent
  // GeoTIFFs share boundaries without overlap or gap; outside-China pixels are
  // exported as the existing v3 NoData value.
  var featureImage = buildPredictionFeatureStack(tile.geometry, tile.clipRegion);
  var result = predictSOCD(featureImage, classifier, tile.clipRegion);
  var taskName = 'SOCD_RF_NATIONAL_30m_v3_' + tile.id;
  Export.image.toDrive({
    image: result.unmask(-9999).toFloat(),
    description: taskName,
    fileNamePrefix: taskName,
    folder: EXPORT_FOLDER,
    region: tile.geometry,
    crs: TARGET_CRS,
    scale: EXPORT_SCALE,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: -9999},
    shardSize: 256,
    fileDimensions: 8192,
    skipEmptyTiles: true
  });
}

EXPORT_TILES.forEach(exportTile);
print('Asset-classifier tiled production script constructed.  Code Editor created one task per selected tile; no task was auto-started.');
