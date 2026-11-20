// ============================================================
// Export frozen National 0-30 cm SOCD RF v3 classifier as an EE Asset.
// No SOCD raster Export is created by this script.
// ============================================================

// The Earth Engine asset IDs below are from the original project and require access
// to that project; they are retained to document the production dependency chain.
var TRAIN_SAMPLE_SCALE = 30;
var TILE_SCALE = 16;
var START_DATE = '2012-01-01';
var END_DATE = '2024-01-01';
var YEARS = 12;

var RF_CLASSIFIER_ASSET =
  'projects/qingdao-coastline-changes/assets/SOCD_RF_v3_classifier';

var CHINA = ee.FeatureCollection(
  'projects/qingdao-coastline-changes/assets/China_Boundary_Standard'
).geometry();

// Keep the server-side projection for sampleRegions.  Its client-side WKT is
// used only as Export's CRS string; CLCD does not expose a stable EPSG code.
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

function buildTrainingFeatureStack() {
  return assembleFeatureStack(buildLandsat(CHINA), CHINA);
}

function buildTrainingSamples(featureImage) {
  var raw = ee.FeatureCollection(
    'projects/qingdao-coastline-changes/assets/china_all_features'
  );
  var sampled = featureImage.sampleRegions({
    collection: raw,
    properties: ['SOCD__kg_m', '真实土层厚度'],
    scale: TRAIN_SAMPLE_SCALE,
    projection: TARGET_PROJECTION,
    tileScale: TILE_SCALE
  }).map(function(feature) {
    return feature.set('True_Soil_Depth', feature.get('真实土层厚度'));
  });
  var requiredFields = RF_FEATURES.concat(['SOCD__kg_m']);
  var nonNull = sampled.filter(ee.Filter.notNull(requiredFields));
  var training = nonNull.filter(ee.Filter.gte('SOCD__kg_m', 0));
  return training;
}

function trainRF(training) {
  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 200,
    variablesPerSplit: 6,
    minLeafPopulation: 5,
    maxNodes: 16384,
    seed: 42
  }).setOutputMode('REGRESSION').train({
    features: training,
    classProperty: 'SOCD__kg_m',
    inputProperties: RF_FEATURES
  });
  return classifier;
}

var trainingFeatureImage = buildTrainingFeatureStack();
print('Final training environmental feature bands:', trainingFeatureImage.bandNames());
var training = buildTrainingSamples(trainingFeatureImage);
var classifier = trainRF(training);

print('RF_FEATURES:', RF_FEATURES);
print('RF classifier Asset ID:', RF_CLASSIFIER_ASSET);

Export.classifier.toAsset({
  classifier: classifier,
  description: 'SOCD_RF_v3_classifier_export',
  assetId: RF_CLASSIFIER_ASSET
});

print('Classifier export task constructed; manually start it in the Tasks tab.');
