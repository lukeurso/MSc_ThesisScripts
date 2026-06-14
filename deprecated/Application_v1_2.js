/**** ================================ create> feb 20, 2026
 * classifier application + results export TEST V1.2 (in active development as of march 3, 2026)
 * Combines RF_application_test_2_median preprocessing + classification
 * with table/image export products based on Psudo_Export note.   
 * No visualization layers are added
 * preprocesses, mosaics, applies RF, exports 
 * masks are hard coded for tuning  
 * this is a development version, comments are limited
 *  ================================ ****/

// ------------------------
// USER SETTINGS
// ------------------------
var APPLY_TO_WORKFLOW_MOSAICS = true;

var TRAIN_FC_ASSET = 'projects/vernal-signal-270100/assets/RF_TrainingData/syncPartial_modified_6_RF_training_stratified_1000_perAOIperClass';
var CLASS_PROPERTY = 'class';
var PRED_BANDS = ['B1','B2','B3','B4','B5','B6','B7','NDWI_ICE'];

var N_TREES = 300;
var VARS_PER_SPLIT = 5;
var MIN_LEAF = 10;
var BAG_FRACTION = 0.632;
var MAX_NODES = null;
var SEED = 123;

var NDWI_ICE_MIN = 0.1;

// Export settings
var EXPORT_TABLE_FOLDER = 'GEE_tables_v1_022626';
var EXPORT_IMAGE_FOLDER = 'GEE_tiffs_v1_022626';
var EXPORT_CRS = 'EPSG:3995';
var EXPORT_SCALE = 30;
var EXPORT_REGION = null; // defaults to AOI when null

// ------------------------
// WORKFLOW SETTINGS
// ------------------------
var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/OST_STUDYAREA_ROCKMASK_SIMPLE';
//var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/PTM_LONGAOI_WFJORDC_ROCKMASK';

var START_DATE = '2018-04-30';
var END_DATE   = '2018-10-30'; 

var MIN_SUN_ELEV     = 15;
var MAX_CLOUD_COVER  = null;  

var MOSAIC_LEN_DAYS = 02;
var MIN_SCENES_PER_WINDOW = 1;

var MEDOID_DISTANCE_BANDS = ['NDWI_ICE','SIWSI'];

// ------------------------
// CLOUD MASK TOGGLES (TOA)
// ------------------------
var USE_B9_CIRRUS = true;
var USE_QA_BITMASK = true;

var USE_QA_DILATED     = false;
var USE_QA_CLOUD_BIT   = true;
var USE_QA_SHADOW_BIT  = true;
var USE_QA_CLOUD_CONF  = true;
var USE_QA_SHADOW_CONF = true;
var USE_QA_CIRRUS_CONF = true;

var QA_CONF_LEVEL = 2;

var USE_QA_SHADOW_VECTOR      = true;
var SHADOW_PROJECT_DISTANCE_M = 10000;
var CIRRUS_B9_THRESH = 0.008;

var BUFFER_M = 15000;
var MIN_BAD_COMPONENT_PIXELS = 24;
var USE_DISTANCE_BUFFER = true;
var DIST_SCALE_M = 90;

// ------------------------
// MODULES
// ------------------------
var rock_mask = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_RockMask');

// ------------------------
// AOI + COLLECTION
// ------------------------
var AOI = ee.FeatureCollection(AOI_ASSET).geometry();
var exportRegion = AOI;
if (EXPORT_REGION !== null) {
  exportRegion = EXPORT_REGION;
}

var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
  .filterBounds(AOI)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.gt('SUN_ELEVATION', MIN_SUN_ELEV));

if (MAX_CLOUD_COVER !== null) {
  L8 = L8.filter(ee.Filter.lte('CLOUD_COVER', MAX_CLOUD_COVER));
}

print('Raw L8 TOA count after filters:', L8.size());

// ------------------------
// CLOUD MASK HELPERS
// ------------------------
function projectCloudShadowVectors(image, cloudMask) {
  image = ee.Image(image);
  cloudMask = ee.Image(cloudMask).gt(0);

  var sunAzDegrees = ee.Number(360).add(ee.Number(image.get('SUN_AZIMUTH')));
  var shadowAzDegrees = sunAzDegrees.add(180).mod(360);
  var shadowAzImgAxis = shadowAzDegrees.add(90).mod(360);

  var proj = image.select('B2').projection();
  var pixelSizeM = ee.Number(proj.nominalScale());
  var maxDistPx = ee.Number(SHADOW_PROJECT_DISTANCE_M).divide(pixelSizeM).ceil();

  var stretched = cloudMask
    .directionalDistanceTransform(shadowAzImgAxis, maxDistPx)
    .select('distance')
    .lte(maxDistPx)
    .or(cloudMask);

  return ee.Image(stretched).rename('bad_QA_shadow_vec').unmask(0);
}

function sieveBadComponents(maskImage) {
  var binaryMask = ee.Image(maskImage).gt(0).unmask(0);

  return binaryMask.updateMask(binaryMask)
    .connectedPixelCount(96, true)
    .gte(MIN_BAD_COMPONENT_PIXELS)
    .and(binaryMask)
    .unmask(0);
}

function detectClouds_QA_STRONG(image) {
  image = ee.Image(image);

  if (!USE_QA_BITMASK) {
    return ee.Image(0).rename('bad_QA');
  }

  var QA = image.select('QA_PIXEL');

  function conf2(startBit) {
    return QA.bitwiseAnd(ee.Number(3).leftShift(startBit))
             .rightShift(startBit);
  }

  var cloudConf  = conf2(8);
  var shadowConf = conf2(10);
  var cirrusConf = conf2(14);

  var cloudBit = QA.bitwiseAnd(1 << 3).neq(0);
  var shadowBit = QA.bitwiseAnd(1 << 4).neq(0);

  var b9CirrusBad = ee.Image(0);
  if (USE_B9_CIRRUS) {
    b9CirrusBad = image.select('B9').gt(CIRRUS_B9_THRESH);
  }

  var cloudBad  = cloudConf.gte(QA_CONF_LEVEL);
  var shadowBad = shadowConf.gte(QA_CONF_LEVEL);
  var cirrusBad = cirrusConf.gte(QA_CONF_LEVEL);

  var dilated = QA.bitwiseAnd(1 << 1).neq(0);

  var cloudSeed = ee.Image(0);
  if (USE_QA_CLOUD_BIT)   { cloudSeed = cloudSeed.or(cloudBit); }
  if (USE_QA_CLOUD_CONF)  { cloudSeed = cloudSeed.or(cloudBad); }
  if (USE_QA_CIRRUS_CONF) { cloudSeed = cloudSeed.or(cirrusBad); }
  if (USE_B9_CIRRUS)      { cloudSeed = cloudSeed.or(b9CirrusBad); }
  cloudSeed = cloudSeed.rename('bad_QA_cloud_seed').unmask(0);

  var cloudSeedSieved = sieveBadComponents(cloudSeed).rename('bad_QA_cloud_seed');

  var shadowVec = ee.Image(0);
  if (USE_QA_SHADOW_VECTOR) {
    shadowVec = projectCloudShadowVectors(image, cloudSeedSieved);
  }

  var bad = ee.Image(0);
  if (USE_QA_DILATED)       { bad = bad.or(dilated); }
  bad = bad.or(cloudSeedSieved);

  if (USE_QA_SHADOW_BIT)    { bad = bad.or(shadowBit); }
  if (USE_QA_SHADOW_CONF)   { bad = bad.or(shadowBad); }
  if (USE_QA_SHADOW_VECTOR) { bad = bad.or(shadowVec); }
  if (USE_QA_CIRRUS_CONF)   { bad = bad.or(cirrusBad); }
  if (USE_B9_CIRRUS)        { bad = bad.or(b9CirrusBad); }

  return sieveBadComponents(bad).rename('bad_QA');
}

function applyCloudMask_TOA(image) {
  image = ee.Image(image);

  var badQA = detectClouds_QA_STRONG(image);
  var bad = badQA.rename('bad').unmask(0);

  var proj30 = image.select('B2').projection();

  var outNoBuf = image.addBands([
    badQA,
    bad,
    bad.rename('bad_buf_zone'),
    bad.rename('bad_buf'),
    bad.not().rename('CLEAR')
  ]).updateMask(bad.not());

  var outBuf = (function() {
    var projCoarse = proj30.atScale(DIST_SCALE_M);
    var badCoarse = bad.reproject(projCoarse);

    var maxPx = ee.Number(BUFFER_M).divide(DIST_SCALE_M).ceil().add(2);
    var distPx = badCoarse.fastDistanceTransform(maxPx).sqrt();
    var distM = distPx.multiply(DIST_SCALE_M);
    var bufferZoneCoarse = distM.lte(BUFFER_M);

    var bufferZone = bufferZoneCoarse
      .reproject(proj30)
      .rename('bad_buf_zone');

    var badBuf = bad.or(bufferZone).rename('bad_buf').unmask(0);
    var clear = badBuf.not().rename('CLEAR');

    return image.addBands([badQA, bad, bufferZone, badBuf, clear])
                .updateMask(clear);
  })();

  return ee.Image(ee.Algorithms.If(USE_DISTANCE_BUFFER, outBuf, outNoBuf));
}

function preprocess(img) {
  img = ee.Image(img);
  var out = ee.Image(img.select(['B1','B2','B3','B4','B5','B6','B7','B9','B10','SAA','QA_PIXEL']))
    .copyProperties(img, img.propertyNames());
  out = ee.Image(out).clip(AOI);
  out = applyCloudMask_TOA(out);
  out = ee.Image(rock_mask.rock_mask(out));

  var clearMask = out.select('CLEAR');
  var ndwiIce = out.normalizedDifference(['B2','B4']).rename('NDWI_ICE');
  var siwsi = out.normalizedDifference(['B5','B6']).multiply(-1).rename('SIWSI');
  var quality = ndwiIce.updateMask(clearMask).rename('QUALITY');

  // Keep only bands required by medoid selection, classification, and exports.
  // This avoids carrying QA/intermediate bands through every 2-day mosaic.
  return out.addBands([ndwiIce, siwsi, quality])
    .select(['B1','B2','B3','B4','B5','B6','B7','NDWI_ICE','SIWSI','QUALITY','CLEAR']);
}

// ------------------------
// N-day mosaics (medoid)
// ------------------------
var mosaicMethodEE = ee.String('medoid');

function buildMedoidMosaic(winProcessed) {
  var medianVector = winProcessed.select(MEDOID_DISTANCE_BANDS).median();

  var withScore = winProcessed.map(function(img) {
    img = ee.Image(img);

    var sqDist = img.select(MEDOID_DISTANCE_BANDS)
      .subtract(medianVector)
      .pow(2)
      .reduce(ee.Reducer.sum())
      .rename('MEDOID_SQDIST');

    var medoidScore = sqDist.multiply(-1).rename('MEDOID_SCORE');
    return img.addBands([sqDist, medoidScore]);
  });

  return withScore.qualityMosaic('MEDOID_SCORE');
}

var start = ee.Date(START_DATE);
var end   = ee.Date(END_DATE);

var nWindows = end.difference(start, 'day').divide(MOSAIC_LEN_DAYS).ceil();

var mosaicIC = ee.ImageCollection(
  ee.List.sequence(0, nWindows.subtract(1)).map(function(i) {
    i = ee.Number(i);
    var wStart = start.advance(i.multiply(MOSAIC_LEN_DAYS), 'day');
    var wEnd   = wStart.advance(MOSAIC_LEN_DAYS, 'day');

    var winRaw = L8.filterDate(wStart, wEnd);
    var count = winRaw.size();

    var mosaic = ee.Image(ee.Algorithms.If(
      count.gte(MIN_SCENES_PER_WINDOW),
      buildMedoidMosaic(winRaw.map(preprocess))
        .set({
          'window_start': wStart.format('YYYY-MM-dd'),
          'window_end':   wEnd.format('YYYY-MM-dd'),
          'window_count': count,
          'mosaic_method': mosaicMethodEE
        }),
      ee.Image(0).updateMask(ee.Image(0)).set({
        'window_start': wStart.format('YYYY-MM-dd'),
        'window_end':   wEnd.format('YYYY-MM-dd'),
        'window_count': count,
        'mosaic_method': mosaicMethodEE
      })
    ));

    return mosaic;
  })
).filter(ee.Filter.gt('window_count', 0));

print('Mosaics (server count):', mosaicIC.size());

// ------------------------
// LEGACY INPUT IMAGES (fallback)
// ------------------------
var images = [
  ee.Image('projects/vernal-signal-270100/assets/SceneAssets/ValidationScenes/OST_PVI_LC08_023248_20140705')
];

var imageNames = [
  'Vi01_OST_LC08_023248_20140705'
];

var targetIC = ee.ImageCollection(ee.Algorithms.If(
  APPLY_TO_WORKFLOW_MOSAICS,
  mosaicIC,
  ee.ImageCollection.fromImages(images)
));

var firstImg = ee.Image(targetIC.first());
print('Applying RF to workflow mosaics:', APPLY_TO_WORKFLOW_MOSAICS);
print('First target image band names:', firstImg.bandNames());

// ------------------------
// TRAIN CLASSIFIER
// ------------------------
var trainingFC = ee.FeatureCollection(TRAIN_FC_ASSET);
var cleanedTraining = trainingFC.filter(ee.Filter.notNull(PRED_BANDS.concat([CLASS_PROPERTY])));

var withRand = cleanedTraining.randomColumn('rand', SEED);
var trainSet = withRand.filter(ee.Filter.lt('rand', 0.8));

var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: N_TREES,
  variablesPerSplit: VARS_PER_SPLIT,
  minLeafPopulation: MIN_LEAF,
  bagFraction: BAG_FRACTION,
  maxNodes: MAX_NODES,
  seed: SEED
}).train({
  features: trainSet,
  classProperty: CLASS_PROPERTY,
  inputProperties: PRED_BANDS
});

print('RF explain():', rf.explain());

// ------------------------
// CLASSIFY + EXPORT PRODUCTS
// ------------------------
function classifyForExport(img) {
  img = ee.Image(img);
  var predictors = img.select(PRED_BANDS)
    .updateMask(img.select('NDWI_ICE').gte(NDWI_ICE_MIN));

  var classified = predictors.classify(rf).rename('classification');
  // shift classes to 1-based values for export so valid classes are non-zero
  // and remain distinct from background/no-data (0).
  var classificationExport = classified.add(1).rename('classification_export');
  return img.addBands([classified, classificationExport]);
}

function buildStatsFeature(classifiedImage) {
  classifiedImage = ee.Image(classifiedImage);

  // 1-based export class band: water=1, slush=2, other=3.
  var classificationBand = classifiedImage.select('classification_export');
  var areaBand = ee.Image.pixelArea().rename('area_m2');

  // all area summaries in one reduceRegion pass to lower EECU usage.
  var areaSummary = ee.Image.cat([
    areaBand.updateMask(classificationBand.eq(1)).rename('water_area_m2'),
    areaBand.updateMask(classificationBand.eq(2)).rename('slush_area_m2'),
    areaBand.updateMask(classificationBand.eq(3)).rename('other_area_m2'),
    areaBand.updateMask(classifiedImage.select('B1').mask()).rename('unmasked_area_m2'),
    areaBand.updateMask(classifiedImage.select('NDWI_ICE').gte(NDWI_ICE_MIN)).rename('ndwi_area_m2')
  ]).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
    crs: EXPORT_CRS,
    scale: EXPORT_SCALE,
    tileScale: 4,
    maxPixels: 1e13
  });

  var startDate = ee.String(ee.Algorithms.If(
    classifiedImage.propertyNames().contains('window_start'),
    classifiedImage.get('window_start'),
    START_DATE
  ));

  var endDate = ee.String(ee.Algorithms.If(
    classifiedImage.propertyNames().contains('window_end'),
    classifiedImage.get('window_end'),
    END_DATE
  ));

  var imageCount = ee.Number(ee.Algorithms.If(
    classifiedImage.propertyNames().contains('window_count'),
    classifiedImage.get('window_count'),
    1
  ));

  var exportId = ee.String(ee.Algorithms.If(
    classifiedImage.propertyNames().contains('export_id'),
    classifiedImage.get('export_id'),
    'unlabeled_image'
  ));

  return ee.Feature(null, {
    Export_ID: exportId,
    Start_Date: startDate,
    End_Date: endDate,
    Water_Area_m2: areaSummary.get('water_area_m2'),
    Slush_Area_m2: areaSummary.get('slush_area_m2'),
    Other_Area_m2: areaSummary.get('other_area_m2'),
    Unmasked_Area_m2: areaSummary.get('unmasked_area_m2'),
    NDWI_Area_m2: areaSummary.get('ndwi_area_m2'),
    Number_Images: imageCount
  });
}

function queueImageExports(classifiedImage, exportId) {

  Export.image.toDrive({
    image: classifiedImage.select('classification_export').toInt8(),
    description: exportId + '_classified',
    fileNamePrefix: exportId + '_classified',
    folder: EXPORT_IMAGE_FOLDER,
    scale: EXPORT_SCALE,
    crs: EXPORT_CRS,
    region: exportRegion,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: classifiedImage.select(['B4','B3','B2']),
    description: exportId + '_base',
    fileNamePrefix: exportId + '_base',
    folder: EXPORT_IMAGE_FOLDER,
    scale: EXPORT_SCALE,
    crs: EXPORT_CRS,
    region: exportRegion,
    maxPixels: 1e13
  });
}

var targetClassifiedIC = targetIC.map(classifyForExport);
var statsSelectors = [
  'Export_ID',
  'Start_Date',
  'End_Date',
  'Slush_Area_m2',
  'Water_Area_m2',
  'Other_Area_m2',
  'Unmasked_Area_m2',
  'NDWI_Area_m2',
  'Number_Images'
];

function queueCombinedStatsExport(statsCollection, exportIdPrefix) {
  Export.table.toDrive({
    collection: statsCollection,
    description: exportIdPrefix + '_stats_combined',
    fileNamePrefix: exportIdPrefix + '_stats_combined',
    fileFormat: 'CSV',
    folder: EXPORT_TABLE_FOLDER,
    selectors: statsSelectors
  });
}

if (APPLY_TO_WORKFLOW_MOSAICS) {
  var mosaicClassifiedList = targetClassifiedIC.sort('window_start').toList(targetClassifiedIC.size());
  var workflowStatsFC = ee.FeatureCollection(targetClassifiedIC.map(function(img) {
    img = ee.Image(img);
    var startStr = ee.String(img.get('window_start'));
    var endStr = ee.String(img.get('window_end'));
    var exportId = startStr.cat('_').cat(endStr);
    return buildStatsFeature(img.set('export_id', exportId));
  }));

  queueCombinedStatsExport(workflowStatsFC, 'workflow_mosaics');

  targetClassifiedIC.size().evaluate(function(nClient) {
    print('Queueing export tasks for classified mosaics:', nClient);

    for (var i = 0; i < nClient; i++) {
      var img = ee.Image(mosaicClassifiedList.get(i));
      var startStr = ee.String(img.get('window_start')).getInfo();
      var endStr = ee.String(img.get('window_end')).getInfo();
      var exportId = startStr + '_' + endStr;

      queueImageExports(img, exportId);
    }
  });
} else {
  var legacyStatsFeatures = images.map(function(img, i) {
    var classifiedImage = classifyForExport(img).set('export_id', imageNames[i]);
    return buildStatsFeature(classifiedImage);
  });

  queueCombinedStatsExport(ee.FeatureCollection(legacyStatsFeatures), 'legacy_images');

  images.forEach(function(img, i) {
    var classifiedImage = classifyForExport(img);
    queueImageExports(classifiedImage, imageNames[i]);
  });
}
