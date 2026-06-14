/**** ================================
 *  EXPORT v5, adds scene-count limiting per window from test_medoids_v3.
 *  Parent script = export_v4. 
 *
 *  V5 change: before any pixel preprocessing, each raw scene in a mosaic
 *  window is scored by its footprint intersection area with the AOI (a cheap
 *  vector/metadata operation).  Only the top MAX_SCENES_PER_WINDOW scenes
 *  are then preprocessed, directly reducing per-window compute cost.
 *  Set MAX_SCENES_PER_WINDOW = null to disable the limit (v4 behaviour).
 *
 *  window_count is now split into:
 *    window_count_raw  — total scenes available in the date window
 *    window_count_used — scenes actually preprocessed (after limiting)
 *
 *  application test of classifier
 *  preprocesses scenes with masks, clip, adds NDWI_ice band
 *  generates medoid mosaics from NDWI_ice
 *  exports results in combined table and per window raster classes and mosaics
 *  no visualization layers are added
 *  ================================ ****/

// ------------------------
// USER SETTINGS
// ------------------------
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
var EXPORT_TABLE_FOLDER = 'GEE_tables_v1_03042026';
var EXPORT_IMAGE_FOLDER = 'GEE_tiffs_v1_03042026';
var EXPORT_CRS = 'EPSG:3995';
var EXPORT_SCALE = 30;
var EXPORT_REGION = null; // defaults to AOI when null

// ------------------------
// WORKFLOW SETTINGS
// ------------------------

var AOI_DESIGNATION = 'PTM'; // set to 'PTM' or 'OST'

//1800m AOIs
//var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/OST_STUDYAREA_ROCKMASK_SIMPLE';
//var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/PTM_LONGAOI_WFJORDC_ROCKMASK';

//1500m AOIs
//var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/OST_AOI_1500m';
var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/PTM_AOI_1500m';


var START_DATE = '2024-07-30';
var END_DATE   = '2024-08-06';

var EXPORT_TABLE_PREFIX = AOI_DESIGNATION + '_' + START_DATE + '_' + END_DATE;

var MIN_SUN_ELEV     = 15;
var MAX_CLOUD_COVER  = null;

var MOSAIC_LEN_DAYS = 02;
var MIN_SCENES_PER_WINDOW = 1;

// Maximum number of scenes to preprocess per mosaic window.
// Raw scenes are ranked by their footprint intersection area with the AOI;
// only the top MAX_SCENES_PER_WINDOW are preprocessed (cloud-masked,
// rock-masked, and index-computed).  Set to null to disable the limit
// and reproduce v4 behaviour.
var MAX_SCENES_PER_WINDOW = 8;

var MEDOID_DISTANCE_BANDS = ['NDWI_ICE'];

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

var BUFFER_M = 5000;
var MIN_BAD_COMPONENT_PIXELS = 24;
var USE_DISTANCE_BUFFER = true;
var DIST_SCALE_M = 30;

// ------------------------
// MODULES
// ------------------------
var rock_mask = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_RockMask');
var cloud_mask = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_CloudMask_TOA');

// ------------------------
// AOI + COLLECTION
// ------------------------
var AOI = ee.FeatureCollection(AOI_ASSET).geometry();
var exportRegion = AOI;
if (EXPORT_REGION !== null) {
  exportRegion = EXPORT_REGION;
}

function getFilteredCollection(collectionId) {
  var filtered = ee.ImageCollection(collectionId)
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.gt('SUN_ELEVATION', MIN_SUN_ELEV));

  if (MAX_CLOUD_COVER !== null) {
    filtered = filtered.filter(ee.Filter.lte('CLOUD_COVER', MAX_CLOUD_COVER));
  }

  return filtered;
}

var L8 = getFilteredCollection('LANDSAT/LC08/C02/T1_TOA');
var L9 = getFilteredCollection('LANDSAT/LC09/C02/T1_TOA');

var landsatTOA = ee.ImageCollection(L8.merge(L9));

print('Raw L8 TOA count after filters:', L8.size());
print('Raw L9 TOA count after filters:', L9.size());
print('Raw merged Landsat TOA count after filters:', landsatTOA.size());

// ------------------------
// CLOUD MASK CONFIG
// ------------------------
var CLOUD_MASK_CFG = {
  USE_B9_CIRRUS: USE_B9_CIRRUS,
  USE_QA_BITMASK: USE_QA_BITMASK,
  USE_QA_DILATED: USE_QA_DILATED,
  USE_QA_CLOUD_BIT: USE_QA_CLOUD_BIT,
  USE_QA_SHADOW_BIT: USE_QA_SHADOW_BIT,
  USE_QA_CLOUD_CONF: USE_QA_CLOUD_CONF,
  USE_QA_SHADOW_CONF: USE_QA_SHADOW_CONF,
  USE_QA_CIRRUS_CONF: USE_QA_CIRRUS_CONF,
  QA_CONF_LEVEL: QA_CONF_LEVEL,
  USE_QA_SHADOW_VECTOR: USE_QA_SHADOW_VECTOR,
  SHADOW_PROJECT_DISTANCE_M: SHADOW_PROJECT_DISTANCE_M,
  CIRRUS_B9_THRESH: CIRRUS_B9_THRESH,
  BUFFER_M: BUFFER_M,
  MIN_BAD_COMPONENT_PIXELS: MIN_BAD_COMPONENT_PIXELS,
  USE_DISTANCE_BUFFER: USE_DISTANCE_BUFFER,
  DIST_SCALE_M: DIST_SCALE_M
};

// ------------------------
// SCENE SCORING + LIMITING
// ------------------------
// Score each scene in a raw window collection by the area of its footprint
// that overlaps the AOI.  Scenes are sorted best-first and then limited to
// MAX_SCENES_PER_WINDOW before any pixel-level preprocessing is performed.
// If MAX_SCENES_PER_WINDOW is null the collection is returned unchanged.
function limitScenesByAOIIntersect(winRaw) {
  if (MAX_SCENES_PER_WINDOW === null) {
    return winRaw;
  }

  // Compute overlap area for each scene using its swath footprint geometry.
  // This is a vector / metadata operation — no pixel data is loaded.
  var scored = winRaw.map(function(img) {
    var overlapArea = img.geometry()
      .intersection(AOI, ee.ErrorMargin(30))
      .area(ee.ErrorMargin(30));
    return img.set('aoi_intersect_area', overlapArea);
  });

  // Sort descending (largest overlap first) and take the top N.
  return scored.sort('aoi_intersect_area', false).limit(MAX_SCENES_PER_WINDOW);
}

function preprocess(img) {
  img = ee.Image(img);
  var out = ee.Image(img.select(['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','QA_PIXEL']))
    .copyProperties(img, img.propertyNames());
  out = ee.Image(out).clip(AOI);
  out = ee.Image(cloud_mask.applyCloudMask_TOA(out, CLOUD_MASK_CFG));
  out = ee.Image(rock_mask.rock_mask(out));

  var ndwiIce = out.normalizedDifference(['B2','B4']).rename('NDWI_ICE');

  // Keep only bands required downstream for RF predictors and export products.
  return out.select(['B1','B2','B3','B4','B5','B6','B7','B8'])
    .addBands(ndwiIce);
}

// ------------------------
// N-day mosaics (medoid)
// ------------------------
function buildMedoidMosaic(winProcessed) {
  var medianVector = winProcessed.select(MEDOID_DISTANCE_BANDS).median();

  var withScore = winProcessed.map(function(img) {
    img = ee.Image(img);

    var medoidScore = img.select(MEDOID_DISTANCE_BANDS)
      .subtract(medianVector)
      .pow(2)
      .reduce(ee.Reducer.sum())
      .multiply(-1)
      .rename('MEDOID_SCORE');
    return img.addBands(medoidScore);
  });

  return withScore.qualityMosaic('MEDOID_SCORE')
    .select(['B1','B2','B3','B4','B5','B6','B7','B8','NDWI_ICE']);
}

var start = ee.Date(START_DATE);
var end   = ee.Date(END_DATE);

var nWindows = end.difference(start, 'day').divide(MOSAIC_LEN_DAYS).ceil();

var mosaicIC = ee.ImageCollection(
  ee.List.sequence(0, nWindows.subtract(1)).map(function(i) {
    i = ee.Number(i);
    var wStart = start.advance(i.multiply(MOSAIC_LEN_DAYS), 'day');
    var wEnd   = wStart.advance(MOSAIC_LEN_DAYS, 'day');

    // Raw scenes for this window — used for the MIN_SCENES count gate only.
    var winRaw  = landsatTOA.filterDate(wStart, wEnd);
    var countRaw = winRaw.size();

    // Score raw scenes by AOI overlap and limit to MAX_SCENES_PER_WINDOW
    // BEFORE any pixel preprocessing occurs.
    var winLimited = limitScenesByAOIIntersect(winRaw);

    var mosaic = ee.Image(ee.Algorithms.If(
      countRaw.gte(MIN_SCENES_PER_WINDOW),
      buildMedoidMosaic(winLimited.map(preprocess))
        .set({
          'window_start':       wStart.format('YYYY-MM-dd'),
          'window_end':         wEnd.format('YYYY-MM-dd'),
          'window_count_raw':   countRaw,
          'window_count_used':  winLimited.size(),
          'mosaic_method':      'medoid'
        }),
      ee.Image(0).updateMask(ee.Image(0)).set({
        'window_start':       wStart.format('YYYY-MM-dd'),
        'window_end':         wEnd.format('YYYY-MM-dd'),
        'window_count_raw':   countRaw,
        'window_count_used':  ee.Number(0),
        'mosaic_method':      'medoid'
      })
    ));

    return mosaic;
  })
).filter(ee.Filter.gt('window_count_raw', 0));

print('Mosaics (server count):', mosaicIC.size());

var firstImg = ee.Image(mosaicIC.first());
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
  // Shift classes to 1-based values for export so valid classes are non-zero
  // and remain distinct from background/no-data (0).
  var classificationExport = classified.add(1).rename('classification_export');
  return img.addBands([classified, classificationExport]);
}

function buildStatsFeature(classifiedImage) {
  classifiedImage = ee.Image(classifiedImage);

  // Use the 1-based export class band: water=1, slush=2, other=3.
  var classificationBand = classifiedImage.select('classification_export');
  var areaBand = ee.Image.pixelArea().rename('area_m2');

  var waterMask = areaBand.updateMask(classificationBand.eq(1));
  var slushMask = areaBand.updateMask(classificationBand.eq(2));
  var otherMask = areaBand.updateMask(classificationBand.eq(3));

  var unmaskedMask = areaBand.updateMask(classifiedImage.select('B1').mask());
  var ndwiMask = areaBand.updateMask(classifiedImage.select('NDWI_ICE').gte(NDWI_ICE_MIN));

  var areaStack = ee.Image.cat([
    waterMask.rename('water_area_m2'),
    slushMask.rename('slush_area_m2'),
    otherMask.rename('other_area_m2'),
    unmaskedMask.rename('unmasked_area_m2'),
    ndwiMask.rename('ndwi_area_m2')
  ]);

  var areaStats = areaStack.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
    crs: EXPORT_CRS,
    scale: EXPORT_SCALE,
    tileScale: 4,
    maxPixels: 1e13
  });

  var startDate    = ee.String(classifiedImage.get('window_start'));
  var endDate      = ee.String(classifiedImage.get('window_end'));
  var countRaw     = ee.Number(classifiedImage.get('window_count_raw'));
  var countUsed    = ee.Number(classifiedImage.get('window_count_used'));
  var exportId     = ee.String(classifiedImage.get('export_id'));

  return ee.Feature(null, {
    Export_ID:        exportId,
    Start_Date:       startDate,
    End_Date:         endDate,
    Water_Area_m2:    areaStats.get('water_area_m2'),
    Slush_Area_m2:    areaStats.get('slush_area_m2'),
    Other_Area_m2:    areaStats.get('other_area_m2'),
    Unmasked_Area_m2: areaStats.get('unmasked_area_m2'),
    NDWI_Area_m2:     areaStats.get('ndwi_area_m2'),
    Number_Images_Raw:  countRaw,
    Number_Images_Used: countUsed
  });
}

function queueAssetExport(classifiedImage, exportId, windowStart, windowEnd, countRaw, countUsed) {

  // Build output: classification + B4/B8, retaining lake pixels + 1-pixel ring around lakes.
  var classExport = classifiedImage.select('classification_export').toInt8();

  // Lake class = 1 in 1-based export band.
  var lakeMask = classExport.eq(1);

  // Include any non-lake pixel that touches a lake pixel (1-pixel ring at native scale).
  var ringMask = lakeMask.focalMax({radius: 1, units: 'pixels'});

  var b4_ring = classifiedImage.select('B4').updateMask(ringMask).toFloat();
  var b8_ring = classifiedImage.select('B8').updateMask(ringMask).toFloat();

  // Binary footprint: 1 wherever the mosaic had at least one valid (unmasked)
  // observation. Stored as uint8 so it is compact. Masked (absent) outside the
  // mosaic's valid-data extent, allowing downstream code to distinguish
  // "observed non-lake" (footprint=1, class≠water) from "no data" (footprint
  // absent) when checking for lake drainage between consecutive mosaics.
  var footprintBand = classifiedImage.select('B1').mask().toByte()
      .rename('mosaic_footprint');

  var outImage = ee.Image.cat([
      classExport,
      b4_ring,
      b8_ring,
      footprintBand
    ])
    .rename([
      'classification',
      'B4_ring',
      'B8_ring',
      'mosaic_footprint'
    ])
    .set({
      'system:time_start':  ee.Date(windowStart).millis(),
      'system:time_end':    ee.Date(windowEnd).millis(),
      'window_start':       windowStart,
      'window_end':         windowEnd,
      'window_count_raw':   countRaw,
      'window_count_used':  countUsed,
      'aoi':                AOI_DESIGNATION,
      'export_id':          exportId
    });

  Export.image.toAsset({
    image: outImage,
    description: exportId + '_asset',
    assetId: 'projects/vernal-signal-270100/assets/ClassifiedMosaics/' + AOI_DESIGNATION + '/' + exportId,
    scale: EXPORT_SCALE,
    crs: EXPORT_CRS,
    region: exportRegion,
    pyramidingPolicy: {
      classification:   'mode',
      B4_ring:          'mean',
      B8_ring:          'mean',
      mosaic_footprint: 'mode'
    },
    maxPixels: 1e13
  });
}

var statsSelectors = [
  'Export_ID',
  'Start_Date',
  'End_Date',
  'Slush_Area_m2',
  'Water_Area_m2',
  'Other_Area_m2',
  'Unmasked_Area_m2',
  'NDWI_Area_m2',
  'Number_Images_Raw',
  'Number_Images_Used'
];

function queueCombinedStatsExport(statsCollection, exportIdPrefix) {
  var csvExportId = EXPORT_TABLE_PREFIX + '_' + exportIdPrefix + '_stats_combined';

  Export.table.toDrive({
    collection: statsCollection,
    description: csvExportId,
    fileNamePrefix: csvExportId,
    fileFormat: 'CSV',
    folder: EXPORT_TABLE_FOLDER,
    selectors: statsSelectors
  });
}

// Classify and assign export_id in a single map pass.
var workflowClassifiedIC = mosaicIC.map(function(img) {
  img = ee.Image(img);
  var classified = classifyForExport(img);
  var startStr = ee.String(img.get('window_start'));
  var endStr   = ee.String(img.get('window_end'));
  var exportId = ee.String(AOI_DESIGNATION).cat('_').cat(startStr).cat('_').cat(endStr);
  return classified.set('export_id', exportId);
});

var workflowStatsFC = ee.FeatureCollection(workflowClassifiedIC.map(function(img) {
  return buildStatsFeature(ee.Image(img));
}));

queueCombinedStatsExport(workflowStatsFC, 'workflow_mosaics');

var workflowMeta = ee.Dictionary({
  export_ids:          workflowClassifiedIC.aggregate_array('export_id'),
  window_starts:       workflowClassifiedIC.aggregate_array('window_start'),
  window_ends:         workflowClassifiedIC.aggregate_array('window_end'),
  window_counts_raw:   workflowClassifiedIC.aggregate_array('window_count_raw'),
  window_counts_used:  workflowClassifiedIC.aggregate_array('window_count_used')
});

workflowMeta.evaluate(function(meta) {
  var exportIds     = meta.export_ids        || [];
  var windowStarts  = meta.window_starts     || [];
  var windowEnds    = meta.window_ends       || [];
  var countsRaw     = meta.window_counts_raw  || [];
  var countsUsed    = meta.window_counts_used || [];

  print('Queueing asset export tasks:', exportIds.length);

  // Build the list once so each image is retrieved by index, not by re-filtering N times.
  var imgList = workflowClassifiedIC.toList(exportIds.length);
  for (var i = 0; i < exportIds.length; i++) {
    var img = ee.Image(imgList.get(i));
    queueAssetExport(img, exportIds[i], windowStarts[i], windowEnds[i], countsRaw[i], countsUsed[i]);
  }
});
