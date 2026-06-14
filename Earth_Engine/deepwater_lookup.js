/****
 *deepwater_lookup.js
 *build and export deep-water Rinfity lookup tables for Landsat 8/9 TOA.
 *based on Moussavi et al 2020.  
 ****/

// =========================
// 1) USER SETTINGS
// =========================
var AOI_OST_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/DeepWater/OST_DEEPWATER';
var AOI_PTM_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/DeepWater/PTM_DEEPWATER';

var START_DATE = '2013-01-01';
var END_DATE = ee.Date(Date.now());
var MIN_SUN_ELEV = 15; // degrees

var LC08_COLLECTION = 'LANDSAT/LC08/C02/T1_TOA';
var LC09_COLLECTION = 'LANDSAT/LC09/C02/T1_TOA';

var RED_BAND = 'B4';   // 30 m
var PAN_BAND = 'B8';   // 15 m

var RINF_PCTL = 5;     // 5th percentile
var RINF_MAX = 0.1;
var MIN_WATER_PX = 200;
var DARK_RED_MAX = 0.08;
var DARK_PAN_MAX = 0.10;

var DEBUG = false;
var PRINT_SUMMARY = false;

// Export destinations (user editable).
var EXPORT_ASSET_OST = 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_OST_L8L9_v1';
var EXPORT_ASSET_PTM = 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_PTM_L8L9_v1';

// =========================
// 2) AOIs + COLLECTION PREP
// =========================
var aoiOST = ee.FeatureCollection(AOI_OST_ASSET);
var aoiPTM = ee.FeatureCollection(AOI_PTM_ASSET);

var lc08 = ee.ImageCollection(LC08_COLLECTION);
var lc09 = ee.ImageCollection(LC09_COLLECTION);

// =========================
// 3) HELPERS
// =========================

/** Returns true image if QA_PIXEL exists in image bands. */
function hasQaPixel(image) {
  return image.bandNames().contains('QA_PIXEL');
}

/** Cloud/shadow mask from QA_PIXEL for Landsat C2 when available. */
function qaCloudShadowMask(image) {
  var qa = image.select('QA_PIXEL');
  var cloudBit = 1 << 3;       // cloud
  var cloudShadowBit = 1 << 4; // cloud shadow
  var mask = qa.bitwiseAnd(cloudBit).eq(0)
    .and(qa.bitwiseAnd(cloudShadowBit).eq(0));
  return mask;
}

/**
 * Build deep-water candidate mask:
 * dark guards + optional QA cloud/shadow filtering.
 */
function buildDeepMask(image) {
  var darkMask = image.select(RED_BAND).lt(DARK_RED_MAX)
    .and(image.select(PAN_BAND).lt(DARK_PAN_MAX));

  var deepMask = ee.Image(
    ee.Algorithms.If(
      hasQaPixel(image),
      darkMask.and(qaCloudShadowMask(image)),
      darkMask
    )
  );

  return deepMask.rename('deepMask');
}

/**
 * Compute one LUT row (Feature) per image for a given AOI and sensor label.
 */
function imageToLutFeature(image, aoiFc, aoiId, sensorLabel, aoiLon, aoiLat, featureGeom) {
  var geom = aoiFc.geometry();
  var deepMask = buildDeepMask(image);

  // Compute both red/pan percentiles in one reduction to lower server aggregation load.
  var stats = image.select([RED_BAND, PAN_BAND]).updateMask(deepMask).reduceRegion({
    reducer: ee.Reducer.percentile([RINF_PCTL]).combine({
      reducer2: ee.Reducer.count(),
      sharedInputs: true
    }),
    geometry: geom,
    scale: 30,
    bestEffort: true,
    maxPixels: 1e8,
    tileScale: 4
  });

  var redKey = RED_BAND + '_p' + RINF_PCTL;
  var countKey = RED_BAND + '_count';
  var panKey = PAN_BAND + '_p' + RINF_PCTL;

  function dictGetOrDefault(dict, key, defaultValue) {
    dict = ee.Dictionary(dict);
    var valueOrDefault = ee.Algorithms.If(dict.contains(key), dict.get(key), defaultValue);
    return ee.Algorithms.If(ee.Algorithms.IsEqual(valueOrDefault, null), defaultValue, valueOrDefault);
  }

  function dictGetNumberOrDefault(dict, key, defaultValue) {
    return ee.Number(dictGetOrDefault(dict, key, defaultValue));
  }

  // Keep raw values for output schema (can be null if no valid pixels).
  var rinfRed = dictGetOrDefault(stats, redKey, null);
  var nWater = dictGetOrDefault(stats, countKey, null);
  var rinfPan = dictGetOrDefault(stats, panKey, null);

  // Safe values for QC math; defaults force QC failure when stats are missing.
  var rinfRedSafe = dictGetNumberOrDefault(stats, redKey, 999);
  var nWaterSafe = dictGetNumberOrDefault(stats, countKey, 0);
  var rinfPanSafe = dictGetNumberOrDefault(stats, panKey, 999);

  var passQc = nWaterSafe.gte(MIN_WATER_PX)
    .and(rinfRedSafe.lt(RINF_MAX))
    .and(rinfPanSafe.lt(RINF_MAX));

  var feat = ee.Feature(featureGeom, {
    'system:time_start': image.get('system:time_start'),
    'system:index': image.get('system:index'),
    'LANDSAT_PRODUCT_ID': image.get('LANDSAT_PRODUCT_ID'),
    'sensor': sensorLabel,
    'aoi_id': aoiId,
    'Rinf_red': rinfRed,
    'Rinf_pan': rinfPan,
    'nWater': nWater,
    'ok': ee.Number(ee.Algorithms.If(passQc, 1, 0)),
    'SUN_ELEVATION': image.get('SUN_ELEVATION'),
    'RINF_PCTL': RINF_PCTL,
    'aoi_lon': aoiLon,
    'aoi_lat': aoiLat
  });

  return feat;
}

/** Build LUT for one AOI from LC08 + LC09. */
function buildLutForAoi(aoiFc, aoiId) {
  var geom = aoiFc.geometry();
  var centroid = geom.centroid(1);
  var featureGeom = ee.Feature(centroid).geometry();
  var lonLat = ee.List(centroid.coordinates());
  var aoiLon = lonLat.get(0);
  var aoiLat = lonLat.get(1);

  var col08 = lc08
    .filterBounds(geom)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.gt('SUN_ELEVATION', MIN_SUN_ELEV));

  var col09 = lc09
    .filterBounds(geom)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.gt('SUN_ELEVATION', MIN_SUN_ELEV));

  var fc08 = ee.FeatureCollection(col08.map(function(img) {
    return imageToLutFeature(img, aoiFc, aoiId, 'LC08', aoiLon, aoiLat, featureGeom);
  }));

  var fc09 = ee.FeatureCollection(col09.map(function(img) {
    return imageToLutFeature(img, aoiFc, aoiId, 'LC09', aoiLon, aoiLat, featureGeom);
  }));

  return fc08.merge(fc09).sort('system:time_start');
}

/** Print summary stats and hist-like aggregates for QC-passing rows. */
function printLutSummary(lut, label) {
  var okLut = lut.filter(ee.Filter.eq('ok', 1));

  print(label + ' total rows:', lut.size());
  print(label + ' rows passing QC (ok==1):', okLut.size());

  var statsReducer = ee.Reducer.min()
    .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.mean(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.percentile([5, 25, 50, 75, 95]), sharedInputs: true});

  print(label + ' Rinf_red stats (ok==1):', okLut.reduceColumns({
    reducer: statsReducer,
    selectors: ['Rinf_red']
  }));

  print(label + ' Rinf_pan stats (ok==1):', okLut.reduceColumns({
    reducer: statsReducer,
    selectors: ['Rinf_pan']
  }));

  print(label + ' Rinf_red histogram (ok==1):', okLut.aggregate_histogram('Rinf_red'));
  print(label + ' Rinf_pan histogram (ok==1):', okLut.aggregate_histogram('Rinf_pan'));

  if (DEBUG) {
    print(label + ' sample rows:', okLut.limit(5));
  }
}

// =========================
// 4) BUILD LUTS
// =========================
var lutOST = buildLutForAoi(aoiOST, 'OST_DEEPWATER');
var lutPTM = buildLutForAoi(aoiPTM, 'PTM_DEEPWATER');

// =========================
// 5) REPORTING
// =========================
if (PRINT_SUMMARY) {
  printLutSummary(lutOST, 'OST LUT');
  printLutSummary(lutPTM, 'PTM LUT');
}

if (DEBUG) {
  Map.centerObject(aoiOST, 10);
  Map.addLayer(aoiOST, {color: 'cyan'}, 'OST_DEEPWATER AOI');
  Map.addLayer(aoiPTM, {color: 'magenta'}, 'PTM_DEEPWATER AOI');
}

// =========================
// 6) EXPORTS (EE ASSETS)
// =========================
Export.table.toAsset({
  collection: lutOST,
  description: 'Export_Rinf_LUT_OST_L8L9_v1',
  assetId: EXPORT_ASSET_OST
});

Export.table.toAsset({
  collection: lutPTM,
  description: 'Export_Rinf_LUT_PTM_L8L9_v1',
  assetId: EXPORT_ASSET_PTM
});
