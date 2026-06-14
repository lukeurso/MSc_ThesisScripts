//import log
var OST2025_lake_1 =
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-45.894595095576555, 81.35247132991711],
          [-45.894595095576555, 81.34100404835044],
          [-45.8146008939164, 81.34100404835044],
          [-45.8146008939164, 81.35247132991711]]], null, false),
    OST2019_lake_2 =
    /* color: #98ff00 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-44.17401536614888, 81.32858804535199],
          [-44.17401536614888, 81.31055631033549],
          [-44.125950180602004, 81.31055631033549],
          [-44.125950180602004, 81.32858804535199]]], null, false),
    OST2023_lake_3 =
    /* color: #0b4a8b */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-44.83749151746474, 81.2062456218746],
          [-44.83749151746474, 81.19668792320479],
          [-44.74960089246474, 81.19668792320479],
          [-44.74960089246474, 81.2062456218746]]], null, false),
    PTM2025_lake_4 =
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-58.9282162365519, 80.15193486116232],
          [-58.9282162365519, 80.14130075506262],
          [-58.84822203489174, 80.14130075506262],
          [-58.84822203489174, 80.15193486116232]]], null, false),
    PTM2020_lake_5 =
    /* color: #98ff00 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-57.89616937862342, 80.49088942550357],
          [-57.89616937862342, 80.47550641320504],
          [-57.77737970577186, 80.47550641320504],
          [-57.77737970577186, 80.49088942550357]]], null, false),
    PTM2018_lake_6 =
    /* color: #0b4a8b */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-59.55601429011974, 80.43172158867334],
          [-59.55601429011974, 80.4165295440419],
          [-59.41387867000255, 80.4165295440419],
          [-59.41387867000255, 80.43172158867334]]], null, false);
// ============================================================
// volume_sensitivity_v1.js
//
// Sensitivity analysis for the Pope et al. (2016) two-band
// depth/volume model used in volume_v6.js.
//
// Independently perturbs:
//   g       - band attenuation coefficients (g_red + g_pan scaled together)
//   Ad_red  - per-lake deep-water ring reflectance, red band only
//   Ad_pan - per-lake deep-water ring reflectance, green (pan) band only
//   Ad_both - ad_red + ad_pan scaled together
//
// For each parameter, sweeps +/- MAX_PERTURB_PCT in N_STEPS intervals,
// holding all other parameters at their baseline values.
//
// Outputs per perturbation step (aggregated across all lakes in studyArea
// (polygon drawn around interest lake), averaged across all images in the
// filtered date range):
//   depth_pct_chg  — % change in mean lake depth from baseline
//   vol_pct_chg    — % change in total lake volume from baseline
//
// Results are:
//   1) Printed as two charts in the GEE Console
//      Chart A: Depth sensitivity  (% change in mean_depth_m)
//      Chart B: Volume sensitivity (% change in total_vol_m3)
//      Each chart shows three series: g, Ad_red, Ad_pan
//  2) Exported as a CSV to G Drive (can also export from chart pane)
// ============================================================


// ============================================================
// 1) SETTINGS
// ============================================================

// Set ACTIVE_GROUP to 'OST' or 'PTM' to run all three lakes for that site.
var ACTIVE_GROUP = 'PTM';

var LAKES_CONFIG = {
  OST: [
    { id: 'OST_lake_1', icName: 'OST2025', startDate: '2025-07-12', endDate: '2025-07-14', studyArea: OST2025_lake_1 },
    { id: 'OST_lake_2', icName: 'OST2019', startDate: '2019-07-10', endDate: '2019-07-12', studyArea: OST2019_lake_2 },
    { id: 'OST_lake_3', icName: 'OST2023', startDate: '2023-07-10', endDate: '2023-07-12', studyArea: OST2023_lake_3 }
  ],
  PTM: [
    { id: 'PTM_lake_4', icName: 'PTM2025', startDate: '2025-08-13', endDate: '2025-08-15', studyArea: PTM2025_lake_4 },
    { id: 'PTM_lake_5', icName: 'PTM2020', startDate: '2020-07-04', endDate: '2020-07-06', studyArea: PTM2020_lake_5 },
    { id: 'PTM_lake_6', icName: 'PTM2018', startDate: '2018-06-30', endDate: '2018-07-02', studyArea: PTM2018_lake_6 }
  ]
};

var LAKES = LAKES_CONFIG[ACTIVE_GROUP];

// Sensitivity sweep parameters
// Steps run from -MAX_PERTURB_PCT to +MAX_PERTURB_PCT.
// N_STEPS should be odd so that 0% falls exactly in the sequence.
var MAX_PERTURB_PCT = 20; // alt: 30 with 13 steps -> sweep is -30%, -25%, ..., 0%, ..., +30%
var N_STEPS         = 21; // alt: 13 steps -> 5% increment between steps

// Drive folder for CSV export
var EXPORT_FOLDER      = ACTIVE_GROUP + '_sensitivity_v1';
var EXPORT_FILE_PREFIX = 'sensitivity_g_Ad';

// ---- Asset paths (same as volume_v6.js) ----
var RINF_LUT_ASSETS = {
  OST: 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_OST_L8L9_v1',
  PTM: 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_PTM_L8L9_v1'
};
var COLLECTION_ASSET_BASE = 'projects/vernal-signal-270100/assets/ClassifiedMosaics/';

// ---- Band names (must match export_v2_5.js output) ----
var BAND_CLASS        = 'classification';
var BAND_RED          = 'B4_ring';
var BAND_PAN          = 'B8_ring';
var WATER_CLASS_VALUE = 1;

// ---- Depth model constants (Pope et al. 2016) ----
var g_red     = 0.80;
var g_pan     = 0.36;
var EPS       = 1e-6;
var MIN_DEPTH = 0;
var MAX_DEPTH = 20;

// ---- LUT time-join tolerance ----
var MAX_DIFF_DAYS = 30;

var EXPORT_CRS   = 'EPSG:3995';
var EXPORT_SCALE = 30;

// ---- Perturbation sequence ----
var perturbStep = (MAX_PERTURB_PCT * 2) / (N_STEPS - 1);
var perturbList = ee.List.sequence(-MAX_PERTURB_PCT, MAX_PERTURB_PCT, perturbStep);


// ============================================================
// 2) MATCH CLOSEST Rinf FEATURE BY TIME
//    (from volume_v6.js main workflow script)
// ============================================================

function attachClosestLut(image, lut) {
  var imageTime = ee.Number(image.get('system:time_start'));

  var withDiff = lut.map(function(f) {
    var lutTime = ee.Number(f.get('system:time_start'));
    var diffMs  = lutTime.subtract(imageTime).abs();
    return f.set({
      timeDiffMs:   diffMs,
      timeDiffDays: diffMs.divide(24 * 60 * 60 * 1000)
    });
  });

  var closest = ee.Feature(withDiff.sort('timeDiffMs').first());

  return image.set({
    Rinf_red:     closest.get('Rinf_red'),
    Rinf_pan:     closest.get('Rinf_pan'),
    nWater:       closest.get('nWater'),
    timeDiffDays: closest.get('timeDiffDays')
  });
}


// ============================================================
// 3) DEPTH FROM ONE BAND
//    (copied from volume_v6.js — works with ee.Number g)
// ============================================================

function depthFromBand(Rw, AdImg, RinfImg, g) {
  var valid = Rw.gt(RinfImg.add(EPS))
    .and(AdImg.gt(RinfImg.add(EPS)))
    .and(AdImg.gt(Rw.add(EPS)));

  var z = AdImg.subtract(RinfImg).log()
    .subtract(Rw.subtract(RinfImg).log())
    .divide(g);

  return z.updateMask(valid);
}


// ============================================================
// 4) COMPUTE AGGREGATED DEPTH & VOLUME STATS
//
//    Runs the full depth/volume workflow for one image with the
//    supplied parameter values, then reduces all lakes inside
//    studyArea to two scalars:
//      total_vol_m3  — sum of per-lake volumes
//      mean_depth_m  — unweighted mean of per-lake mean depths
//
//    Parameters:
//      image          — classified mosaic image (with Rinf attached)
//      g_red_val      — ee.Number  attenuation coefficient for red band
//      g_pan_val      — ee.Number  attenuation coefficient for pan band
//      Ad_red_factor  — ee.Number  multiplicative scaling of Ad_red image
//                       (1 = baseline; 1.10 = +10%; 0.90 = -10%)
//      Ad_pan_factor  — ee.Number  multiplicative scaling of Ad_pan image
//                       (1 = baseline; 1.10 = +10%; 0.90 = -10%)
//      studyArea      — ee.Geometry  clipping extent for this lake
// ============================================================

function computeStats(image, g_red_val, g_pan_val, Ad_red_factor, Ad_pan_factor, studyArea) {
  var classBand = image.select(BAND_CLASS);
  var b4        = image.select(BAND_RED).rename('B4');
  var b8raw     = image.select(BAND_PAN).rename('B8');
  var lakeMask  = classBand.eq(WATER_CLASS_VALUE).rename('lake');

  var b4Proj = b4.projection();
  var scale  = b4Proj.nominalScale();

  // Pan resampled to B4 resolution (30 m)
  var b8 = b8raw
    .resample('bilinear')
    .reproject({crs: b4Proj.crs(), scale: scale})
    .rename('B8_30m');

  // ---- Vectorize lakes within studyArea only ----
  var lakeVectors = lakeMask.selfMask().reduceToVectors({
    geometry:       studyArea,
    crs:            b4Proj,
    scale:          scale,
    geometryType:   'polygon',
    eightConnected: true,
    maxPixels:      1e9,
    bestEffort:     true
  });
  lakeVectors = lakeVectors.filter(ee.Filter.gte('count', 2));

  // ---- Per-lake Ad from 1-pixel ring buffer ----
  var lakeVectorsWithAd = lakeVectors.map(function(feat) {
    var lakeGeom = feat.geometry();
    var buffered = lakeGeom.buffer({distance: scale, maxError: 10});
    var ring     = buffered.difference({right: lakeGeom, maxError: 10});

    var adDict = image.select([BAND_RED, BAND_PAN]).reduceRegion({
      reducer:    ee.Reducer.mean(),
      geometry:   ring,
      scale:      scale,
      maxPixels:  1e8,
      bestEffort: true
    });

    return feat
      .set('Ad_red', ee.Number(adDict.get(BAND_RED)))
      .set('Ad_pan', ee.Number(adDict.get(BAND_PAN)));
  });

  // ---- Paint per-lake Ad back to image space, then apply scaling ----
  var Ad_red_img = ee.Image(0).float()
    .paint(lakeVectorsWithAd, 'Ad_red')
    .updateMask(lakeMask)
    .reproject({crs: b4Proj.crs(), scale: scale})
    .rename('Ad_red')
    .multiply(Ad_red_factor);    // applies red-channel perturbation (1 = no change)

  var Ad_pan_img = ee.Image(0).float()
    .paint(lakeVectorsWithAd, 'Ad_pan')
    .updateMask(lakeMask)
    .reproject({crs: b4Proj.crs(), scale: scale})
    .rename('Ad_pan')
    .multiply(Ad_pan_factor);  // applies green-channel perturbation (1 = no change)

  // ---- Rinf from matched LUT ----
  var Rinf_red   = ee.Number(image.get('Rinf_red'));
  var Rinf_pan   = ee.Number(image.get('Rinf_pan'));
  var RinfRedImg = ee.Image.constant(Rinf_red).reproject(b4Proj);
  var RinfPanImg = ee.Image.constant(Rinf_pan).reproject(b4Proj);

  var Rw_red = b4.updateMask(lakeMask);
  var Rw_pan = b8.updateMask(lakeMask);

  // ---- Depth raster (average of red and pan, clamped) ----
  var zRed = depthFromBand(Rw_red, Ad_red_img,   RinfRedImg, g_red_val);
  var zPan = depthFromBand(Rw_pan, Ad_pan_img, RinfPanImg, g_pan_val);

  var depth = zRed.add(zPan).divide(2)
    .clamp(MIN_DEPTH, MAX_DEPTH)
    .updateMask(lakeMask)
    .rename('lake_depth_m');

  var pixelArea = ee.Image.pixelArea();
  var volImg    = depth.multiply(pixelArea).rename('pixel_vol_m3');

  // ---- Aggregate across all lake polygons in studyArea ----
  //   total_vol_m3  = sum of per-lake volume sums
  //   mean_depth_m  = mean of per-lake mean depths
  var lakeStats = lakeVectorsWithAd.map(function(feat) {
    var geom = feat.geometry();

    var volDict = volImg.reduceRegion({
      reducer:    ee.Reducer.sum(),
      geometry:   geom,
      scale:      scale,
      maxPixels:  1e8,
      bestEffort: true
    });

    var depthDict = depth.reduceRegion({
      reducer:    ee.Reducer.mean(),
      geometry:   geom,
      scale:      scale,
      maxPixels:  1e8,
      bestEffort: true
    });

    return feat
      .set('lake_vol_m3',   ee.Number(volDict.get('pixel_vol_m3')))
      .set('lake_depth_m',  ee.Number(depthDict.get('lake_depth_m')));
  });

  // Sum volumes and average depths across all lakes
  var total_vol   = ee.Number(lakeStats.aggregate_sum('lake_vol_m3'));
  var mean_depth  = ee.Number(lakeStats.aggregate_mean('lake_depth_m'));

  return ee.Dictionary({
    total_vol_m3: total_vol,
    mean_depth_m: mean_depth
  });
}


// ============================================================
// 5) BUILD SENSITIVITY TABLE FOR ONE IMAGE
//
//    Returns an ee.FeatureCollection with one Feature per
//    (param, perturbation step), carrying:
//      lake_id, export_id, win_start, win_end,
//      param         — 'g' or 'Ad'
//      perturb_pct   — perturbation applied (e.g. -30, -25, … +30)
//      depth_pct_chg — % change in mean_depth_m vs baseline
//      vol_pct_chg   — % change in total_vol_m3 vs baseline
// ============================================================

function buildSensitivityForImage(image, lut, studyArea, lakeId) {
  image = attachClosestLut(image, lut);

  var exportId = image.get('export_id');
  var winStart = image.get('window_start');
  var winEnd   = image.get('window_end');

  // ---- Baseline (no perturbation) ----
  var baseline   = computeStats(image, ee.Number(g_red), ee.Number(g_pan), ee.Number(1), ee.Number(1), studyArea);
  var base_vol   = ee.Number(baseline.get('total_vol_m3'));
  var base_depth = ee.Number(baseline.get('mean_depth_m'));

  // ---- g sensitivity (both Ad factors = 1) ----
  var gSens = perturbList.map(function(pct) {
    pct = ee.Number(pct);
    var factor = pct.divide(100).add(1);
    var stats  = computeStats(image,
      ee.Number(g_red).multiply(factor),
      ee.Number(g_pan).multiply(factor),
      ee.Number(1),
      ee.Number(1),
      studyArea);

    var vol_pct   = ee.Number(stats.get('total_vol_m3'))
                      .subtract(base_vol).divide(base_vol).multiply(100);
    var depth_pct = ee.Number(stats.get('mean_depth_m'))
                      .subtract(base_depth).divide(base_depth).multiply(100);

    return ee.Feature(null, {
      lake_id:       lakeId,
      export_id:     exportId,
      win_start:     winStart,
      win_end:       winEnd,
      param:         'g',
      perturb_pct:   pct,
      depth_pct_chg: depth_pct,
      vol_pct_chg:   vol_pct
    });
  });

  // ---- Ad_red sensitivity (g at baseline, Ad_pan_factor = 1) ----
  var AdRedSens = perturbList.map(function(pct) {
    pct = ee.Number(pct);
    var factor = pct.divide(100).add(1);
    var stats  = computeStats(image,
      ee.Number(g_red),
      ee.Number(g_pan),
      factor,          // Ad_red perturbed
      ee.Number(1),    // Ad_pan held at baseline
      studyArea);

    var vol_pct   = ee.Number(stats.get('total_vol_m3'))
                      .subtract(base_vol).divide(base_vol).multiply(100);
    var depth_pct = ee.Number(stats.get('mean_depth_m'))
                      .subtract(base_depth).divide(base_depth).multiply(100);

    return ee.Feature(null, {
      lake_id:       lakeId,
      export_id:     exportId,
      win_start:     winStart,
      win_end:       winEnd,
      param:         'Ad_red',
      perturb_pct:   pct,
      depth_pct_chg: depth_pct,
      vol_pct_chg:   vol_pct
    });
  });

  // ---- Ad_pan sensitivity (g at baseline, Ad_red_factor = 1) ----
  var AdPanSens = perturbList.map(function(pct) {
    pct = ee.Number(pct);
    var factor = pct.divide(100).add(1);
    var stats  = computeStats(image,
      ee.Number(g_red),
      ee.Number(g_pan),
      ee.Number(1),    // Ad_red held at baseline
      factor,          // Ad_pan perturbed
      studyArea);

    var vol_pct   = ee.Number(stats.get('total_vol_m3'))
                      .subtract(base_vol).divide(base_vol).multiply(100);
    var depth_pct = ee.Number(stats.get('mean_depth_m'))
                      .subtract(base_depth).divide(base_depth).multiply(100);

    return ee.Feature(null, {
      lake_id:       lakeId,
      export_id:     exportId,
      win_start:     winStart,
      win_end:       winEnd,
      param:         'Ad_pan',
      perturb_pct:   pct,
      depth_pct_chg: depth_pct,
      vol_pct_chg:   vol_pct
    });
  });

  // ---- Ad_red + Ad_pan scaled together (g at baseline) ----
  var AdBothSens = perturbList.map(function(pct) {
    pct = ee.Number(pct);
    var factor = pct.divide(100).add(1);
    var stats  = computeStats(image,
      ee.Number(g_red),
      ee.Number(g_pan),
      factor,    // Ad_red perturbed
      factor,    // Ad_pan perturbed together
      studyArea);

    var vol_pct   = ee.Number(stats.get('total_vol_m3'))
                      .subtract(base_vol).divide(base_vol).multiply(100);
    var depth_pct = ee.Number(stats.get('mean_depth_m'))
                      .subtract(base_depth).divide(base_depth).multiply(100);

    return ee.Feature(null, {
      lake_id:       lakeId,
      export_id:     exportId,
      win_start:     winStart,
      win_end:       winEnd,
      param:         'Ad_both',
      perturb_pct:   pct,
      depth_pct_chg: depth_pct,
      vol_pct_chg:   vol_pct
    });
  });

  return ee.FeatureCollection(gSens.cat(AdRedSens).cat(AdPanSens).cat(AdBothSens));
}


// ============================================================
// 6) MAIN — parallel evaluate() per lake, finalize when all done
// ============================================================

// Shared LUT for the active group (all lakes in a group use the same LUT asset).
var groupLut = ee.FeatureCollection(RINF_LUT_ASSETS[ACTIVE_GROUP])
  .filter(ee.Filter.eq('ok', 1))
  .sort('system:time_start');

// Finalize is called once every lake's evaluate() has returned.
function finalizeResults(allLakeResults) {
  var mergedFC = ee.FeatureCollection(allLakeResults).flatten();

  // ----------------------------------------------------------------
  // Average % changes across all images and lakes
  // (group by param + perturb_pct for the charts).
  // ----------------------------------------------------------------
  var avgFC = perturbList.map(function(pct) {
    pct = ee.Number(pct);

    var gSubset      = mergedFC
      .filter(ee.Filter.eq('param', 'g'))
      .filter(ee.Filter.eq('perturb_pct', pct));
    var AdRedSubset  = mergedFC
      .filter(ee.Filter.eq('param', 'Ad_red'))
      .filter(ee.Filter.eq('perturb_pct', pct));
    var AdPanSubset  = mergedFC
      .filter(ee.Filter.eq('param', 'Ad_pan'))
      .filter(ee.Filter.eq('perturb_pct', pct));
    var AdBothSubset = mergedFC
      .filter(ee.Filter.eq('param', 'Ad_both'))
      .filter(ee.Filter.eq('perturb_pct', pct));

    return ee.Feature(null, {
      perturb_pct:            pct,
      g_depth_pct_chg:        ee.Number(gSubset.aggregate_mean('depth_pct_chg')),
      g_vol_pct_chg:          ee.Number(gSubset.aggregate_mean('vol_pct_chg')),
      Ad_red_depth_pct_chg:   ee.Number(AdRedSubset.aggregate_mean('depth_pct_chg')),
      Ad_red_vol_pct_chg:     ee.Number(AdRedSubset.aggregate_mean('vol_pct_chg')),
      Ad_pan_depth_pct_chg:   ee.Number(AdPanSubset.aggregate_mean('depth_pct_chg')),
      Ad_pan_vol_pct_chg:     ee.Number(AdPanSubset.aggregate_mean('vol_pct_chg')),
      Ad_both_depth_pct_chg:  ee.Number(AdBothSubset.aggregate_mean('depth_pct_chg')),
      Ad_both_vol_pct_chg:    ee.Number(AdBothSubset.aggregate_mean('vol_pct_chg'))
    });
  });

  var avgFeatureCollection = ee.FeatureCollection(avgFC);

  // ----------------------------------------------------------------
  // Chart A: Depth sensitivity
  // ----------------------------------------------------------------
  var depthChart = ui.Chart.feature.byFeature({
    features:    avgFeatureCollection,
    xProperty:   'perturb_pct',
    yProperties: ['g_depth_pct_chg', 'Ad_red_depth_pct_chg', 'Ad_pan_depth_pct_chg', 'Ad_both_depth_pct_chg']
  })
  .setChartType('LineChart')
  .setOptions({
    title:  'Depth Sensitivity — % change in mean lake depth (' + ACTIVE_GROUP + ', ' + LAKES.length + ' lakes)',
    hAxis:  {
      title: 'Parameter perturbation (%)',
      viewWindow: {min: -MAX_PERTURB_PCT, max: MAX_PERTURB_PCT}
    },
    vAxis:  {title: '% change in mean depth from baseline'},
    series: {
      0: {color: '1f77b4', labelInLegend: 'g sensitivity',              lineWidth: 2, pointSize: 4},
      1: {color: 'd62728', labelInLegend: 'Ad_red sensitivity',          lineWidth: 2, pointSize: 4},
      2: {color: '2ca02c', labelInLegend: 'Ad_pan sensitivity',          lineWidth: 2, pointSize: 4},
      3: {color: 'ff7f0e', labelInLegend: 'Ad_red & Ad_pan sensitivity', lineWidth: 2, pointSize: 4}
    },
    legend: {position: 'bottom'},
    interpolateNulls: true
  });

  print('Chart A — Depth Sensitivity:', depthChart);

  // ----------------------------------------------------------------
  // Chart B: Volume sensitivity
  // ----------------------------------------------------------------
  var volumeChart = ui.Chart.feature.byFeature({
    features:    avgFeatureCollection,
    xProperty:   'perturb_pct',
    yProperties: ['g_vol_pct_chg', 'Ad_red_vol_pct_chg', 'Ad_pan_vol_pct_chg', 'Ad_both_vol_pct_chg']
  })
  .setChartType('LineChart')
  .setOptions({
    title:  'Volume Sensitivity — % change in total lake volume (' + ACTIVE_GROUP + ', ' + LAKES.length + ' lakes)',
    hAxis:  {
      title: 'Parameter perturbation (%)',
      viewWindow: {min: -MAX_PERTURB_PCT, max: MAX_PERTURB_PCT}
    },
    vAxis:  {title: '% change in total volume from baseline'},
    series: {
      0: {color: '1f77b4', labelInLegend: 'g sensitivity',              lineWidth: 2, pointSize: 4},
      1: {color: 'd62728', labelInLegend: 'Ad_red sensitivity',          lineWidth: 2, pointSize: 4},
      2: {color: '2ca02c', labelInLegend: 'Ad_pan sensitivity',          lineWidth: 2, pointSize: 4},
      3: {color: 'ff7f0e', labelInLegend: 'Ad_red & Ad_pan sensitivity', lineWidth: 2, pointSize: 4}
    },
    legend: {position: 'bottom'},
    interpolateNulls: true
  });

  print('Chart B — Volume Sensitivity:', volumeChart);

  // ----------------------------------------------------------------
  // CSV export to Google Drive
  // Fields: lake_id, export_id, win_start, win_end, param,
  //         perturb_pct, depth_pct_chg, vol_pct_chg
  // (one row per lake × image × param × perturbation step)
  // ----------------------------------------------------------------
  Export.table.toDrive({
    collection:     mergedFC,
    description:    EXPORT_FILE_PREFIX,
    fileNamePrefix: EXPORT_FILE_PREFIX,
    folder:         EXPORT_FOLDER,
    fileFormat:     'CSV',
    selectors:      ['lake_id', 'export_id', 'win_start', 'win_end',
                     'param', 'perturb_pct',
                     'depth_pct_chg', 'vol_pct_chg']
  });

  print('CSV export queued to Drive folder: ' + EXPORT_FOLDER);
}

// Fire one evaluate() per lake in parallel; finalize once all complete.
var pendingLakes = LAKES.length;
var allLakeResults = [];

LAKES.forEach(function(lakeConfig) {
  var collection = ee.ImageCollection(COLLECTION_ASSET_BASE + lakeConfig.icName)
    .filterDate(lakeConfig.startDate, lakeConfig.endDate)
    .filterBounds(lakeConfig.studyArea);

  print('Collection size for ' + lakeConfig.id + ':', collection.size());
  print('Collection export_ids for ' + lakeConfig.id + ':', collection.aggregate_array('export_id'));

  collection.aggregate_array('export_id').evaluate(function(exportIds) {
    if (!exportIds || exportIds.length === 0) {
      print('WARNING: No images found for lake ' + lakeConfig.id + '. Skipping.');
      pendingLakes--;
      if (pendingLakes === 0) finalizeResults(allLakeResults);
      return;
    }

    print('Processing ' + exportIds.length + ' image(s) for ' + lakeConfig.id);

    var lakeResults = [];
    exportIds.forEach(function(exportId) {
      if (!exportId) {
        print('WARNING: skipping image with null export_id in ' + lakeConfig.id);
        return;
      }

      var img = ee.Image(
        collection.filter(ee.Filter.eq('export_id', exportId)).first()
      );

      var sensitivityFC = buildSensitivityForImage(
        img, groupLut, lakeConfig.studyArea, lakeConfig.id
      );
      lakeResults.push(sensitivityFC);

      print('Queued sensitivity computation for: ' + exportId + ' (' + lakeConfig.id + ')');
    });

    if (lakeResults.length > 0) {
      allLakeResults.push(ee.FeatureCollection(lakeResults).flatten());
    }

    pendingLakes--;
    if (pendingLakes === 0) finalizeResults(allLakeResults);
  });
});
