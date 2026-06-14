// ============================================================
// volume_v6.js
//  - mosaic footprint vectorized and exported
// Cycles through the OST or PTM classified ImageCollection
// produced by export_v2_5.js and applies the volume_test_v4
// depth workflow to every image.
//
// For each mosaic window the script queues three SHP exports
// to a single shared Drive folder, all carrying the standard
// linking fields: export_id, win_start, win_end, layer_type.
// (win_start / win_end are abbreviated from window_start / window_end
// to stay within the shapefile 10-character field name limit.)
//
//   1) Lake vectors      – layer_type='lake'
//        fields: export_id, win_start, win_end, layer_type,
//                area_m2, volume_m3, mean_dpt_m, Ad_red, Ad_pan
//   2) Slush vectors     – layer_type='slush'
//        fields: export_id, win_start, win_end, layer_type,
//                area_m2
//   3) Footprint vectors – layer_type='footprint'
//        fields: export_id, win_start, win_end, layer_type
//
// All three exports share export_id + win_start/win_end as join keys,
// and layer_type distinguishes them if merged into a single table.
//
// Depth model: Pope et al. (2016) two-band (red + pan) approach.
// Per-lake Ad is derived from a 1-pixel ring around each lake
// polygon, identical to the volume_test_v4.js workflow.
// ============================================================


// ============================================================
// 1) USER SETTINGS
// ============================================================

var AOI_DESIGNATION = 'OST'; // 'OST' or 'PTM'

// Rinf LUT assets (one per site, produced by deepwater_lookup.js)
var RINF_LUT_ASSETS = {
  OST: 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_OST_L8L9_v1',
  PTM: 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_PTM_L8L9_v1'
};
var RINF_LUT_ASSET = RINF_LUT_ASSETS[AOI_DESIGNATION];

// ImageCollection of classified mosaics produced by export_v2_5.js.
// Each image has bands: classification (1=water,2=slush,3=other),
// B4_ring, B8_ring  and properties: export_id, window_start,
// window_end, system:time_start.
var COLLECTION_ASSET =
  'projects/vernal-signal-270100/assets/ClassifiedMosaics/' + AOI_DESIGNATION;

// Optional date filter on the collection (null = all images).
var FILTER_START = null; //'2025-05-01'; 
var FILTER_END   = null; //'2025-06-01'; 

// Google Drive folder — all three layer types land here together so
// every window's lake/slush/footprint files are co-located and
// linkable via export_id / win_start / win_end / layer_type.
var EXPORT_VECTORS_FOLDER = 'OST_TEST_GEE_volume_v6_vectors';

var EXPORT_CRS   = 'EPSG:3995';
var EXPORT_SCALE = 30;

// ---- Band names in export_v4.js output images ----
var BAND_CLASS     = 'classification';
var BAND_RED       = 'B4_ring';
var BAND_PAN       = 'B8_ring';
var BAND_FOOTPRINT = 'mosaic_footprint';
var WATER_CLASS_VALUE = 1;
var SLUSH_CLASS_VALUE = 2;

// ---- Depth model constants (Pope et al. 2016) ----
var g_red     = 0.80;
var g_pan     = 0.36;
var EPS       = 1e-6;
var MIN_DEPTH = 0;
var MAX_DEPTH = 20;

// ---- LUT time-join tolerance ----
var MAX_DIFF_DAYS = 30;

var DEBUG = true;


// ============================================================
// 2) LOAD LUT + COLLECTION
// ============================================================

var lut = ee.FeatureCollection(RINF_LUT_ASSET)
  .filter(ee.Filter.eq('ok', 1))
  .sort('system:time_start');

var collection = ee.ImageCollection(COLLECTION_ASSET);
if (FILTER_START !== null && FILTER_END !== null) {
  collection = collection.filterDate(FILTER_START, FILTER_END);
}

print('Collection size:', collection.size());
if (DEBUG) {
  print('Collection export_ids:', collection.aggregate_array('export_id'));
}


// ============================================================
// 3) MATCH CLOSEST Rinf FEATURE BY TIME
// ============================================================

function attachClosestLut(image) {
  var imageTime = ee.Number(image.get('system:time_start'));
  var maxDiffMs = ee.Number(MAX_DIFF_DAYS).multiply(24 * 60 * 60 * 1000);

  // Compute time difference for every LUT entry and pick the globally
  // closest match.  Filtering to maxDiffMs BEFORE calling .first() caused
  // an empty collection when no entry fell within the window, making
  // .first() return null and crashing all downstream .get() calls with:
  //   "Element.get: Parameter 'object' is required and may not be null."
  // timeDiffDays is still written onto the image so exports can be QA-
  // filtered for poor matches after the fact.
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
// 4) DEPTH FROM ONE BAND
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
// 5) PROCESS ONE IMAGE
//    Returns { lakeVectors, slushVectors, footprintVectors }
//    All three FeatureCollections carry the standard linking fields:
//      export_id, win_start, win_end, layer_type
//    Additional fields per layer:
//      lakeVectors      – area_m2, volume_m3, mean_dpt_m, Ad_red, Ad_pan
//      slushVectors     – area_m2
//      footprintVectors – (linking fields only)
// ============================================================

function processImage(image) {
  image = attachClosestLut(image);

  var classBand = image.select(BAND_CLASS);
  var b4        = image.select(BAND_RED).rename('B4');
  var b8raw     = image.select(BAND_PAN).rename('B8');
  var lakeMask  = classBand.eq(WATER_CLASS_VALUE).rename('lake');
  var slushMask = classBand.eq(SLUSH_CLASS_VALUE).rename('slush');

  var b4Proj = b4.projection();
  var scale  = b4Proj.nominalScale();
  var region = image.geometry();

  // Pan resampled to B4 resolution (30 m)
  var b8 = b8raw
    .resample('bilinear')
    .reproject({crs: b4Proj.crs(), scale: scale})
    .rename('B8_30m');

  // ----------------------------------------------------------
  // Vectorize lakes
  // ----------------------------------------------------------
  var lakeVectors = lakeMask.selfMask().reduceToVectors({
    geometry:       region,
    crs:            b4Proj,
    scale:          scale,
    geometryType:   'polygon',
    eightConnected: true,
    maxPixels:      1e9,
    bestEffort:     true
  });
  // Remove single-pixel lakes (area < 1800 m² at 30 m resolution)
  lakeVectors = lakeVectors.filter(ee.Filter.gte('count', 2));

  // ----------------------------------------------------------
  // Vectorize slush
  // ----------------------------------------------------------
  var slushVectors = slushMask.selfMask().reduceToVectors({
    geometry:       region,
    crs:            b4Proj,
    scale:          scale,
    geometryType:   'polygon',
    eightConnected: true,
    maxPixels:      1e9,
    bestEffort:     true
  });

  // ----------------------------------------------------------
  // Vectorize mosaic footprint (valid-data extent)
  // Value=1 where the mosaic had at least one unmasked observation.
  // Exported as a polygon layer so ArcGIS spatial queries can
  // distinguish confirmed lake drainage (footprint covers area,
  // no lake_N+1 present) from unobservable gaps (footprint absent).
  // ----------------------------------------------------------
  var footprintVectors = image.select(BAND_FOOTPRINT).eq(1).selfMask()
    .reduceToVectors({
      geometry:       region,
      crs:            b4Proj,
      scale:          scale,
      geometryType:   'polygon',
      eightConnected: false,
      maxPixels:      1e9,
      bestEffort:     true
    });

  var footprintVectorsTagged = footprintVectors.map(function(feat) {
    return feat.set({
      'export_id':  image.get('export_id'),
      'win_start':  image.get('window_start'),
      'win_end':    image.get('window_end'),
      'layer_type': 'footprint'
    });
  });

  // ----------------------------------------------------------
  // Per-lake Ad via 1-pixel ring buffer
  // ----------------------------------------------------------
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

  // ----------------------------------------------------------
  // Paint per-lake Ad values back to image space
  // ----------------------------------------------------------
  var Ad_red_img = ee.Image(0).float()
    .paint(lakeVectorsWithAd, 'Ad_red')
    .updateMask(lakeMask)
    .reproject({crs: b4Proj.crs(), scale: scale})
    .rename('Ad_red');

  var Ad_pan_img = ee.Image(0).float()
    .paint(lakeVectorsWithAd, 'Ad_pan')
    .updateMask(lakeMask)
    .reproject({crs: b4Proj.crs(), scale: scale})
    .rename('Ad_pan');

  // ----------------------------------------------------------
  // Rinf constants from matched LUT row (per-scene scalar)
  // ----------------------------------------------------------
  var Rinf_red   = ee.Number(image.get('Rinf_red'));
  var Rinf_pan   = ee.Number(image.get('Rinf_pan'));
  var RinfRedImg = ee.Image.constant(Rinf_red).reproject(b4Proj);
  var RinfPanImg = ee.Image.constant(Rinf_pan).reproject(b4Proj);

  var Rw_red = b4.updateMask(lakeMask);
  var Rw_pan = b8.updateMask(lakeMask);

  // ----------------------------------------------------------
  // Depth raster (average of red and pan estimates)
  // ----------------------------------------------------------
  var zRed = depthFromBand(Rw_red, Ad_red_img, RinfRedImg, g_red);
  var zPan = depthFromBand(Rw_pan, Ad_pan_img, RinfPanImg, g_pan);

  var depth = zRed.add(zPan).divide(2)
    .clamp(MIN_DEPTH, MAX_DEPTH)
    .updateMask(lakeMask)
    .rename('lake_depth_m');

  // ----------------------------------------------------------
  // Per-lake area, volume, and mean depth
  // ----------------------------------------------------------
  var pixelArea = ee.Image.pixelArea();
  var volImg    = depth.multiply(pixelArea).rename('pixel_vol_m3');

  var lakeVectorsWithStats = lakeVectorsWithAd.map(function(feat) {
    var geom = feat.geometry();

    var areaDict = pixelArea.rename('area_m2').reduceRegion({
      reducer:    ee.Reducer.sum(),
      geometry:   geom,
      scale:      scale,
      maxPixels:  1e8,
      bestEffort: true
    });

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
      .set('area_m2',    ee.Number(areaDict.get('area_m2')))
      .set('volume_m3',  ee.Number(volDict.get('pixel_vol_m3')))
      .set('mean_dpt_m', ee.Number(depthDict.get('lake_depth_m')))
      .set('export_id',  image.get('export_id'))
      .set('win_start',  image.get('window_start'))
      .set('win_end',    image.get('window_end'))
      .set('layer_type', 'lake');
  });

  // ----------------------------------------------------------
  // Per-slush-polygon area
  // ----------------------------------------------------------
  var slushVectorsWithArea = slushVectors.map(function(feat) {
    var geom = feat.geometry();

    var areaDict = pixelArea.rename('area_m2').reduceRegion({
      reducer:    ee.Reducer.sum(),
      geometry:   geom,
      scale:      scale,
      maxPixels:  1e8,
      bestEffort: true
    });

    return feat
      .set('area_m2',    ee.Number(areaDict.get('area_m2')))
      .set('export_id',  image.get('export_id'))
      .set('win_start',  image.get('window_start'))
      .set('win_end',    image.get('window_end'))
      .set('layer_type', 'slush');
  });

  return {
    lakeVectors:      lakeVectorsWithStats,
    slushVectors:     slushVectorsWithArea,
    footprintVectors: footprintVectorsTagged
  };
}


// ============================================================
// 6) QUEUE EXPORTS — client-side iteration over export_ids
// ============================================================

// All three layer types export to EXPORT_VECTORS_FOLDER so every
// window's files are co-located in one Drive folder.
//
// Resolve export_id list from the collection metadata, then
// process and queue each image individually (mirrors the
// export_v2_5.js client-side loop pattern).
collection.aggregate_array('export_id').evaluate(function(exportIds) {
  if (!exportIds || exportIds.length === 0) {
    print('No images found in collection. Check COLLECTION_ASSET and date filters.');
    return;
  }

  print('Queueing exports for ' + exportIds.length + ' image(s).');

  exportIds.forEach(function(exportId) {
    // Guard against null export_id entries
    if (!exportId) {
      print('WARNING: skipping image with null export_id.');
      return;
    }

    var img    = ee.Image(collection.filter(ee.Filter.eq('export_id', exportId)).first());
    var result = processImage(img);

    // ----------------------------------------------------------
    // Export 1: lake vectors
    //   fields: export_id, win_start, win_end, layer_type,
    //           area_m2, volume_m3, mean_dpt_m, Ad_red, Ad_pan
    // ----------------------------------------------------------
    Export.table.toDrive({
      collection:     result.lakeVectors,
      description:    exportId + '_lake',
      fileNamePrefix: exportId + '_lake',
      folder:         EXPORT_VECTORS_FOLDER,
      fileFormat:     'SHP'
    });

    // ----------------------------------------------------------
    // Export 2: slush vectors
    //   fields: export_id, win_start, win_end, layer_type, area_m2
    // ----------------------------------------------------------
    Export.table.toDrive({
      collection:     result.slushVectors,
      description:    exportId + '_slush',
      fileNamePrefix: exportId + '_slush',
      folder:         EXPORT_VECTORS_FOLDER,
      fileFormat:     'SHP'
    });

    // ----------------------------------------------------------
    // Export 3: mosaic footprint polygon (valid-data extent)
    //   fields: export_id, win_start, win_end, layer_type
    // ----------------------------------------------------------
    Export.table.toDrive({
      collection:     result.footprintVectors,
      description:    exportId + '_footprint',
      fileNamePrefix: exportId + '_footprint',
      folder:         EXPORT_VECTORS_FOLDER,
      fileFormat:     'SHP'
    });

    if (DEBUG) {
      print('Queued lake + slush + footprint vector exports for: ' + exportId);
    }
  });
});
