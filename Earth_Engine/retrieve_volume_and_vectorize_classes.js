// ============================================================
// volume_v6.js
// Cycles through the OST or PTM classified ImageCollection
// produced by export_v2_5.js and applies the volume_test_v4
// depth workflow to every image.
//
// For each mosaic window the script queues:
//   1) A lake vector shapefile (SHP) with per-lake fields:
//        area_m2, volume_m3, mean_depth_m, export_id.
//   2) A slush vector shapefile (SHP) with per-polygon fields:
//        area_m2, export_id.
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
var FILTER_START = null; // e.g. '2019-07-01'
var FILTER_END   = null; // e.g. '2019-09-01'

// Google Drive folders
var EXPORT_LAKE_FOLDER      = 'GEE_volume_v6_lake_vectors';
var EXPORT_SLUSH_FOLDER     = 'GEE_volume_v6_slush_vectors';
var EXPORT_FOOTPRINT_FOLDER = 'GEE_volume_v6_footprint_vectors';

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

  var withDiff = lut.map(function(f) {
    var lutTime = ee.Number(f.get('system:time_start'));
    var diffMs  = lutTime.subtract(imageTime).abs();
    return f.set({
      timeDiffMs:   diffMs,
      timeDiffDays: diffMs.divide(24 * 60 * 60 * 1000)
    });
  }).filter(ee.Filter.lte('timeDiffMs', maxDiffMs));

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
//    Returns { lakeVectors, slushVectors }
//      lakeVectors  – lake polygons with area_m2, volume_m3,
//                     mean_depth_m, export_id
//      slushVectors – slush polygons with area_m2, export_id
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
    eightConnected: false,
    maxPixels:      1e9,
    bestEffort:     true
  });

  // ----------------------------------------------------------
  // Vectorize slush
  // ----------------------------------------------------------
  var slushVectors = slushMask.selfMask().reduceToVectors({
    geometry:       region,
    crs:            b4Proj,
    scale:          scale,
    geometryType:   'polygon',
    eightConnected: false,
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
      'export_id':    image.get('export_id'),
      'window_start': image.get('window_start'),
      'window_end':   image.get('window_end')
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
      .set('export_id',  image.get('export_id'));
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
      .set('area_m2',   ee.Number(areaDict.get('area_m2')))
      .set('export_id', image.get('export_id'));
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
    // Export 1: lake vectors with area, volume, mean_depth
    // ----------------------------------------------------------
    Export.table.toDrive({
      collection:     result.lakeVectors,
      description:    exportId + '_lake_vectors',
      fileNamePrefix: exportId + '_lake_vectors',
      folder:         EXPORT_LAKE_FOLDER,
      fileFormat:     'SHP'
    });

    // ----------------------------------------------------------
    // Export 2: slush vectors with area
    // ----------------------------------------------------------
    Export.table.toDrive({
      collection:     result.slushVectors,
      description:    exportId + '_slush_vectors',
      fileNamePrefix: exportId + '_slush_vectors',
      folder:         EXPORT_SLUSH_FOLDER,
      fileFormat:     'SHP'
    });

    // ----------------------------------------------------------
    // Export 3: mosaic footprint polygon (valid-data extent)
    // ----------------------------------------------------------
    Export.table.toDrive({
      collection:     result.footprintVectors,
      description:    exportId + '_footprint_vectors',
      fileNamePrefix: exportId + '_footprint_vectors',
      folder:         EXPORT_FOOTPRINT_FOLDER,
      fileFormat:     'SHP'
    });

    if (DEBUG) {
      print('Queued lake + slush + footprint vector exports for: ' + exportId);
    }
  });
});
