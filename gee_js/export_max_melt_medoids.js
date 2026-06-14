
/**** =====================================================================
 *  export_max_melt_medoids.js
 *
 *  Interactive tool for browsing and exporting annual max-melt 2-day
 *  medoids.  For each AOI (OST / PTM) and each year's start date, five
 *  sequential 2-day medoid windows are generated.
 *
 *  Workflow:
 *    1. Select AOI in the side panel.
 *    2. Select a start date — five windows appear as map layers.
 *    3. Check the windows you want to export.
 *    4. Click "Export selected" to queue Drive tasks.
 *
 *  Exported files are saved to Google Drive folder: annual_max_melt_medoids
 *  File naming: {AOI}_{start-date}_{end-date}_max_melt
 *  ===================================================================== ****/

// ============================================================
// CONFIGURATION
// ============================================================

var AOI_ASSETS = {
  OST: 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/OST_AOI_1500m',
  PTM: 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/PTM_AOI_1500m'
};


// Annual max-melt start dates, one per year per AOI
var START_DATES = {
  OST: [
    '2014-06-20', '2015-06-30', '2016-07-20', '2017-07-20',
    '2018-07-30', '2019-06-10', '2020-06-30', '2021-07-30',
    '2022-07-10', '2023-06-20', '2024-06-20', '2025-08-09'
  ],
  PTM: [
    '2014-06-30', '2015-06-30', '2016-07-10', '2017-07-20',
    '2018-07-30', '2019-06-30', '2020-07-10', '2021-07-30',
    '2022-07-10', '2023-06-30', '2024-06-30', '2025-07-20'
  ]
};

var N_WINDOWS             = 4;
var MOSAIC_LEN_DAYS       = 5;
var MIN_SUN_ELEV          = 20;
var MAX_CLOUD_COVER       = 10;   // null = no pre-filter
var MIN_SCENES_PER_WINDOW = 1;
var MAX_SCENES_PER_WINDOW = 20;     // null = no limit
var MEDOID_DISTANCE_BANDS = ['NDWI_ICE'];

var APPLY_CLOUD_MASK = false;
var APPLY_ROCK_MASK  = false;

var EXPORT_FOLDER = 'annual_max_melt_medoids';
var EXPORT_CRS    = 'EPSG:3413';
var EXPORT_SCALE  = 60;

var RGB_VIS_PARAMS = {
  bands: ['B4', 'B3', 'B2'],
  min:   0,
  max:   1,
  gamma: 1.0
};

// ============================================================
// CLOUD MASK SETTINGS
// ============================================================

var cloudCfg = {
  USE_B9_CIRRUS:             true,
  USE_QA_BITMASK:            true,
  USE_QA_DILATED:            false,
  USE_QA_CLOUD_BIT:          true,
  USE_QA_SHADOW_BIT:         true,
  USE_QA_CLOUD_CONF:         true,
  USE_QA_SHADOW_CONF:        true,
  USE_QA_CIRRUS_CONF:        true,
  QA_CONF_LEVEL:             2,
  USE_QA_SHADOW_VECTOR:      false,
  SHADOW_PROJECT_DISTANCE_M: 1000,
  CIRRUS_B9_THRESH:          0.008,
  BUFFER_M:                  5000,
  MIN_BAD_COMPONENT_PIXELS:  24,
  USE_DISTANCE_BUFFER:       true,
  DIST_SCALE_M:              30
};

// ============================================================
// MODULES
// ============================================================

var cloudMaskModule = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_CloudMask_TOA_beta');
var rockMaskModule  = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_RockMask');

// ============================================================
// CLIENT-SIDE DATE HELPERS
// ============================================================

function pad2(n) { return (n < 10 ? '0' : '') + n; }

// Format a UTC Date object as YYYY-MM-DD
function formatDate(d) {
  return d.getUTCFullYear() + '-' +
         pad2(d.getUTCMonth() + 1) + '-' +
         pad2(d.getUTCDate());
}

// Returns N_WINDOWS [{wStart, wEnd}] string pairs starting at startDateStr
function computeWindowDates(startDateStr) {
  var parts   = startDateStr.split('-');
  var base    = new Date(Date.UTC(+parts[0], +parts[1] - 1, +parts[2]));
  var msPerDay = 86400000;
  var wins    = [];
  for (var i = 0; i < N_WINDOWS; i++) {
    var s = new Date(base.getTime() + i * MOSAIC_LEN_DAYS * msPerDay);
    var e = new Date(s.getTime()     +     MOSAIC_LEN_DAYS * msPerDay);
    wins.push({ wStart: formatDate(s), wEnd: formatDate(e) });
  }
  return wins;
}

// ============================================================
// IMAGE PROCESSING
// ============================================================

function limitScenes(winRaw, aoi) {
  if (MAX_SCENES_PER_WINDOW === null) return winRaw;
  var scored = winRaw.map(function(img) {
    var area = img.geometry()
      .intersection(aoi, ee.ErrorMargin(30))
      .area(ee.ErrorMargin(30));
    return img.set('aoi_intersect_area', area);
  });
  return scored.sort('aoi_intersect_area', false).limit(MAX_SCENES_PER_WINDOW);
}

// Returns a preprocessing function closed over aoi
function makePreprocess(aoi) {
  return function(img) {
    img = ee.Image(img);
    var out = img.select(['B2','B3','B4','B5','B9','B10','QA_PIXEL'])
                 .copyProperties(img, img.propertyNames());
    out = ee.Image(out).clip(aoi);
    if (APPLY_CLOUD_MASK) { out = cloudMaskModule.applyCloudMask_TOA(out, cloudCfg); }
    if (APPLY_ROCK_MASK)  { out = ee.Image(rockMaskModule.rock_mask(out)); }
    var ndwiIce = out.normalizedDifference(['B2', 'B4']).rename('NDWI_ICE');
    return out.select(['B2','B3','B4']).addBands(ndwiIce);
  };
}

function buildMedoidMosaic(winProcessed) {
  var medianVec = winProcessed.select(MEDOID_DISTANCE_BANDS).median();
  var withScore = winProcessed.map(function(img) {
    img = ee.Image(img);
    var score = img.select(MEDOID_DISTANCE_BANDS)
      .subtract(medianVec)
      .pow(2)
      .reduce(ee.Reducer.sum())
      .multiply(-1)
      .rename('MEDOID_SCORE');
    return img.addBands(score);
  });
  return withScore.qualityMosaic('MEDOID_SCORE')
    .select(['B2','B3','B4','NDWI_ICE']);
}

// Build one 2-day medoid window; returns a masked empty image if no scenes pass
function buildWindow(aoi, wStartStr, wEndStr) {
  var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
    .filterBounds(aoi)
    .filterDate(wStartStr, wEndStr)
    .filter(ee.Filter.gt('SUN_ELEVATION', MIN_SUN_ELEV));
  var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA')
    .filterBounds(aoi)
    .filterDate(wStartStr, wEndStr)
    .filter(ee.Filter.gt('SUN_ELEVATION', MIN_SUN_ELEV));

  var raw = L8.merge(L9);
  if (MAX_CLOUD_COVER !== null) {
    raw = raw.filter(ee.Filter.lte('CLOUD_COVER', MAX_CLOUD_COVER));
  }

  var count   = raw.size();
  var limited = limitScenes(raw, aoi);

  return ee.Image(ee.Algorithms.If(
    count.gte(MIN_SCENES_PER_WINDOW),
    buildMedoidMosaic(limited.map(makePreprocess(aoi))),
    ee.Image(0).updateMask(ee.Image(0))
  ));
}

// ============================================================
// APPLICATION STATE
// ============================================================

var state = {
  aoiName:    null,
  aoi:        null,
  startDate:  null,
  windows:    [],   // [{wStart, wEnd, img}]
  checkboxes: []    // parallel array of ui.Checkbox
};

// ============================================================
// UI
// ============================================================

Map.setOptions('TERRAIN');

var panel = ui.Panel({ style: { width: '300px', padding: '8px' } });

panel.add(ui.Label('Max Melt Medoids', {
  fontWeight: 'bold',
  fontSize:   '15px',
  margin:     '0 0 10px 0'
}));

// -- AOI selector --
panel.add(ui.Label('1.  AOI', { fontWeight: 'bold' }));
var aoiSelect = ui.Select({
  items:       ['OST', 'PTM'],
  placeholder: 'Select AOI…',
  style:       { stretch: 'horizontal' },
  onChange:    onAOIChange
});
panel.add(aoiSelect);

// -- Date selector --
panel.add(ui.Label('2.  Start date', { fontWeight: 'bold', margin: '8px 0 0 0' }));
var dateSelect = ui.Select({
  items:       [],
  placeholder: 'Select start date…',
  style:       { stretch: 'horizontal' },
  onChange:    onDateChange
});
panel.add(dateSelect);

// -- Window checkboxes --
panel.add(ui.Label('3.  Windows to export', { fontWeight: 'bold', margin: '8px 0 0 0' }));
var checkboxPanel = ui.Panel({ style: { margin: '0' } });
panel.add(checkboxPanel);

// -- Export button --
var exportBtn = ui.Button({
  label:   'Export selected',
  style:   { stretch: 'horizontal', margin: '10px 0 0 0' },
  onClick: onExport
});
panel.add(exportBtn);

// -- Status --
var statusLabel = ui.Label('', { color: '#555', fontSize: '12px', margin: '6px 0 0 0' });
panel.add(statusLabel);

ui.root.add(panel);

// ============================================================
// UI CALLBACKS
// ============================================================

function onAOIChange(aoiName) {
  state.aoiName = aoiName;
  state.aoi     = ee.FeatureCollection(AOI_ASSETS[aoiName]).geometry();

  dateSelect.items().reset(START_DATES[aoiName]);
  dateSelect.setValue(null, false);

  resetState();
  Map.centerObject(state.aoi, 7);
}

function onDateChange(startDateStr) {
  state.startDate = startDateStr;
  resetState();
  statusLabel.setValue('Loading windows…');

  var wins = computeWindowDates(startDateStr);

  for (var i = 0; i < wins.length; i++) {
    (function(idx) {
      var win   = wins[idx];
      var img   = buildWindow(state.aoi, win.wStart, win.wEnd);
      var label = 'W' + (idx + 1) + ': ' + win.wStart + ' – ' + win.wEnd;

      state.windows.push({ wStart: win.wStart, wEnd: win.wEnd, img: img });

      Map.addLayer(img, RGB_VIS_PARAMS, label, true);

      var cb = ui.Checkbox({ label: label, value: false });
      state.checkboxes.push(cb);
      checkboxPanel.add(cb);
    })(i);
  }

  statusLabel.setValue(
    'Showing ' + wins.length + ' windows. Check boxes then click Export.'
  );
}

function onExport() {
  if (!state.aoiName || !state.startDate) {
    statusLabel.setValue('Select an AOI and start date first.');
    return;
  }

  var queued = 0;

  for (var i = 0; i < state.checkboxes.length; i++) {
    if (state.checkboxes[i].getValue()) {
      var win  = state.windows[i];
      var name = state.aoiName + '_' + win.wStart + '_' + win.wEnd + '_max_melt';

      Export.image.toDrive({
        image:          win.img.select(['B4', 'B3', 'B2']),
        description:    name,
        fileNamePrefix: name,
        folder:         EXPORT_FOLDER,
        scale:          EXPORT_SCALE,
        crs:            EXPORT_CRS,
        region:         state.aoi,
        maxPixels:      1e13
      });

      queued++;
    }
  }

  statusLabel.setValue(queued > 0
    ? 'Queued ' + queued + ' export task(s) — check the Tasks tab.'
    : 'No windows checked — tick at least one box.');
}

// Clear map layers, checkboxes, and window state
function resetState() {
  Map.layers().reset([]);
  checkboxPanel.clear();
  state.windows    = [];
  state.checkboxes = [];
  statusLabel.setValue('');
}
