/*** ====================================================================
 *  drainage_event_viewer.js
 *
 *  Interactive viewer for rapid lake drainage events.
 *
 *  Reads the drainage_events_points asset (exported from
 *  analyze_drainage_events.py) and for each event displays
 *  Landsat 8/9 TOA quality mosaics (NDWI_ICE) for the pre- and
 *  post-drainage observation windows.
 *
 *  NDWI_ICE = normalizedDifference(['B2', 'B4'])  (Blue − Red / Blue + Red)
 *  Mosaics use qualityMosaic on NDWI_ICE (highest value per pixel wins).
 *
 *  Controls
 *  --------
 *  · AOI filter      - restrict list to PTM or OST events
 *  · Type filter     - restrict to full or partial drainage events
 *  · Dropdown        - jump directly to any listed event
 *  · Prev / Next     - step through the filtered list one event at a time
 *
 *  Map layers (toggled in the standard layer panel)
 *  --------
 *  · Pre  RGB             (on by default)
 *  · Post RGB             (on by default)
 *  · Pre  NDWIice         (off by default)
 *  · Post NDWIice         (off by default)
 *  · Lake polygon outline (on by default, white)
 *  · Event point          (on by default, yellow)
 *
 *  Setup
 *  -----
 *  1. Upload drainage_events_points.shp to your GEE asset folder.
 *  2. Update ASSET_PATH below.
 *  3. Paste into the GEE Code Editor and click Run.
 * ==================================================================== ***/


// ------------------------
// USER SETTINGS
// ------------------------

//var ASSET_PATH = 'projects/vernal-signal-270100/assets/drainage_tracking/buffered_drainage_events_points';
var ASSET_PATH = buffered_drainage_events_points; //#imported asset name

// Persistent lake polygon assets (one per AOI)
//var POLYGONS_PTM = 'projects/vernal-signal-270100/assets/drainage_tracking/buffered_PTM_persistent_lake_polygons';
//var POLYGONS_OST = 'projects/vernal-signal-270100/assets/drainage_tracking/buffered_OST_persistent_lake_polygons';
var POLYGONS_PTM = buffered_PTM_persistent_lake_polygons; //#imported asset name
var POLYGONS_OST = buffered_OST_persistent_lake_polygons; //#imported asset name

// Landsat Collection 2, Tier 1 TOA (same collections used by export pipeline)
var L8_TOA = 'LANDSAT/LC08/C02/T1_TOA';
var L9_TOA = 'LANDSAT/LC09/C02/T1_TOA';

// Minimum sun elevation (degrees) - matches export pipeline setting
var MIN_SUN_ELEV = 15;

// Days added either side of each window's start/end date when searching
// for scenes.  Windows are ~2 days; 1 day handles boundary edge cases.
var DATE_BUFFER_DAYS = 1;

// Years to inspect - only events whose post_start falls in one of these years
// are loaded.  Add or remove years from the array and re-run to change the set. 2020, 2019, 2018, 2017, 2016, 2015, 2014
var YEARS = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014];

// Lake ID range filter - lakes are ordered largest-first so this is also
// a de-facto size filter.  Values are the numeric part only (1 = P001 / C001).
var ID_RANGE_PTM = [1, 100];   // P001 – P050
var ID_RANGE_OST = [1, 100];   // C001 – C050

// Map zoom level when centring on an event
var EVENT_ZOOM = 12;

// Default map centre (overridden once events load)
var DEFAULT_LON  = -52.0;
var DEFAULT_LAT  =  80.0;
var DEFAULT_ZOOM =  6;


// ------------------------
// CLOUD MASK  (QA_PIXEL bitmask - simplified from project module)
// ------------------------
// For full pipeline-accurate masking swap this function body with:
//   var cloud_mask = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_CloudMask_TOA_beta');
//   return cloud_mask.applyCloudMask_TOA(image, CLOUD_MASK_CFG);

function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 3).eq(0)   // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))        // cloud shadow
    .and(qa.bitwiseAnd(1 << 1).eq(0))        // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0));       // cirrus
  return image.updateMask(clear);
}


// ------------------------
// LANDSAT HELPERS
// ------------------------

function addNDWIice(image) {
  // NDWI_ICE = (Blue − Red) / (Blue + Red)  - matches export pipeline
  return image.addBands(
    image.normalizedDifference(['B2', 'B4']).rename('NDWI_ICE')
  );
}

function buildMosaic(dateStart, dateEnd, point) {
  var start = ee.Date(dateStart).advance(-DATE_BUFFER_DAYS, 'day');
  var end   = ee.Date(dateEnd).advance(  DATE_BUFFER_DAYS, 'day');

  var col = ee.ImageCollection(L8_TOA)
    .merge(ee.ImageCollection(L9_TOA))
    .filterDate(start, end)
    .filterBounds(point)
    .filter(ee.Filter.gt('SUN_ELEVATION', MIN_SUN_ELEV))
    .map(maskClouds)
    .map(addNDWIice);

  // Quality mosaic: pixel with highest NDWI_ICE value wins
  return col.qualityMosaic('NDWI_ICE');
}


// ------------------------
// VISUALISATION PARAMETERS
// ------------------------

var VIS_NDWI = {
  bands:   ['NDWI_ICE'],
  min:     -0.3,
  max:      0.5,
  palette: ['8b0000', 'f0f0f0', '0055cc'],  // dry → bare ice → open water
};

var VIS_RGB = {
  bands: ['B4', 'B3', 'B2'],   // TOA: Red, Green, Blue
  min:   0.0,
  max:   1.0,
  gamma: 1.0,
};


// ------------------------
// LAKE POLYGON ASSETS
// ------------------------

// Loaded once at startup; filtered per event in displayEvent
var lakePolygons = {
  PTM: ee.FeatureCollection(POLYGONS_PTM),
  OST: ee.FeatureCollection(POLYGONS_OST),
};


// ------------------------
// STATE
// ------------------------

var allFeatures   = [];         // full client-side feature array (from evaluate)
var filtered      = [];         // subset matching current filters
var currentIdx    = 0;
var selectedYears = YEARS.slice();  // years currently checked in the UI


// ------------------------
// MAP PANEL
// ------------------------

var mapPanel = ui.Map();
mapPanel.setOptions('TERRAIN');
mapPanel.setCenter(DEFAULT_LON, DEFAULT_LAT, DEFAULT_ZOOM);
mapPanel.style().set({ cursor: 'crosshair' });


// ------------------------
// CONTROL PANEL - LAYOUT
// ------------------------

var controlPanel = ui.Panel({
  style: { width: '310px', padding: '8px', backgroundColor: '#f8f8f8' }
});

controlPanel.add(ui.Label('Drainage Event Viewer', {
  fontWeight: 'bold', fontSize: '16px', margin: '0 0 6px 0',
}));

// AOI filter
controlPanel.add(ui.Label('AOI', { fontWeight: 'bold', margin: '6px 0 2px 0' }));
var aoiSelect = ui.Select({
  items: ['All', 'PTM', 'OST'], value: 'All',
  onChange: applyFilters,
  style: { stretch: 'horizontal' },
});
controlPanel.add(aoiSelect);

// Drain type filter
controlPanel.add(ui.Label('Drain type', { fontWeight: 'bold', margin: '6px 0 2px 0' }));
var typeSelect = ui.Select({
  items: ['All', 'full', 'partial'], value: 'All',
  onChange: applyFilters,
  style: { stretch: 'horizontal' },
});
controlPanel.add(typeSelect);

// Year filter (one checkbox per entry in YEARS)
controlPanel.add(ui.Label('Year(s)', { fontWeight: 'bold', margin: '6px 0 2px 0' }));
var yearCheckPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style:  { margin: '0 0 4px 0', padding: '0' },
});
YEARS.forEach(function(yr) {
  yearCheckPanel.add(ui.Checkbox({
    label:    String(yr),
    value:    true,
    onChange: function(checked) {
      if (checked) {
        if (selectedYears.indexOf(yr) === -1) selectedYears.push(yr);
      } else {
        selectedYears = selectedYears.filter(function(y) { return y !== yr; });
      }
      applyFilters();
    },
    style: { margin: '0 8px 0 0', fontSize: '12px' },
  }));
});
controlPanel.add(yearCheckPanel);

// Event count
var countLabel = ui.Label('Loading…', {
  color: '#666666', fontSize: '11px', margin: '4px 0 6px 0',
});
controlPanel.add(countLabel);

// Event selector
controlPanel.add(ui.Label('Event', { fontWeight: 'bold', margin: '2px 0 2px 0' }));
var eventSelect = ui.Select({
  items: ['Loading…'], value: 'Loading…',
  onChange: function(value) {
    var idx = eventSelect.items().indexOf(value);
    if (idx >= 0) { currentIdx = idx; displayEvent(idx); }
  },
  style: { stretch: 'horizontal' },
});
controlPanel.add(eventSelect);

// Prev / Next buttons
var prevBtn = ui.Button({
  label: '← Prev', onClick: function() { navigateTo(currentIdx - 1); },
  style: { margin: '4px 4px 4px 0' },
});
var nextBtn = ui.Button({
  label: 'Next →', onClick: function() { navigateTo(currentIdx + 1); },
  style: { margin: '4px 0 4px 4px' },
});
controlPanel.add(ui.Panel(
  [prevBtn, nextBtn], ui.Panel.Layout.flow('horizontal')
));

// Event info
controlPanel.add(ui.Label('Event details', { fontWeight: 'bold', margin: '8px 0 2px 0' }));
var infoPanel = ui.Panel({
  style: {
    backgroundColor: '#ffffff', border: '1px solid #dddddd',
    padding: '6px', margin: '0 0 6px 0',
  }
});
controlPanel.add(infoPanel);

// Status
var statusLabel = ui.Label('', { color: '#888888', fontSize: '11px' });
controlPanel.add(statusLabel);

// NDWIice legend
controlPanel.add(ui.Label('NDWI_ICE', { fontWeight: 'bold', margin: '8px 0 3px 0' }));
controlPanel.add(makeLegend());


// ------------------------
// LEGEND
// ------------------------

function makeLegend() {
  var items = [
    { color: '#0055cc', label: 'Open water  (high NDWI_ICE)' },
    { color: '#f0f0f0', label: 'Bare ice / snow  (≈ 0)' },
    { color: '#8b0000', label: 'Dry / no water  (negative)' },
  ];
  var panel = ui.Panel({ style: { margin: '0' } });
  items.forEach(function(item) {
    panel.add(ui.Panel(
      [
        ui.Label('', {
          backgroundColor: item.color,
          padding: '6px 10px', margin: '2px 6px 2px 0',
        }),
        ui.Label(item.label, { fontSize: '11px', margin: '2px 0' }),
      ],
      ui.Panel.Layout.flow('horizontal')
    ));
  });
  return panel;
}


// ------------------------
// DISPLAY LOGIC
// ------------------------

function navigateTo(idx) {
  if (filtered.length === 0) return;
  currentIdx = Math.max(0, Math.min(filtered.length - 1, idx));
  // Update selector without triggering onChange (avoid double-render)
  eventSelect.setValue(eventSelect.items()[currentIdx], false);
  displayEvent(currentIdx);
}

function displayEvent(idx) {
  if (filtered.length === 0 || idx < 0 || idx >= filtered.length) return;

  var feat  = filtered[idx];
  var props = feat.properties;

  statusLabel.setValue('Computing mosaics…');
  mapPanel.layers().reset();

  // Derive AOI from lake_id prefix; filter polygon FC to this lake
  var aoiKey   = (props.lake_id.charAt(0) === 'P') ? 'PTM' : 'OST';
  var lakePoly = lakePolygons[aoiKey].filter(ee.Filter.eq('lake_id', props.lake_id));

  // Centre map on polygon centroid - lightweight two-number evaluate,
  // no geometry serialisation issues
  lakePoly.first().geometry().centroid(1).coordinates()
    .evaluate(function(coords) {
      if (coords) mapPanel.setCenter(coords[0], coords[1], EVENT_ZOOM);
    });

  // Build mosaics using the lake polygon geometry for filterBounds
  var lakeGeom   = lakePoly.geometry();
  var preMosaic  = buildMosaic(props.pre_start,  props.pre_end,  lakeGeom);
  var postMosaic = buildMosaic(props.post_start, props.post_end, lakeGeom);

  // RGB on by default; NDWIice off by default
  mapPanel.addLayer(preMosaic,  VIS_RGB,  'Pre  RGB       (' + props.pre_start  + ')',  true);
  mapPanel.addLayer(postMosaic, VIS_RGB,  'Post RGB       (' + props.post_start + ')',  true);
  mapPanel.addLayer(preMosaic,  VIS_NDWI, 'Pre  NDWI_ICE  (' + props.pre_start  + ')',  false);
  mapPanel.addLayer(postMosaic, VIS_NDWI, 'Post NDWI_ICE  (' + props.post_start + ')',  false);

  // Lake polygon outline
  mapPanel.addLayer(
    lakePoly.style({ color: 'dd4b39', fillColor: '00000000', width: 2 }),
    {}, props.lake_id + '  outline', true
  );

  // Centroid marker (yellow dot)
  mapPanel.addLayer(
    ee.FeatureCollection([ee.Feature(lakePoly.first().geometry().centroid(1))])
      .style({ color: 'ffff00', pointSize: 5, width: 2 }),
    {}, props.lake_id + '  ' + props.drain_type, true
  );

  updateInfoPanel(props, idx);
  statusLabel.setValue('');
}

function updateInfoPanel(p, idx) {
  infoPanel.clear();

  // Format volume loss as percentage
  var lossStr = (p.vol_loss !== undefined && p.vol_loss !== null)
    ? (p.vol_loss * 100).toFixed(1) + '%' : '-';

  // Chain event string
  var chainStr = (p.chain_evt === true || p.chain_evt === 'true' || p.chain_evt === 1)
    ? 'Yes  (group ' + p.chain_grp + ')' : 'No';

  var rows = [
    ['Event',         (idx + 1) + ' / ' + filtered.length],
    ['Lake',          p.lake_id],
    ['Type',          p.drain_type],
    ['Pre window',    p.pre_start  + ' – ' + p.pre_end],
    ['Post window',   p.post_start + ' – ' + p.post_end],
    ['Pre vol (m³)',  p.pre_vol_m3  !== undefined ? Number(p.pre_vol_m3).toLocaleString()  : '-'],
    ['Post vol (m³)', p.pst_vol_m3  !== undefined ? Number(p.pst_vol_m3).toLocaleString()  : '-'],
    ['Vol loss',      lossStr],
    ['Window gap',    p.window_gap !== undefined ? p.window_gap + ' window(s)' : '-'],
    ['Confidence',    p.conf_score !== undefined ? p.conf_score : '-'],
    ['Chain event',   chainStr],
  ];

  rows.forEach(function(row) {
    infoPanel.add(ui.Panel(
      [
        ui.Label(row[0] + ':', { fontWeight: 'bold', fontSize: '11px', width: '85px' }),
        ui.Label(String(row[1]), { fontSize: '11px' }),
      ],
      ui.Panel.Layout.flow('horizontal'),
      { margin: '1px 0' }
    ));
  });
}


// ------------------------
// FILTERING
// ------------------------

function applyFilters() {
  var aoi  = aoiSelect.getValue();
  var type = typeSelect.getValue();

  filtered = allFeatures.filter(function(f) {
    var p = f.properties;
    // AOI derived from lake_id prefix: P→PTM, C→OST
    var featureAoi  = (p.lake_id.charAt(0) === 'P') ? 'PTM' : 'OST';
    var featureYear = parseInt(p.post_start.substring(0, 4), 10);
    return ((aoi  === 'All') || (featureAoi  === aoi))
        && ((type === 'All') || (p.drain_type === type))
        && (selectedYears.length === 0 || selectedYears.indexOf(featureYear) !== -1);
  });

  // Rebuild selector labels
  var labels = filtered.map(function(f, i) {
    var p = f.properties;
    var loss = (p.vol_loss !== undefined && p.vol_loss !== null)
      ? (p.vol_loss * 100).toFixed(0) + '%' : '?';
    return (i + 1) + '.  ' + p.lake_id + '  |  ' + p.drain_type +
           '  |  ' + p.post_start + '  (' + loss + ' loss)';
  });

  countLabel.setValue(filtered.length + ' event(s) shown');

  if (labels.length === 0) {
    eventSelect.items().reset(['No events match filter']);
    mapPanel.layers().reset();
    infoPanel.clear();
    return;
  }

  eventSelect.items().reset(labels);
  currentIdx = 0;
  eventSelect.setValue(labels[0], false);
  displayEvent(0);
}


// ------------------------
// INITIALISE
// ------------------------

// Zero-pad a number to 3 digits for lake_id string comparison ("P001", "P050" …)
function padN(n) { return n < 10 ? '00' + n : n < 100 ? '0' + n : '' + n; }

function loadEvents() {
  allFeatures = [];
  filtered = [];
  mapPanel.layers().reset();
  infoPanel.clear();
  countLabel.setValue('Loading…');

  // Load the contiguous range that spans all requested years; the year checkboxes
  // then handle client-side filtering within that range.
  var _minYear = YEARS.reduce(function(a, b) { return a < b ? a : b; });
  var _maxYear = YEARS.reduce(function(a, b) { return a > b ? a : b; });

  var idFilter = ee.Filter.or(
    ee.Filter.and(
      ee.Filter.gte('lake_id', 'P' + padN(ID_RANGE_PTM[0])),
      ee.Filter.lte('lake_id', 'P' + padN(ID_RANGE_PTM[1]))
    ),
    ee.Filter.and(
      ee.Filter.gte('lake_id', 'C' + padN(ID_RANGE_OST[0])),
      ee.Filter.lte('lake_id', 'C' + padN(ID_RANGE_OST[1]))
    )
  );

  var query = ee.FeatureCollection(ASSET_PATH)
    .filter(ee.Filter.gte('post_start', _minYear + '-01-01'))
    .filter(ee.Filter.lt( 'post_start', (_maxYear + 1) + '-01-01'))
    .filter(idFilter);

  // Evaluate only property columns - avoids serialising the projected
  // geometries (EPSG:3413), which causes a full FeatureCollection
  // evaluate() to return undefined.  Geometry is handled server-side
  // via the persistent lake polygon assets throughout.
  query.size().evaluate(function(n, sizeErr) {
    if (sizeErr) {
      countLabel.setValue('Cannot reach asset: ' + sizeErr);
      return;
    }
    if (n === 0) {
      countLabel.setValue('Asset exists but has 0 features - check the upload.');
      return;
    }

    countLabel.setValue('Loading ' + n + ' event(s)…');

    var propDict = ee.Dictionary({
      lake_id:    query.aggregate_array('lake_id'),
      drain_type: query.aggregate_array('drain_type'),
      pre_start:  query.aggregate_array('pre_start'),
      pre_end:    query.aggregate_array('pre_end'),
      post_start: query.aggregate_array('post_start'),
      post_end:   query.aggregate_array('post_end'),
      pre_vol_m3: query.aggregate_array('pre_vol_m3'),
      pst_vol_m3: query.aggregate_array('pst_vol_m3'),
      vol_loss:   query.aggregate_array('vol_loss'),
      window_gap: query.aggregate_array('window_gap'),
      conf_score: query.aggregate_array('conf_score'),
      chain_evt:  query.aggregate_array('chain_evt'),
      chain_grp:  query.aggregate_array('chain_grp'),
    });

    propDict.evaluate(function(dict, dictErr) {
      if (dictErr) {
        countLabel.setValue('Error loading properties: ' + dictErr);
        print('propDict error:', dictErr);
        return;
      }

      // Reconstruct feature-like objects from the parallel property arrays
      allFeatures = dict.lake_id.map(function(_, i) {
        return {
          properties: {
            lake_id:    dict.lake_id[i],
            drain_type: dict.drain_type[i],
            pre_start:  dict.pre_start[i],
            pre_end:    dict.pre_end[i],
            post_start: dict.post_start[i],
            post_end:   dict.post_end[i],
            pre_vol_m3: dict.pre_vol_m3[i],
            pst_vol_m3: dict.pst_vol_m3[i],
            vol_loss:   dict.vol_loss[i],
            window_gap: dict.window_gap[i],
            conf_score: dict.conf_score[i],
            chain_evt:  dict.chain_evt[i],
            chain_grp:  dict.chain_grp[i],
          }
        };
      });

      allFeatures.sort(function(a, b) {
        var la = a.properties.lake_id, lb = b.properties.lake_id;
        if (la < lb) return -1;
        if (la > lb) return  1;
        return a.properties.post_start <= b.properties.post_start ? -1 : 1;
      });

      applyFilters();
    });
  });
}

loadEvents();


// ------------------------
// ROOT LAYOUT
// ------------------------

ui.root.clear();
ui.root.add(ui.SplitPanel({
  firstPanel:  controlPanel,
  secondPanel: mapPanel,
  orientation: 'horizontal',
}));
