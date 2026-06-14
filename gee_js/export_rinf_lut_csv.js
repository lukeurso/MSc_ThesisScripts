// ============================================================
// export_rinf_lut_csv.js
//
// Exports the pre-built Rinf LUT assets (OST + PTM) to Google
// Drive as CSV files for local use with add_depth_band.py.
//
// All properties are exported so the Python script can identify
// the exact column names present in the asset.  A human-readable
// date field and time_start_ms are added from system:time_start.
// ============================================================

// ---- USER SETTINGS ----
var RINF_LUT_OST = 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_OST_L8L9_v1';
var RINF_LUT_PTM = 'projects/vernal-signal-270100/assets/deepwater_lookups/Rinf_LUT_PTM_L8L9_v1';

var DRIVE_FOLDER = 'GEE_Exports';  // change to your preferred Drive folder

// -------------------------------------------------------
// Add a readable date string from system:time_start.
// All other properties are passed through unchanged so no
// columns are silently dropped by a .select() call.
// -------------------------------------------------------
function addDateField(feat) {
  var ms = ee.Number(feat.get('system:time_start'));
  var dateStr = ee.Date(ms).format('YYYY-MM-dd');
  return feat.set({
    'date':          dateStr,
    'time_start_ms': ms
  });
}

function prepLut(assetId) {
  // No .select() — export every property so we can inspect actual column names.
  return ee.FeatureCollection(assetId).map(addDateField);
}

var lutOST = prepLut(RINF_LUT_OST);
var lutPTM = prepLut(RINF_LUT_PTM);

// Print a sample row to the Console so column names are visible before export.
print('OST LUT size:', lutOST.size());
print('OST ok==1 rows:', lutOST.filter(ee.Filter.eq('ok', 1)).size());
print('OST sample row (inspect for Rinf column names):', lutOST.first());
print('PTM LUT size:', lutPTM.size());
print('PTM ok==1 rows:', lutPTM.filter(ee.Filter.eq('ok', 1)).size());
print('PTM sample row (inspect for Rinf column names):', lutPTM.first());

// ---- EXPORT OST ----
Export.table.toDrive({
  collection:     lutOST,
  description:    'Rinf_LUT_OST_L8L9_v1_csv',
  fileNamePrefix: 'Rinf_LUT_OST_L8L9_v1',
  folder:         DRIVE_FOLDER,
  fileFormat:     'CSV'
});

// ---- EXPORT PTM ----
Export.table.toDrive({
  collection:     lutPTM,
  description:    'Rinf_LUT_PTM_L8L9_v1_csv',
  fileNamePrefix: 'Rinf_LUT_PTM_L8L9_v1',
  folder:         DRIVE_FOLDER,
  fileFormat:     'CSV'
});
