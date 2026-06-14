/**** ================================
 *  ERA5-Land Daily Climate Export v1
 *
 *  Extracts daily ERA5-Land:
 *    - 2m air temperature (mean, converted K → °C)
 *    - Downward shortwave radiation (daily sum J/m² → mean W/m²)
 *
 *  Spatial units: aspect-elevation bins (OST and PTM)
 *  Period: 2013–2025, May 01 – Sep 30 each year
 *
 *  Output: one merged CSV exported to Google Drive,
 *          one row per (bin_id × date)
 *  ================================ ****/

// ------------------------
// USER SETTINGS
// ------------------------
var START_YEAR = 2013;
var END_YEAR   = 2025;

var EXPORT_FOLDER      = 'GEE_ERA5Land_Climate';
var EXPORT_DESCRIPTION = 'ERA5Land_T2m_SWdown_ElevBins_OST_PTM_2013_2025_MaySep';

// ERA5-Land daily aggregated collection
var ERA5_LAND_ID = 'ECMWF/ERA5_LAND/DAILY_AGGR';

// Band names in ECMWF/ERA5_LAND/DAILY_AGGR
//   temperature_2m                       : mean daily 2m air temperature (K)
//   surface_solar_radiation_downwards_sum : daily accumulated SW-down (J m⁻²)
var TEMP_BAND = 'temperature_2m';
var RAD_BAND  = 'surface_solar_radiation_downwards_sum';

// ERA5-Land native pixel size (≈0.1° at the equator → 11,132 m)
var ERA5_SCALE = 11132;

// Aspect-elevation bin assets (bin_id field format: {AOI}{upper_elv_hundreds}{sector_code})
var OST_BINS_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/OST_aspect_elevation_bins';
var PTM_BINS_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/PTM_aspect_elevation_bins';

// ------------------------
// LOAD ELEVATION BINS
// ------------------------
var ostBins = ee.FeatureCollection(OST_BINS_ASSET);
var ptmBins = ee.FeatureCollection(PTM_BINS_ASSET);

var allBins = ostBins.merge(ptmBins);

print('OST bins:', ostBins.size());
print('PTM bins:', ptmBins.size());
print('Total elevation bins:', allBins.size());

// ------------------------
// LOAD ERA5-LAND (May–Sep, all years)
// ------------------------
// filterDate end is exclusive; Oct 01 of END_YEAR captures Sep 30.
var era5 = ee.ImageCollection(ERA5_LAND_ID)
  .filterDate(START_YEAR + '-05-01', END_YEAR + '-10-01')
  .filter(ee.Filter.calendarRange(5, 9, 'month'))
  .select([TEMP_BAND, RAD_BAND]);

print('ERA5-Land images selected:', era5.size());

// ------------------------
// EXTRACT DAILY MEANS PER ELEVATION BIN
// ------------------------
// reduceRegions handles all bins in one server-side pass per image.
// Each mapped image returns a FeatureCollection; flatten() merges them all.
var results = era5.map(function(img) {
  var date = ee.Date(img.get('system:time_start'));

  return img.reduceRegions({
    collection: allBins,
    reducer:    ee.Reducer.mean(),
    scale:      ERA5_SCALE,
    tileScale:  4
  }).map(function(f) {
    // Build a null-geometry feature so CSV rows stay as pure attributes.
    return ee.Feature(null, {
      'bin_id':    f.get('bin_id'),
      'date':      date.format('YYYY-MM-dd'),
      'year':      date.get('year'),
      'doy':       date.getRelative('day', 'year').add(1), // 0-indexed → DOY 1-366
      'temp_2m_C': ee.Number(f.get(TEMP_BAND)).subtract(273.15), // K → °C
      'srad_Wm2':  ee.Number(f.get(RAD_BAND)).divide(86400)      // J/m² sum → mean W/m²
    });
  });
}).flatten();

print('Total features to export:', results.size());

// ------------------------
// EXPORT TO GOOGLE DRIVE
// ------------------------
Export.table.toDrive({
  collection:     results,
  description:    EXPORT_DESCRIPTION,
  folder:         EXPORT_FOLDER,
  fileNamePrefix: EXPORT_DESCRIPTION,
  fileFormat:     'CSV'
});

print('Task submitted — open the Tasks tab to run the export.');
