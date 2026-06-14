/**** ================================
 *  ERA5-Land Daily Climate Export v1
 *
 *  Extracts daily ERA5-Land:
 *    - 2m air temperature (mean, converted K → °C)
 *    - Downward shortwave radiation (daily sum J/m² → mean W/m²)
 *
 *  Sites : OST and PTM 1500m AOIs
 *  Period: 2013–2025, May 01 – Sep 30 each year
 *
 *  Output: one merged CSV exported to Google Drive
 *  ================================ ****/

// ------------------------
// USER SETTINGS
// ------------------------
var START_YEAR = 2013;
var END_YEAR   = 2025;

var EXPORT_FOLDER      = 'GEE_ERA5Land_Climate';
var EXPORT_DESCRIPTION = 'ERA5Land_T2m_SWdown_OST_PTM_2013_2025_MaySep';

// ERA5-Land daily aggregated collection
var ERA5_LAND_ID = 'ECMWF/ERA5_LAND/DAILY_AGGR';

// Band names in ECMWF/ERA5_LAND/DAILY_AGGR
//   temperature_2m                      : mean daily 2m air temperature (K)
//   surface_solar_radiation_downwards_sum: daily accumulated SW-down (J m⁻²)
var TEMP_BAND = 'temperature_2m';
var RAD_BAND  = 'surface_solar_radiation_downwards_sum';

// ERA5-Land native pixel size (≈0.1° at the equator → 11,132 m)
var ERA5_SCALE = 11132;

// AOI assets (1500m)
var OST_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/OST_AOI_1500m';
var PTM_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/1500m_AOIs/PTM_AOI_1500m';

// ------------------------
// LOAD AOIs
// ------------------------
var ostGeom = ee.FeatureCollection(OST_ASSET).geometry();
var ptmGeom = ee.FeatureCollection(PTM_ASSET).geometry();

// ------------------------
// LOAD ERA5-LAND (May–Sep, all years)
// ------------------------
// filterDate end is exclusive, so use Oct 01 of END_YEAR to include Sep 30.
var era5 = ee.ImageCollection(ERA5_LAND_ID)
  .filterDate(START_YEAR + '-05-01', END_YEAR + '-10-01')
  .filter(ee.Filter.calendarRange(5, 9, 'month'))
  .select([TEMP_BAND, RAD_BAND]);

print('ERA5-Land: images selected', era5.size());

// ------------------------
// EXTRACT DAILY MEANS PER AOI
// ------------------------
function extractDailyValues(aoiName, aoiGeom) {
  return ee.FeatureCollection(era5.map(function(img) {
    var date  = ee.Date(img.get('system:time_start'));
    var stats = img.reduceRegion({
      reducer:   ee.Reducer.mean(),
      geometry:  aoiGeom,
      scale:     ERA5_SCALE,
      maxPixels: 1e9,
      bestEffort: true
    });

    var tempK  = ee.Number(stats.get(TEMP_BAND));
    var radJm2 = ee.Number(stats.get(RAD_BAND));

    return ee.Feature(null, {
      'site':       aoiName,
      'date':       date.format('YYYY-MM-dd'),
      'year':       date.get('year'),
      'doy':        date.getRelative('day', 'year').add(1), // 0-indexed → DOY 1-366
      'temp_2m_C':  tempK.subtract(273.15),                 // K → °C
      'srad_Wm2':   radJm2.divide(86400)                    // J/m² daily sum → mean W/m²
    });
  }));
}

// ------------------------
// MERGE BOTH SITES & SORT
// ------------------------
var merged = extractDailyValues('OST', ostGeom)
               .merge(extractDailyValues('PTM', ptmGeom))
               .sort('date');

print('Total features to export:', merged.size());

// ------------------------
// EXPORT TO GOOGLE DRIVE
// ------------------------
Export.table.toDrive({
  collection:     merged,
  description:    EXPORT_DESCRIPTION,
  folder:         EXPORT_FOLDER,
  fileNamePrefix: EXPORT_DESCRIPTION,
  fileFormat:     'CSV'
});

print('Task submitted — open the Tasks tab to run the export.');
