/******************************************************* 
 * training image preprocesser version 1.4
 * create:          14 Nov 2025
 * 
 This script was used to preprocess Landsat 08, 09 training scenes. 
 Landsat Scenes are called by ID, corrected with cloud and rock masks, clipped to the relevant AOI. NDWI(ice) band is added in separate script. 
 The workflow is based on the method descripbed in Dell et al 2022, and the rock mask is adappted from the method described by Moussavi et al 2020, with an added NDWI threshold component. 
 This script was used to pre-processing scenes used to generate training data with an additional separate script to add an NDWI_ICE band.
 Byy Luke Urso during MSc work at SU. 
 *******************************************************/

// list of Landsat scene IDs

//****** OSTENFELD SCENES *******  

var SCENE_IDS = [
  'LC08_048244_20180528',
  'LC09_021248_20250510',
  'LC09_020248_20220628',
  'LC09_045244_20230630',
  'LC08_021248_20140707',
  'LC08_026247_20220716',
  'LC08_026247_20180806',

  
];


//****** PETERMANN SCENES *******
/*
var SCENE_IDS = [
  'LC09_036002_20250807',
  'LC08_037002_20180819',
  'LC08_039001_20140705',
  'LC08_037002_20200707',
  'LC08_066242_20190630',
  'LC08_036002_20210617',
  'LC08_037002_20200520',

];
*/


// AOI asset 
//OSTENFELD
var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/OST_STUDYAREA_ROCKMASK_SIMPLE';

//PETERMANN (reg)
//var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/Simple/PG_STUDYAREA_1800m_MinIce_simple'

//PETERMANN (long/short man rockmask)
//var AOI_ASSET = 'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/PTM_LONGAOI_WFJORDC_ROCKMASK';


// destination image collection (or folder) for processed scenes
var DEST_IC_ID = 'projects/vernal-signal-270100/assets/SceneAssets/PreProcessedScenes/PreprocessedTrainingImages/Petermann';

// export parameters
var EXPORT_SCALE_M  = 30;      // Landsat TOA scale
var MAX_PIXELS      = 1e13;    


/*******************************************************
 * MODULES
 *******************************************************/

var cloud_mask = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_CloudMask');  
var rock_mask  = require('users/LukeUrso/GEEScripts:ProcessingModules/LC08_RockMask');


/*******************************************************
 * AOI
 *******************************************************/
var aoi_fc   = ee.FeatureCollection(AOI_ASSET);
var aoi_geom = aoi_fc.geometry();

//Map.centerObject(aoi_fc, 8);
Map.addLayer(aoi_geom, {color: '1f1e25'}, 'AOI');

//alt AOI display
//var aoi_alt = ee.FeatureCollection('projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/PTM_LONGAOI_WFJORDC_ROCKMASK');
//Map.addLayer(aoi_alt, {color: 'blue'}, 'AOI_alt');

/*******************************************************
 * METADATA TO KEEP
 *******************************************************/
var PROPS_TO_COPY = [
  'LANDSAT_PRODUCT_ID',
  'LANDSAT_SCENE_ID',
  'COLLECTION_CATEGORY',   // tier info
  'SENSOR_ID',
  'SPACECRAFT_ID',
  'WRS_PATH',
  'WRS_ROW',
  'system:time_start',
  'CLOUD_COVER',
  'SUN_ELEVATION'
];


/*******************************************************
 * LOAD SCENES FROM LIST OF IDs
 *******************************************************/

// build a JS array of ee.Image objects from product IDs
var srcImages = SCENE_IDS.map(function(id) {
  var mission = id.slice(0, 4);   // 'LC08' or 'LC09'
  var colId   = 'LANDSAT/' + mission + '/C02/T1_TOA/' + id;

  var img = ee.Image(colId)
    .set('LANDSAT_PRODUCT_ID', id)
    .set('SENSOR_ID', mission); 

  return img;
});

var srcIC = ee.ImageCollection(srcImages);

print('Loaded scenes:', srcIC);


/*******************************************************
 * Apply masks + clip
 *******************************************************/
function applyMasksAndClip(img) {
  img = cloud_mask.maskClouds(img);

  img = rock_mask.rock_mask(img);

  img = img.clip(aoi_geom);

  img = img.copyProperties(img, PROPS_TO_COPY);

  return img;
}


/*******************************************************
 * PROCESS COLLECTION
 *******************************************************/

// apply masks & clipping
var procIC = srcIC.map(applyMasksAndClip);

// check in the map
var list = procIC.toList(procIC.size());

var pids = procIC.aggregate_array('LANDSAT_PRODUCT_ID');

pids.evaluate(function(pidArray) {
  pidArray.forEach(function(pid, i) {
    var img = ee.Image(list.get(i));
    Map.addLayer(img, {bands:['B4','B3','B2'], min:0, max:1.0}, pid);
  });
});


print('Source collection size:', srcIC.size());
print('Processed collection size:', procIC.size());


/*******************************************************
 * EXPORT TO GEE ASSETS: QUEUE EACH SCENE
 *******************************************************/
/*
var procList = procIC.toList(procIC.size());

var productIds = procIC.aggregate_array('LANDSAT_PRODUCT_ID').getInfo();

print('Product IDs for export:', productIds);

for (var i = 0; i < productIds.length; i++) {
  var img  = ee.Image(procList.get(i));
  var pid  = productIds[i] || ('scene_' + i);  
  var name = pid.replace(/\s+/g, '_');         

  // Build assetId inside dest. collection
  var assetId = DEST_IC_ID + '/' + 'PTM_PTI_' + name;

  Export.image.toAsset({
    image: img,
    description: 'PTM_PTI_' + name,
    assetId: assetId,
    region: aoi_geom,
    scale: EXPORT_SCALE_M,
    maxPixels: MAX_PIXELS
  });
}
*/

/*******************************************************
 * EXPORT: QUEUE EACH SCENE TO GOOGLE DRIVE
 *******************************************************/
var procList = procIC.toList(procIC.size());

var productIds = procIC.aggregate_array('LANDSAT_PRODUCT_ID').getInfo();

print('Product IDs for Drive export:', productIds);

// loop client-side and start an export per scene
for (var i = 0; i < productIds.length; i++) {
  // cast all bands to Float32 to avoid mixed-type error
  var img  = ee.Image(procList.get(i)).toFloat();
  var pid  = productIds[i] || ('scene_' + i);   
  var name = pid.replace(/\s+/g, '_');          

  Export.image.toDrive({
    image: img,
    description: 'PTM_PTI_' + name,             
    folder: 'TrainingImages',                  
    fileNamePrefix: 'PTM_PTI_' + name,          
    region: aoi_geom,
    scale: EXPORT_SCALE_M,
    maxPixels: MAX_PIXELS
    //
    // crs: 'EPSG:3413',
    // fileFormat: 'GeoTIFF'
  });
}





