/**** script info 
 * generates training data for random forest classifer of supra meltwater 
 * filters pixels by NDWI_ice > 0.01
 * applies k-means clusterer to group pixels from pre-processed training images
 * applies sub clusterer is run for ambiguous groups
 * defines main and sub clusters as lake, slush, or ice (used as 'other' class)
 * samples classes on stratified per-aoi, per-class basis 
 * exports  sampled classified pixels as CSV for training data used in RF_ prefix scripts
 * 
 * adpted workflow from Dell et al. 2022

/**** version info ===========================
 * training data generation script version 5.2
 * create:           14 jan 2026
 * derived from KMeans_ClassDeff_UI_Stats.js
 * added: define classes, sampling-export
 * ---
 * last major edit:  27 jan 2026  
 * modified AOI defs to highlight/search for clusters
 * ---
 * this script was created by Luke Urso as part of MSc thesis 
 ===============================================****/
 
/**** ================================
 *    PARAMS
 *  ================================ ****/

// require module: ndwi_ice (B3, B4) threshold module                                                                            
var extra_filter = require('users/LukeUrso/GEEScripts:ProcessingModules/NDWI_filter');  // NDWI_ICE filter > 0.1 filter

// load pre-processed training images from imported image assets
var TRAINING_IMAGES = [
  image0, image1, image2, image3, image4, image5, image6,         //ostenfeld scenes 

  image7, image8, image10, image11, image12, image13              //petermann scenes image9,
 
];

// set variable for number of training images to allow tests with sub sets (devopment hold over)
var N_MAIN = TRAINING_IMAGES.length 

// define AOIs from shapefile GEE assets 
var PTM_AOI = ee.FeatureCollection(
  'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/PTM_LONGAOI_WFJORDC_ROCKMASK'
);
var OST_AOI = ee.FeatureCollection(
  'projects/vernal-signal-270100/assets/StudyArea/Mannual_RockMask/OST_STUDYAREA_ROCKMASK_SIMPLE'
);

// define slope masks to prevent sampling where slope > 15% 
// derived from MEASURE Greenland DEM.
var OST_SLOPEMASK_10 = ee.Image('projects/vernal-signal-270100/assets/PTM_OST_slopeMasks/OST_SlopeMask15prct');
var PTM_SLOPEMASK_10 = ee.Image('projects/vernal-signal-270100/assets/PTM_OST_slopeMasks/PTM_SlopeMask15prct');

// define per aoi cleanup polygons for manual masking from imported GEE geomtery asset
var CLEAN_POLY_1 = ee.FeatureCollection(OST_cleanup);        // applies to image0–image6 (Ostenfeld)
var CLEAN_POLY_2 = ee.FeatureCollection(PTM_cleanup);        // applies to image7–image13 (Petermann)

// cleanup polygon feature collections used for specific for image groups
var CLEAN_POLY_img11 = ee.FeatureCollection(PTM_img11_cleanup);   //ptm 11 mask for cloud over tongue on img 11 only 
var CLEAN_POLY_img2 = ee.FeatureCollection(OST_img2_cleanup);     //ost 1, 2, 3, 6 mask for crevasse field on main tongue 
var CLEAN_POLY_img8 = ee.FeatureCollection(PTM_img8_cleanup);     //ptm 8 mask tributary 
var CLEAN_POLY_img12 = ee.FeatureCollection(PTM_img12_cleanup);   //ptm 12 mask fracture shadows on tongue  

// use the union geometry of each table
var CLEAN_GEOM_1     = CLEAN_POLY_1.geometry();
var CLEAN_GEOM_2     = CLEAN_POLY_2.geometry();
var CLEAN_GEOM_img11 = CLEAN_POLY_img11.geometry();          
var CLEAN_GEOM_img2  = CLEAN_POLY_img2.geometry();
var CLEAN_GEOM_img8  = CLEAN_POLY_img8.geometry();
var CLEAN_GEOM_img12 = CLEAN_POLY_img12.geometry();

// define bands used in clustering 
var BANDS = ['B1','B2','B3','B4','B5','B6','B7','NDWI_ICE']; //NDWI_ICE band added to training images in preprocessing with 'AddNDWIiceBand'

// set RGB VIS PARAMS
var VIZ_TOA = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 1,
  gamma: [1.0]   //alt gamma: [0.95, 1.1, 1]
};

// set NDWI_ICE VIS PARAMS 
var VIZ_NDWI_ICE = {
  min: 0.115,   //alts: 0.0, 0.12
  max: 0.22,   //alts: 0.12, 0.14, 0.22
  palette: ['181a1b','697578','00ffe7']
};

// XMeans settings
var MAIN_XMEANS_MIN_CLUSTERS = 5;
var MAIN_XMEANS_MAX_CLUSTERS = 70;

var SUB_XMEANS_MIN_CLUSTERS = 20;   
var SUB_XMEANS_MAX_CLUSTERS = 50;

// define of ambig main cluster IDs to feed into the sub-clusterer 
// added after main classification is complete
var SUBSET_CLUSTER_LIST = [5, 6, 18];  

// sampling settings
var MAIN_SAMPLES_PER_IMAGE   = 100000;  // main clusterer
var SUB_SAMPLES_PER_IMAGE    = 20000;   // sub-clusterer

// index to inspect visually
var IMAGE_INDEX_TO_VIEW = 12;  // 0..N_MAIN-1

// use sub clusterer branch toggle 
var USE_SUB_CLUSTER  = true;

// ----------------------------------------------------
// AOI-SPECIFIC CLUSTER DEFINITIONS
// class main/sub cluster IDs per AOI
// JAN12
// config: syncPartial_modified_6_ 
// (see OBN notes RF_Tests, Cluster_ID_Lists)
// ----------------------------------------------------

var AOI_DEFS = {
  OST: {
    main: {
      lake:  [17, 16, 15, 14, 13, 12, 3],    
      slush: [10, 7, 4, 2],                      
      blue:  [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 9, 8, 1, 0]   
    },
    sub: {
      lake:  [],
      slush: [19, 17, 11, 8, 6, 4, 2, 1, 0],
      blue:  [18, 16, 15, 14, 13, 12, 10, 9, 7, 5, 3]
    }
  },

  PTM: {
    main: {
      lake:  [17, 16, 15, 14, 13, 12, 3],    
      slush: [26, 25, 20, 10, 7, 4, 2],           
      blue:  [30, 29, 28, 27, 21, 19]               
    },
    sub: {
      lake:  [],
      slush: [19, 17, 11, 8, 6, 4, 2, 1, 0],
      blue:  [18, 16, 14, 13, 12, 5, 3]
    }
  }
};

/**** ================================
 *  HELPERS
 *  ================================ ****/

// decide AOI key from image index
function getAoiKey(index) {
  return (index < 7) ? 'OST' : 'PTM';
}

// mask INSIDE a geometry (for manual cleanup)
function maskInside(image, geometry) {
  if (image === null) return null;

  var mask = ee.Image.constant(1).clip(geometry).mask().not();
  return ee.Image(image).updateMask(mask);
}

// mask OUT pixels where slope mask is > 15%
function maskSlopeBad(image, slopeMaskImg) {
  if (image === null) return null;
  var bad  = slopeMaskImg.unmask(0).gt(0);  
  var keep = bad.not();                    
  return ee.Image(image).updateMask(keep);
}

// turn JS array of images into an ImageCollection
var mainCollection = ee.ImageCollection(TRAINING_IMAGES);

// get the nth image from a collection, given the number of images N is known on the client side
function getImageByIndex(collection, index, nImages) {
  return ee.Image(collection.toList(nImages).get(index));
}

// build a mask from a list of main cluster IDs
function maskFromClusterList(clusterImg, idList) {
  var mask = ee.Image(0).byte();
  (idList || []).forEach(function(id) {
    mask = mask.or(clusterImg.eq(id));
  });
  return mask.eq(1);
}

// sample a set of images (NDWI-filtered) to train an XMeans clusterer
function sampleCollectionForXMeans(imageArray, bands, samplesPerImage) {
  var n = imageArray.length;
  var collection = ee.ImageCollection(imageArray);
  var allSamples = ee.FeatureCollection([]);

  for (var i = 0; i < n; i++) {
    var img = getImageByIndex(collection, i, n).select(bands);
    // Sample across image footprint (already NDWI-filtered in pipeline)
    var samples = img.sample({
      region: img.geometry(),
      scale: 30,
      numPixels: samplesPerImage
    });
    allSamples = allSamples.merge(ee.FeatureCollection([samples]));
  }

  // flatten nested feature collections
  return allSamples.flatten();
}


/**** ================================
 *  visualise AOIs as layers 
 *  ================================ ****/

Map.addLayer(PTM_AOI, {}, 'PTM AOI');
Map.addLayer(OST_AOI, {}, 'OST AOI');


/**** ================================
 *  STEP 1 – filter pixels for NDWI > 0.1
 *  ================================ ****/

// apply NDWI filter to main training images
var mainNdwi = mainCollection.map(extra_filter.extra_filter);
print('Main NDWI-filtered collection', mainNdwi);
Map.addLayer(mainCollection, VIZ_TOA, 'All training images', false);

// show one "base" image unfiltered for context
var baseImage = getImageByIndex(mainCollection, IMAGE_INDEX_TO_VIEW, N_MAIN);
Map.addLayer(baseImage, VIZ_TOA, 'Base image ' + IMAGE_INDEX_TO_VIEW, true);

// show NDWI_ice filtered version for the same index
var baseNdwi = getImageByIndex(mainNdwi, IMAGE_INDEX_TO_VIEW, N_MAIN);
Map.addLayer(baseNdwi, VIZ_TOA, 'Base NDWI-filtered ' + IMAGE_INDEX_TO_VIEW, false);

/**** ================================
 *  STEP 2 – MAIN CLUSTERER
 *  ================================ ****/

// sample image pixels where NDWI_ice > 0.1 to train main clusterer
var mainSamples = sampleCollectionForXMeans(
  TRAINING_IMAGES.map(function(img) { return extra_filter.extra_filter(img); }),
  BANDS,
  MAIN_SAMPLES_PER_IMAGE
);

print('main training sample size', mainSamples.size());

// train xMeans clusterer
var mainClusterer = ee.Clusterer.wekaXMeans({
  minClusters: MAIN_XMEANS_MIN_CLUSTERS,
  maxClusters: MAIN_XMEANS_MAX_CLUSTERS
}).train({
  features: mainSamples,
  inputProperties: BANDS
});

print('Main XMeans clusterer', mainClusterer);

/**** ================================
 *  EXPORT REFLECTANCE DATA FOR STATS
 *  PER CLUSTER, PER TRAINING IMAGE
 * (used in development for cluster spectral analysis)
 *  ================================ ****/

/**** ================================
 *  STEP 3 – SUB-CLUSTERER
 *  (for overlapping clusters/mixed clusters, off until main clusters are classed)
 *  ================================ ****/

// declare subClusterer so it's defined
var subClusterer;

if (USE_SUB_CLUSTER) {

  // build sample set from only the selected main cluster IDs, for a second XMeans
  var subSamples = ee.FeatureCollection([]);

  for (var i = 0; i < N_MAIN; i++) {
    var imgNdwi  = getImageByIndex(mainNdwi, i, N_MAIN).select(BANDS);
    var clusters = imgNdwi.cluster(mainClusterer);

    // mask to only the chosen main cluster IDs
    var subsetMask = maskFromClusterList(clusters, SUBSET_CLUSTER_LIST);
    var imgSubset  = imgNdwi.updateMask(subsetMask);

    var samples = imgSubset.sample({
      region: imgSubset.geometry(),
      scale: 30,
      numPixels: SUB_SAMPLES_PER_IMAGE
    });

    subSamples = subSamples.merge(ee.FeatureCollection([samples]));
  }

  subSamples = subSamples.flatten();
  print('Sub-clusterer training sample size', subSamples.size());

  // train second XMeans clusterer for ambig subset
  subClusterer = ee.Clusterer.wekaXMeans({
    minClusters: SUB_XMEANS_MIN_CLUSTERS,
    maxClusters: SUB_XMEANS_MAX_CLUSTERS
  }).train({
    features: subSamples,
    inputProperties: BANDS
  });

  print('Sub XMeans clusterer', subClusterer);
}


/**** ================================================
 *  SUB-CLUSTER REFLECTANCE DATA FOR STATS
 *  (similar to main cluster stats, but for subResult)
 *  ================================================ ****/
 
/**** UPDATED UI ================================
 *  STEP 4 – UI TO INSPECT MAIN + SUB CLUSTERS
 *  UI built enable inspection/analysis in GEE for classing clusters
 * (removed for computational load. see KMeans_ClusterDeff_UI)
 *  ================================ ****/
 
//===== END UI ==========


 /**** ================================
 *  STEP 5 – DEFINE CLASSES + SAMPLIING + EXPORT 
 *  ================================ ****/

// ------------------------
// PARAMETERS
// ------------------------
var SCALE = 30;
var SEED_BASE = 234;

// set target pixels N per class, per AOI
var N_PER_CLASS_PER_AOI = 1000;
var N_IMAGES_PER_AOI = 7

// oversample factor
var OVERSAMPLE_FACTOR = 5;

// preserve predictor bands used for RF
var PRED_BANDS = ['B1','B2','B3','B4','B5','B6','B7','NDWI_ICE'];

// set class codes
var CLASS_LAKE  = 0;
var CLASS_SLUSH = 1;
var CLASS_BLUE  = 2;

// export settings (drive)
var EXPORT_FOLDER = 'RF_Training';
var EXPORT_NAME   = 'syncPartial_modified_6_RF_training_stratified_' + N_PER_CLASS_PER_AOI + '_perAOIperClass';

// export settings (GEEassets)
var ASSET_ROOT = 'projects/vernal-signal-270100/assets/RF_TrainingData/';

// ------------------------
// OUTPUT: TRAINING TABLE (FeatureCollection)
// ------------------------
var training_fc = ee.FeatureCollection([]);

// keep visual collections for QA (not required for training)
var lakes_vis  = ee.ImageCollection([]);
var slushs_vis = ee.ImageCollection([]);
var blues_vis  = ee.ImageCollection([]);

 
// ------------------------
// MAIN LOOP
// ------------------------
for (var i = 0; i < N_MAIN; i++) {

  // ----- load images -----
  var imgNdwi = getImageByIndex(mainNdwi, i, N_MAIN).select(PRED_BANDS);
  var mainClust = imgNdwi.cluster(mainClusterer);

  // AOI- specific cluster definitions
  var aoiKey = getAoiKey(i);
  var defs = AOI_DEFS[aoiKey];

  // set main class masks
  var mainLakeMask  = maskFromClusterList(mainClust, defs.main.lake);
  var mainSlushMask = maskFromClusterList(mainClust, defs.main.slush);
  var mainBlueMask  = maskFromClusterList(mainClust, defs.main.blue);

  // set sub class mask (where subclustering applies)
  var subsetMask = maskFromClusterList(mainClust, SUBSET_CLUSTER_LIST);

  // set defaults for sub masks
  var subLakeMask  = ee.Image(0).byte();
  var subSlushMask = ee.Image(0).byte();
  var subBlueMask  = ee.Image(0).byte();

  if (USE_SUB_CLUSTER && subClusterer) {

    // restrict to subset pixels then subcluster
    var subsetImg = imgNdwi.updateMask(subsetMask);
    var subClust  = subsetImg.cluster(subClusterer); // band "cluster" or "sub_cluster" 

    // build sub masks from AOI sub lists
    subLakeMask  = maskFromClusterList(subClust, defs.sub.lake);
    subSlushMask = maskFromClusterList(subClust, defs.sub.slush);
    subBlueMask  = maskFromClusterList(subClust, defs.sub.blue);
  }

  //force main outside subset / sub inside subset
  subsetMask = subsetMask.unmask(0).neq(0);
  var outsideSubset = subsetMask.not();

  subLakeMask  = subLakeMask.unmask(0).neq(0).and(subsetMask);
  subSlushMask = subSlushMask.unmask(0).neq(0).and(subsetMask);
  subBlueMask  = subBlueMask.unmask(0).neq(0).and(subsetMask);

  mainLakeMask  = mainLakeMask.unmask(0).neq(0).and(outsideSubset);
  mainSlushMask = mainSlushMask.unmask(0).neq(0).and(outsideSubset);
  mainBlueMask  = mainBlueMask.unmask(0).neq(0).and(outsideSubset);

  // combined final masks
  var lakeMask  = mainLakeMask.or(subLakeMask);
  var slushMask = mainSlushMask.or(subSlushMask);
  var blueMask  = mainBlueMask.or(subBlueMask);

  //  apply masks to build class images
  var lakeImg  = imgNdwi.updateMask(lakeMask);
  var slushImg = imgNdwi.updateMask(slushMask);
  var blueImg  = imgNdwi.updateMask(blueMask);

  //  apply cleanup polys + slope masks 
  var cleanGeom = (i < 7) ? CLEAN_GEOM_1 : CLEAN_GEOM_2;
  var slopeMask = (i < 7) ? OST_SLOPEMASK_10 : PTM_SLOPEMASK_10;

  lakeImg  = maskSlopeBad(lakeImg,  slopeMask);
  slushImg = maskSlopeBad(slushImg, slopeMask);
  blueImg  = maskSlopeBad(blueImg,  slopeMask);

  lakeImg  = maskInside(lakeImg,  cleanGeom);
  slushImg = maskInside(slushImg, cleanGeom);
  blueImg  = maskInside(blueImg,  cleanGeom);

  if (i === 11) {
    lakeImg  = maskInside(lakeImg,  CLEAN_GEOM_img11);
    slushImg = maskInside(slushImg, CLEAN_GEOM_img11);
    blueImg  = maskInside(blueImg,  CLEAN_GEOM_img11);
  }
  if (i === 8) {
    lakeImg  = maskInside(lakeImg,  CLEAN_GEOM_img8);
    slushImg = maskInside(slushImg, CLEAN_GEOM_img8);
    blueImg  = maskInside(blueImg,  CLEAN_GEOM_img8);
  }
  if (i === 1 || i === 2 || i === 3 || i === 6) {
    lakeImg  = maskInside(lakeImg,  CLEAN_GEOM_img2);
    slushImg = maskInside(slushImg, CLEAN_GEOM_img2);
    blueImg  = maskInside(blueImg,  CLEAN_GEOM_img2);
  }
  if (i === 12) {
    lakeImg  = maskInside(lakeImg,  CLEAN_GEOM_img12);
    slushImg = maskInside(slushImg, CLEAN_GEOM_img12);
    blueImg  = maskInside(blueImg,  CLEAN_GEOM_img12);
  }

  // AOI name + region geom for sampling
  var AOI_NAME = (i < 7) ? 'OST' : 'PTM';
  var regionGeom = (i < 7) ? OST_AOI.geometry() : PTM_AOI.geometry();

  // build a single class band from the FINAL masked images
  // Use the masks on the final images to define membership
  var lakeM  = lakeImg.select('B1').mask().unmask(0).neq(0);
  var slushM = slushImg.select('B1').mask().unmask(0).neq(0);
  var blueM  = blueImg.select('B1').mask().unmask(0).neq(0);

  // force main,sub exclusivity
  var lakeOnly  = lakeM;
  var slushOnly = slushM.and(lakeM.not());
  var blueOnly  = blueM.and(lakeM.not()).and(slushM.not());

  var anyClass = lakeOnly.or(slushOnly).or(blueOnly);

  // create class image: 0 lake, 1 slush, 2 blue
  var classImg = ee.Image(0)
    .where(slushOnly, CLASS_SLUSH)
    .where(blueOnly,  CLASS_BLUE)
    .where(lakeOnly,  CLASS_LAKE)
    .updateMask(anyClass)
    .rename('class')
    .toByte();

  // predictor stack + class band
  var stack = imgNdwi.select(PRED_BANDS).addBands(classImg);

  // stratified sampling per image 
  var nCandPerClass = Math.ceil((N_PER_CLASS_PER_AOI * OVERSAMPLE_FACTOR) / N_IMAGES_PER_AOI);
  
  var candidates_i = stack.stratifiedSample({
    numPoints: nCandPerClass,
    classBand: 'class',
    region: regionGeom,
    scale: SCALE,
    seed: SEED_BASE + i,
    geometries: true,
    tileScale: 4
  })
  .filter(ee.Filter.inList('class', [CLASS_LAKE, CLASS_SLUSH, CLASS_BLUE]))
  .map(function(f) {
    return ee.Feature(f).set({
      AOI: AOI_NAME,
      imgIndex: i,
      aoiKey: aoiKey,
      imgId: imgNdwi.get('system:index')
    });
  });
  
  training_fc = training_fc.merge(candidates_i);
  
print('Candidate class histogram:', training_fc.aggregate_histogram('class'));
print('Candidate AOI histogram:', training_fc.aggregate_histogram('AOI'));

  // OPTIONAL visuals
  lakes_vis  = lakes_vis.merge(ee.ImageCollection([lakeImg.set({AOI: AOI_NAME, imgIndex: i})]));
  slushs_vis = slushs_vis.merge(ee.ImageCollection([slushImg.set({AOI: AOI_NAME, imgIndex: i})]));
  blues_vis  = blues_vis.merge(ee.ImageCollection([blueImg.set({AOI: AOI_NAME, imgIndex: i})]));
}

// OPTIONAL: Visual QA
//Map.addLayer(blues_vis,  VIZ_TOA, 'blue (vis)',   false);
//Map.addLayer(slushs_vis, VIZ_TOA, 'slush (vis)',  false);
//Map.addLayer(lakes_vis,  VIZ_TOA, 'lakes (vis)',  false);

// ------------------------
// CAP TO EXACT TARGET PER AOI PER CLASS
// ------------------------
function capAOIClass(fc, aoiName, classValue, n, seed) {
  return fc
    .filter(ee.Filter.eq('AOI', aoiName))
    .filter(ee.Filter.eq('class', classValue))
    .randomColumn('rand', seed)
    .sort('rand')
    .limit(n);
}

var lake_ost  = capAOIClass(training_fc, 'OST', CLASS_LAKE,  N_PER_CLASS_PER_AOI, SEED_BASE + 1000);
var slush_ost = capAOIClass(training_fc, 'OST', CLASS_SLUSH, N_PER_CLASS_PER_AOI, SEED_BASE + 1100);
var blue_ost  = capAOIClass(training_fc, 'OST', CLASS_BLUE,  N_PER_CLASS_PER_AOI, SEED_BASE + 1200);

var lake_ptm  = capAOIClass(training_fc, 'PTM', CLASS_LAKE,  N_PER_CLASS_PER_AOI, SEED_BASE + 2000);
var slush_ptm = capAOIClass(training_fc, 'PTM', CLASS_SLUSH, N_PER_CLASS_PER_AOI, SEED_BASE + 2100);
var blue_ptm  = capAOIClass(training_fc, 'PTM', CLASS_BLUE,  N_PER_CLASS_PER_AOI, SEED_BASE + 2200);

var training_final = lake_ost.merge(slush_ost).merge(blue_ost)
  .merge(lake_ptm).merge(slush_ptm).merge(blue_ptm);

// ------------------------
// DEBUGGING PRINTS
// ------------------------
print('Raw training_fc total (before capping):', training_fc.size());
print('Raw class histogram:', training_fc.aggregate_histogram('class'));
print('Raw AOI histogram:', training_fc.aggregate_histogram('AOI'));

//print('OST Lake samples', lake_ost.size());
//print('OST Slush samples', slush_ost.size());
//print('OST Blue samples', blue_ost.size());

//print('PTM Lake samples', lake_ptm.size());
//print('PTM Slush samples', slush_ptm.size());
//print('PTM Blue samples', blue_ptm.size());

print('Training total (final)', training_final.size());
print('Final class histogram:', training_final.aggregate_histogram('class'));
print('Final AOI histogram:', training_final.aggregate_histogram('AOI'));
print('Final ImgID histogram:', training_final.aggregate_histogram('imgId'));

// ------------------------
// EXPORT TABLE
// ------------------------
Export.table.toDrive({
  collection: training_final,
  description: 'DRIVE' + EXPORT_NAME,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_NAME,
  fileFormat: 'CSV'
});


Export.table.toAsset({
  collection: training_final, 
  description: 'ASSETS' + EXPORT_NAME,
  assetId: ASSET_ROOT + EXPORT_NAME
});



/*
Dell, R. L., Banwell, A. F., Willis, I. C., Arnold, N. S., Halberstadt, A. R. W., Chudley, T. R., & Pritchard, H. D. (2022). 
Supervised classification of slush and ponded water on Antarctic ice shelves using Landsat 8 imagery. 
Journal of Glaciology, 68(268), 401–414. https://doi.org/10.1017/jog.2021.114
*/
