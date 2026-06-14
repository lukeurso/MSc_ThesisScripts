// validation_classifier.js
// (c) Luke Urso, Stockholm Univeristy (2025)
//
// Classifies 4 preprocessed validation scene assets using the Random Forest
// trained in export_v6.js and exports one sampled-point FeatureCollection per
// scene for use in expert_validation.js.
//
// Prerequisites:
//   - Validation scene assets exist at SCENE_ASSET_BASE + scene ID.
//   - Each asset contains bands B1–B7 and NDWI_ICE (pre-computed).
//   - Training data asset exists at TRAIN_FC_ASSET.
//   - Output folder SAMPLES_ASSET_BASE exists in your GEE asset manager.

// ========================
// USER SETTINGS
// ========================

// Training data (from export_v6.js)
var TRAIN_FC_ASSET  = 'projects/vernal-signal-270100/assets/RF_TrainingData/syncPartial_modified_6_RF_training_stratified_1000_perAOIperClass';
var CLASS_PROPERTY  = 'class';
var PRED_BANDS      = ['B1','B2','B3','B4','B5','B6','B7','NDWI_ICE'];

// RF hyperparameters (from export_v6.js)
var N_TREES         = 300;
var VARS_PER_SPLIT  = 5;
var MIN_LEAF        = 10;
var BAG_FRACTION    = 0.632;
var MAX_NODES       = null;
var SEED            = 123;

// NDWI threshold — pixels below this are masked before classification
var NDWI_ICE_MIN    = 0.1;

// Asset base paths
var SCENE_ASSET_BASE   = 'projects/vernal-signal-270100/assets/SceneAssets/ValidationScenes/';
var SAMPLES_ASSET_BASE = 'projects/vernal-signal-270100/assets/SceneAssets/ValidationSamples/';

// The 4 validation scene IDs (the final segment of the asset path).
// Replace placeholders with actual asset IDs before running.
var SCENE_IDS = [
  'ValidationScene_0',
  'ValidationScene_1',
  'ValidationScene_2',
  'ValidationScene_3'
];

// numPixels passed to .sample() per scene.
// Tune each value until ~350 unmasked points are returned — masked pixels are
// dropped automatically.  Scenes with sparse meltwater may need very large
// values (see guidance in dell_Classifier_for_Validation_Scenes.js).
var NUM_PIXELS_PER_SCENE = {
  'ValidationScene_0': 50000,
  'ValidationScene_1': 50000,
  'ValidationScene_2': 50000,
  'ValidationScene_3': 50000
};

var SAMPLE_SCALE = 30; // metres — native Landsat pixel size

// ========================
// VIZ PARAMS
// ========================

var vizParamsRGB = {
  bands:  ['B4', 'B3', 'B2'],
  min:    0,
  max:    1,
  gamma:  [0.95, 1.1, 1]
};

// Classification palette — colours map to RF class indices (0-based).
// Update to match your training-data class scheme if needed.
var CLASS_PALETTE = ['5e65c1', '49a1ee', 'a4eaee'];

// ========================
// TRAIN CLASSIFIER
// ========================

var trainingFC      = ee.FeatureCollection(TRAIN_FC_ASSET);
var cleanedTraining = trainingFC.filter(ee.Filter.notNull(PRED_BANDS.concat([CLASS_PROPERTY])));

// Use the same 80 / 20 random split as export_v6.js
var withRand  = cleanedTraining.randomColumn('rand', SEED);
var trainSet  = withRand.filter(ee.Filter.lt('rand', 0.8));

var rf = ee.Classifier.smileRandomForest({
  numberOfTrees:     N_TREES,
  variablesPerSplit: VARS_PER_SPLIT,
  minLeafPopulation: MIN_LEAF,
  bagFraction:       BAG_FRACTION,
  maxNodes:          MAX_NODES,
  seed:              SEED
}).train({
  features:        trainSet,
  classProperty:   CLASS_PROPERTY,
  inputProperties: PRED_BANDS
});

print('RF explain():', rf.explain());

// ========================
// PROCESS EACH SCENE
// ========================

for (var idx = 0; idx < SCENE_IDS.length; idx++) {
  var sceneId = SCENE_IDS[idx];
  var image   = ee.Image(SCENE_ASSET_BASE + sceneId);

  // Display base image
  Map.addLayer(image, vizParamsRGB, sceneId + ' RGB');

  // Mask pixels below NDWI threshold then classify
  var predictors = image.select(PRED_BANDS)
    .updateMask(image.select('NDWI_ICE').gte(NDWI_ICE_MIN));

  var classified = predictors.classify(rf).rename('classification');

  Map.addLayer(
    classified,
    {min: 0, max: 2, palette: CLASS_PALETTE},
    sceneId + ' classified'
  );

  // Build the image that will be sampled:
  //   classification | original spectral bands | interp1 placeholder | confidence placeholder
  // Reproject everything to the native B1 grid so all bands share a
  // consistent projection (mirrors dell_Classifier_for_Validation_Scenes.js).
  var proj = image.select('B1').projection();

  var outputImage = classified
    .addBands(image.select(PRED_BANDS))
    .addBands(ee.Image([0]).rename('interp1'))    // filled in by expert during validation
    .addBands(ee.Image([0]).rename('confidence')) // filled in by expert during validation
    .reproject(proj);

  print(sceneId + ' output band names:', outputImage.bandNames());

  // Sample ~350 unmasked points (masked pixels are dropped by default)
  var numPx  = NUM_PIXELS_PER_SCENE[sceneId];
  var points = outputImage.sample({
    numPixels:  numPx,
    region:     image.geometry(),
    geometries: true,
    scale:      SAMPLE_SCALE
  });

  print(sceneId + ' sampled point count:', points.size());
  Map.addLayer(points, {color: 'FF0000'}, sceneId + ' sample points');

  // Export sampled points to asset — one asset per scene
  Export.table.toAsset({
    collection:  points,
    description: 'ValidationSamples_' + sceneId,
    assetId:     SAMPLES_ASSET_BASE + sceneId + '_samples'
  });
}
