exports.applyCloudMask_TOA = function(image, cfg) {
  image = ee.Image(image);
  var QA = image.select('QA_PIXEL');
  var proj = image.select('B2').projection();

  // ------------------------
  // 1) CLOUD + CIRRUS DETECTION
  // ------------------------
  var cloudConf  = QA.bitwiseAnd(ee.Number(3).leftShift(8)).rightShift(8);
  var cirrusConf = QA.bitwiseAnd(ee.Number(3).leftShift(14)).rightShift(14);
  var cloudBit   = QA.bitwiseAnd(1 << 3).neq(0);
  var b9Cirrus   = image.select('B9').gt(cfg.CIRRUS_B9_THRESH);

  // Combined cloud seed — anything confidently cloudy or cirrus
  var cloudSeed = cloudBit
    .or(cloudConf.gte(cfg.QA_CONF_LEVEL))
    .or(cirrusConf.gte(cfg.QA_CONF_LEVEL))
    .or(b9Cirrus)
    .unmask(0);

  // ------------------------
  // 2) REMOVE SMALL COMPONENTS FROM CLOUD SEED
  // ------------------------
  var coarseProj = proj.atScale(90);
  var seedCoarse = cloudSeed.reproject(coarseProj);
  var seedSieved = seedCoarse
    .updateMask(seedCoarse)
    .connectedPixelCount(24, true)
    .gte(ee.Number(cfg.MIN_BAD_COMPONENT_PIXELS).divide(4).ceil())
    .and(seedCoarse)
    .unmask(0)
    .reproject(proj);

// ------------------------
  // 3) PROJECT CLOUD SHADOWS
  // ------------------------
  var sunAz = ee.Number(360).add(ee.Number(image.get('SUN_AZIMUTH')));
  var shadowAzDegrees = sunAz.add(180).mod(360);
  var shadowAzImgAxis = shadowAzDegrees.add(90).mod(360);
  var pixelSize = ee.Number(proj.nominalScale());
  var maxDistPx = ee.Number(cfg.SHADOW_PROJECT_DISTANCE_M).divide(pixelSize).ceil();

  var shadowProj = seedSieved
    .directionalDistanceTransform(shadowAzImgAxis, maxDistPx)
    .select('distance')
    .lte(maxDistPx)
    .unmask(0);
  
  // ------------------------
  // 4) COMBINE: CLOUD + SHADOW + QA SHADOW BIT
  // ------------------------
  var shadowBit = QA.bitwiseAnd(1 << 4).neq(0);
  var shadowConf = QA.bitwiseAnd(ee.Number(3).leftShift(10)).rightShift(10);

  var bad = seedSieved
    .or(shadowProj)
    .or(shadowBit)
    .or(shadowConf.gte(cfg.QA_CONF_LEVEL))
    .unmask(0);

  // ------------------------
  // 5) BUFFER AROUND ALL BAD PIXELS
  // ------------------------
  var badCoarse = bad.reproject(proj.atScale(cfg.DIST_SCALE_M));
  var bufferPx  = ee.Number(cfg.BUFFER_M).divide(cfg.DIST_SCALE_M);
  var maxPx     = bufferPx.ceil().add(2);
  // fastDistanceTransform returns squared pixel distance; compare in squared
  // space to avoid a per-pixel sqrt and multiply.
  var buffered  = bad
    .or(badCoarse.fastDistanceTransform(maxPx).lte(bufferPx.pow(2)).reproject(proj))
    .unmask(0);

  return image.updateMask(buffered.not());
};
