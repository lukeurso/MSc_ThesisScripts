//luke urso, Stockhlm U. 2025
//filter for NDWI_ICE greater than 0.01 
exports.extra_filter = function(image){ 

var ndwi = image.normalizedDifference(['B2', 'B4']).rename('NDWI');
var threshold_1 = ndwi.gt(0.1);
var threshold_1_mask = threshold_1.eq(1);
var finalimage = image.updateMask(threshold_1);
return finalimage;
}; 
