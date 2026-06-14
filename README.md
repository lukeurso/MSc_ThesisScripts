## Dear reader! 

This repository hosts code used in my Master's thesis, "Twelve years of slush and supraglacial lakes on Petermann and C.H. Ostenfeld glaciers from supervised classification", at Stockholm University, Department of Physical Geography. Meltwater was classified from Landsat 08 and 09 in Google Earth Engine JavaScript API using a Random Forest classifier based on Dell et al. (2022). Volume retrieval, also performed in Earth Engine, followed the Pope et al. (2016) and Moussavi et al. (2020) methods. I incorporated new slope shadow and cloud shadow masking algorithms, which allow for 2-day temporal resolution and tracking of rapid drainage events. The dataset (hosted on the Bolin Center database) adds new, high-frequency, long-term area and elevation data for both slush and lakes, coupled with lake volume and rapid drainage events. These results represent a new combined slush-lake-volume-rapid-drainage dataset for Petermann and the first detailed surface melt record for Ostenfeld.  

**_PS: If you find yourself here and happen to have an Earth Engine account, try the [[public_drainage_event_viewer_for_EE.js]]. It uses EE assets from this study I've made public to interactively show the before and after of drainage events with fast composite RGB Landsat scenes. I think it is very fun :)_** 

  
My name is Luke Urso, my email is lukeurso@me.com, and my website is https://lukeurso.github.io/. 

---


Dell, R. L., Banwell, A. F., Willis, I. C., Arnold, N. S., Halberstadt, A. R. W., Chudley, T. R., & Pritchard, H. D. (2022). Supervised classification of slush and ponded water on Antarctic ice shelves using Landsat 8 imagery. Journal of Glaciology, 68(268), 401–414. https://doi.org/10.1017/jog.2021.114
Moussavi, M., Pope, A., Halberstadt, A. R. W., Trusel, L. D., Cioffi, L., & Abdalati, W. (2020). Antarctic Supraglacial Lake Detection Using Landsat 8 and Sentinel-2 Imagery: Towards Continental Generation of Lake Volumes. Remote Sensing, 12(1), Article 1. https://doi.org/10.3390/rs12010134
Pope, A., Scambos, T. A., Moussavi, M., Tedesco, M., Willis, M., Shean, D., & Grigsby, S. (2016). Estimating supraglacial lake depth in West Greenland using Landsat 8 and comparison with other multispectral methods. The Cryosphere, 10(1), 15–27. https://doi.org/10.5194/tc-10-15-2016

