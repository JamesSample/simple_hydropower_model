# Simulating climate change on run-of-river hydropower

This repo contains basic Python code for estimating the potential impacts of climate change on Scotland's run-of-river hydropower potential. The code makes use of the following publically available datasets:

1. **[Future Flows catchment level summaries](https://catalogue.ceh.ac.uk/documents/f3723162-4fed-4d9d-92c6-dd17412fa37b)** <br><br>
The Future Flows (FF) project used gridded output from the Met Office's HadRM3 Regional Climate Model (RCM) to estimate changing river flows for around 280 catchments across the UK. 11 different simulations from the RCM (all based on A1B emissions) were first bias-corrected and downscaled to 1 km resolution. These simulations were then used to drive up to three hydrological models in each catchment, resulting in between 11 and 33 flow time series per site. The simulations run from 1951 to 2098 and the range of output gives a broad indication of the range of uncertainty in flows under the assumptions of the medium (A1B) emissions scenario and the HadRM3 climate model. <br><br>

2. **[Historic flow data from the National River Flow Archive (NRFA)](http://nrfa.ceh.ac.uk/data/search)**<br><br>
Observed data for some of the FF catchments can be downloaded from the NFRA. A summary of the FF sites and whether observed data is easily available is given in 


