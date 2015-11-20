# Simulating climate change on run-of-river hydropower

This repo contains basic Python code for estimating the potential impacts of climate change on Scotland's run-of-river hydropower potential. The code makes use of the following publically available datasets:

1. **[Future Flows catchment level summaries](https://catalogue.ceh.ac.uk/documents/f3723162-4fed-4d9d-92c6-dd17412fa37b)** <br><br>
The Future Flows (FF) project used gridded output from the Met Office's HadRM3 Regional Climate Model (RCM) to estimate changing river flows for around 280 catchments across the UK. 11 different simulations from the RCM (all based on A1B emissions) were first bias-corrected and downscaled to 1 km resolution. These simulations were then used to drive up to three hydrological models in each catchment, resulting in between 11 and 33 flow time series per site. The simulations run from 1951 to 2098 and the range of output gives a broad indication of the range of uncertainty in flows under the assumptions of the medium (A1B) emissions scenario and the HadRM3 climate model. <br><br>

2. **[Historic flow data from the National River Flow Archive (NRFA)](http://nrfa.ceh.ac.uk/data/search)**<br><br>
Observed data for some (148) of the FF catchments can be downloaded from the NFRA. A summary of the FF sites and whether observed data is easily available is given in [NRFA_FF_Stations.csv](https://github.com/JamesSample/simple_hydropower_model/blob/master/NRFA_FF_Stations.csv). <br><br>

The observed datasets provide an important check on the quality of the FF output. For many catchments, the FF simulations for historic periods are *not* a close match for the observed data, suggesting they are not suitable for investigating future flows. In catchments where FF model perfromance is better, we may have some justification for using the data to simulate future hydropower, although this still involves making some fairly sweeping assumptions.

## 1. Estimating hydropower potential

Many guidance documents recommend that *preliminary* hydropower assessments are based on **Flow Duration Curves (FDCs)** derived from daily flow data. Given a flow time series with daily resolution, the code here estimates hydropower potential in the following way:

1. Calculate the FDC using standard methods. <br><br>

2. Assume the site has a fixed head, H, and that turbine size is determined by selecting a particular exceedance percentage, ![Popt](http://www.sciweavers.org/tex2img.php?eq=P_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) (which broadly represents the proportion of the time that a turbine is expected to run at full capacity). Smaller values of ![Popt](http://www.sciweavers.org/tex2img.php?eq=P_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) represent higher design flows and bigger turbines, running at full capacity for a smaller proportion of the time. 

  The choice of ![Popt](http://www.sciweavers.org/tex2img.php?eq=P_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is not obvious and has a large effect on the scheme power output. It is therefore necessary to consider a range of values for ![Popt](http://www.sciweavers.org/tex2img.php?eq=P_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0), which involves repeating the calculations below many times.

3. 	Using ![Popt](http://www.sciweavers.org/tex2img.php?eq=P_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) and the FDC, read off the associated flow rate (![Qopt](http://www.sciweavers.org/tex2img.php?eq=Q_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)) and estimate the turbine capacity using the equation:

  ![eq1](http://www.sciweavers.org/tex2img.php?eq=C%20%3D%20%5Crho%20g%20Q%20H%20%5Cfrac%7BE%7D%7B100%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

  where	C is the turbine capacity (W), ρ is the density of water (kg/m3), Q is the flow rate (m3/s), H is the head (m) and E is the overall plant efficiency factor (%).
  
4. At flows greater than or equal to ![Qopt](http://www.sciweavers.org/tex2img.php?eq=Q_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0), the turbine will function at its rated peak capacity, C. At flows below this, the turbine will function at reduced output down to some lower threshold, ![Qmin](http://www.sciweavers.org/tex2img.php?eq=Q_%7Bmin%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0), where power generation stops. ![Qmin](http://www.sciweavers.org/tex2img.php?eq=Q_%7Bmin%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0), is usually approximated as a percentage (e.g. 30%) of ![Qopt](http://www.sciweavers.org/tex2img.php?eq=Q_%7Bopt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0).


