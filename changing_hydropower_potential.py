#------------------------------------------------------------------------------
# Name:        changing_hydropower_potential_github.py
# Purpose:     Estimate climate change impacts on hydropower potential using
#              Future Flows catchment level summaries.
#
# Author:      James Sample
#
# Created:     
# Copyright:   (c) James Sample and JHI, 2014
#------------------------------------------------------------------------------
""" This code uses the Future Flows (FF) data to esimate run-of-river 
    hydropower potential under different scenarios of climate change.
    
    Further details available here:
    https://github.com/JamesSample/simple_hydropower_model
    
    Future Flows data available here:    
    https://catalogue.ceh.ac.uk/documents/f3723162-4fed-4d9d-92c6-dd17412fa37b
    
    Observed datasets for some sites (for evaluation of the FF simulations) 
    available here:
    http://nrfa.ceh.ac.uk/data/search
"""

import pandas as pd, matplotlib.pyplot as plt, numpy as np, os, glob
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator 
from scipy.interpolate import interp1d
from scipy.integrate import trapz

# Get matplotlib to render TrueType font when saving to PDF
mpl.rcParams['pdf.fonttype'] = 42

# Change to 'Agg' backend to avoid memory leak
plt.switch_backend('Agg')

def validate_inputs(plant_eff_fac, turb_min_pct, hof_pct, pct_over_hof):
    """ Validate user input.
    
    Args:
        plant_eff_fac: Overall % efficiency of the plant
        turb_min_pct:  Percentage of the optimum turbine capacity below which 
                       generation stops
        hof_pct:       Hands off flow percentile
        pct_over_hof:  Pct of flow above the HOF available to plant
    
    Raises:
        AssertionErrors if Args are not valid percentages.
    """
    for pct in [plant_eff_fac, turb_min_pct, hof_pct, pct_over_hof]:
        assert (0 <= pct <= 100)

def read_station_data(ff_stns_csv, base_st, base_end, scale_area):
    """ Identify stations to process based on baseline period of interest 
        and scheme catchment area.
    
    Args:
        ff_stns_csv: Summary data for FF stations (available from GitHub repo)
        base_st:     Start date for baseline (yyyy-mm-dd)
        base_end:    End date for baseline (yyyy-mm-dd)
        scale_area:  Bool. Scale flows to catch_area
    
    Returns:
        Data frame of stations to process.
    """ 
    # Read the station data
    stn_df = pd.read_csv(ff_stns_csv, index_col=0)
    
    # Just consider sites with obs data available within the period of interest
    st_yr = int(base_st[:4])
    end_yr = int(base_end[:4])
    if scale_area == True:
        stn_df = stn_df[(stn_df['Obs_Avail']=='Y')&(stn_df['Start']<st_yr)
                        &(stn_df['End']>end_yr)&(stn_df['Area_km2']>catch_area)]
    else:
        stn_df = stn_df[(stn_df['Obs_Avail']=='Y')&(stn_df['Start']<st_yr)
                        &(stn_df['End']>end_yr)]
    
    return stn_df

def read_obs(obs_csv, scale_area, area_fac):
    """ Read observed data and resample to desired frequency. 
    
    Args:
        obs_csv:     A CSV of observed data from the NRFA website
        scale_area:  Bool. Scale observed flows to scheme area
        area_fac:    Factor to use for flow scaling
        
    Returns:
        Data frame.
    """
    # Reading in the data is a bit of a faff due to the way it's laid out in 
    # the NRFA CSVs. The code below seems to handle the differences robustly
    df = pd.read_csv(obs_csv, skiprows=15, error_bad_lines=False)
    df.columns=['Dates', 'Flow', 'Flags']
    df['Dates'] = pd.to_datetime(df['Dates'], dayfirst=True)
    df.index = df['Dates']
    df = df[['Flow']]    
    df = df.dropna()
    
    # Extract baseline period
    df = df[base_st:base_end] 
    
    # Resample
    df = df.resample(freq, how='mean')
    
    df = df.dropna()
    
    # Scale by area if necessary
    if scale_area == True:
        df = df*area_fac

    return df

def read_ff(ff_csvs, scale_area, area_fac):
    """ Read FF data and resample to desired frequency.
        
        Takes a list of CSVs, where each CSV contains daily flows from the 11
        climate simulations, as predicted by one of the FF water balance models
        (CLASSIC, CERF or PDM). Where more than one FF model is available, the 
        output is merged to represent the full range of flow scenarios (up to 
        33 in total).    
        
    Args:
        ff_csvs:     List of FF CSVs for desired site (from CEH website)
        scale_area:  Bool. Scale observed flows to scheme area
        area_fac:    Factor to use for flow scaling
        
    Returns:
        Data frame.
    """
    # Merge data if necessary
    if len(ff_csvs) == 1:
        ff_csv = ff_csvs[0]
        df = pd.read_csv(ff_csv, index_col=2, parse_dates=True, dayfirst=True)
        del df['catchmentID'], df['HydModID (flow cumecs)']
        df.columns = range(1,12)
    else:
        df_list = []
        for idx, file_path in enumerate(ff_csvs):
            # Read to df
            ff_df = pd.read_csv(file_path, index_col=2, parse_dates=True, 
                                dayfirst=True)
            del ff_df['catchmentID'], ff_df['HydModID (flow cumecs)']
            
            # Rename the columns with unique integers to avoid naming conflicts
            ff_df.columns = range(idx*11+1, idx*11+12)
            df_list.append(ff_df)
        df = pd.concat(df_list, axis=1)
    
    # Extract baseline and future portions
    base_df = df[base_st:base_end] 
    fut_df = df[fut_st:fut_end]
    
    # Resample to desired frequency
    base_df = base_df.resample(freq, how='mean')
    fut_df = fut_df.resample(freq, how='mean')
       
    # Area-scale flow if necessary
    if scale_area == True:
        base_df = base_df*area_fac
        fut_df = fut_df*area_fac

    return base_df, fut_df  

def select_season(df, season):
    """ Takes a data frame with a date time index and selects just the rows
        corresponding to the selected season.
    
    Args:
        df:     Data frame of flow data
        season: 'Spring' - M, A, M
                'Summer' - J, J, A
                'Autumn' - S, O, N
                'Winter' - D, J, F
    
    Returns:
        Data frame.
    """
    seasons_dict = {'spring':[3, 4, 5],
                    'summer':[6, 7, 8],
                    'autumn':[9, 10, 11],
                    'winter':[1, 2, 12]}
    
    return df[df.index.map(lambda x: x.month in seasons_dict[season.lower()])]

def fdc_from_obs(df):
    """ Calculates a Flow Duration Curve (FDC) from observed data.
    
    Args:
        df: Data frame of observed data. Must have two columns: Dates and Flow.
    
    Returns:
        Arrays of flows and associated exceedence probabilities.
    """   
    # Delete date info as we don't need it anymore
    df = df.reset_index(drop=True)
    
    # Sort
    df = df[['Flow']].sort(columns='Flow',
                           ascending=False).reset_index(drop=True)
    
    # Get ranks, where rank 1 = largest
    ranks = np.arange(1, len(df)+1)
    
    # Get exceedence prob
    ex_prob = 100.*ranks/(len(ranks)+1)
    
    return ex_prob, df

def fdc_from_ff(base_df, fut_df):
    """ Calculates baseline and future FDCs from the FF data.

    Args:
        base_df: Data frame of FF simulations (<33) for the baseline period.
        fut_df: Data frame of FF simulations (<33) for the future period.
    
    Returns:
        Arrays of flows and associated exceedence probabilities for both 
        periods.
    """   
    # Delete date info as we don't need it anymore
    base_df = base_df.reset_index(drop=True)
    fut_df = fut_df.reset_index(drop=True)
    
    # Individually sort columns
    for col in base_df.columns:
        base_df[col] = base_df[[col,]].sort(columns=col,
                                            ascending=False).reset_index(drop=True)
    
    for col in fut_df.columns:
        fut_df[col] = fut_df[[col,]].sort(columns=col,
                                          ascending=False).reset_index(drop=True)
        
    # Get ranks, where rank 1 = largest
    base_ranks = np.arange(1, len(base_df)+1)
    fut_ranks = np.arange(1, len(fut_df)+1)
    
    # Get exceedence prob
    base_ex_prob = 100.*base_ranks/(len(base_ranks)+1)
    fut_ex_prob = 100.*fut_ranks/(len(fut_ranks)+1)
      
    return base_ex_prob, base_df, fut_ex_prob, fut_df 

def estimate_hof(flows, probs):
    """ Estimate the Hands Off Flow (HOF) based on output from fdc_from_obs or
        fdc_from_ff.
    
    Args:
        flows: Array of (ranked) flows 
        probs: Associated array of exceedence probabilities.
    Returns:
        HOF (float).
    """
    # Build interpolator
    interp_p2q = interp1d(probs, flows)
    
    # Get the hands-off flow
    return float(interp_p2q(hof_pct)) 
    
def calc_plant_obs_fdc(obs_df, orig_flows, orig_ps):
    """ Calculates the amount of water available to the plant based on the 
        observed data, accounting for environmental regulations (HOF and 
        pct_over_hof).
    
    Args:
        obs_df:     Data frame of observed data
        orig_flows: Ranked flows from observed data
        orig_ps:    Exceedence probabilities from observed data
    
    Returns:
        Arrays of flows and exceedence probabilities actually available to 
        scheme.        
    """ 
    # Get the obs hof
    obs_hof = estimate_hof(orig_flows, orig_ps)
    
    # Get the water available to the scheme
    obs_df = (obs_df - obs_hof)*pct_over_hof/100.
    obs_df[obs_df<0] = 0
    
    # Recalculate observed FDC     
    obs_p, obs_q = fdc_from_obs(obs_df)
    
    return obs_p, obs_q

def calc_plant_ff_fdc(base_df, base_ff_q_orig, base_ff_p_orig,
                      fut_df,  fut_ff_q_orig, fut_ff_p_orig):
    """ Calculates the amount of water available to the plant based on the 
        FF data, accounting for environmental regulations (HOF and 
        pct_over_hof).
    
    Args:
        base_df:         Data frame of baseline FF flow data
        base_ff_q_orig:  Ranked baseline FF flows
        base_ff_p_orig:  Baseline FF exceedence probabilities
        fut_df:          Data frame of future FF flow data
        fut_ff_q_orig:   Ranked future FF flows
        fut_ff_p_orig:   Future FF exceedence probabilities
    
    Returns:
        Arrays of flows and exceedence probabilities actually available to 
        scheme.  
    """
    # Get the water available to the scheme for each scenario
    for col in base_ff_q_orig.columns:
        # Baseline
        base_ff_hof = estimate_hof(base_ff_q_orig[col], base_ff_p_orig)
        base_df[col] = (base_df[col] - base_ff_hof)*pct_over_hof/100.
        base_df[col][base_df[col]<0] = 0
        
        # Future
        fut_ff_hof = estimate_hof(fut_ff_q_orig[col], fut_ff_p_orig)
        fut_df[col] = (fut_df[col] - fut_ff_hof)*pct_over_hof/100.
        fut_df[col][fut_df[col]<0] = 0

    # Recalculate the future FDCs
    base_ff_p, base_ff_q, fut_ff_p, fut_ff_q = fdc_from_ff(base_df, fut_df)   

    return base_ff_p, base_ff_q, fut_ff_p, fut_ff_q                                              
                                                 
def select_turbine(flows, probs, turb_cap_pct):
    """ Takes arrays of flows and exceedence probabilities and estimates 
        the turbine size for the given exceedence percentage. 
    
    Args:
        flows:        Ranked flows
        probs:        Exceedence probabilities 
        turb_cap_pct: Design flow percentile for turbine
        
    Returns:        
        Floats (optimum_flow, turbine capacity)
    """
    # Build interpolator
    interp_p2q = interp1d(probs, flows)
    
    # Estimate turbine capacity
    opt_flow = float(interp_p2q(turb_cap_pct))
    turb_cap = 9.81*opt_flow*head*plant_eff_fac/100. # in kW
    
    return opt_flow, turb_cap

def estimate_energy_output(opt_flow, turb_cap, turb_cap_pct, turb_min_pct,
                           flows, probs, season):
    """ Estimate energy output and load factor for a particular turbine given
        the FDC data. 
        
    Args:
        opt_flow:      Optimum flow for turbine
        turb_cap:      Turbine capacity
        turb_cap_pct:  Exceedence percentile for turbine capacity
        turb_min_pct:  Exceedence percentile for turbine cut-out
        flows:         Ranked flows available to scheme
        probs:         Exceedence probabilities
        season:        Season of interest
        
    Returns:
        Energy output (MWh)
        Load factor (%).
    """  
    # Build interpolator to estimate exceedence from Q
    interp_q2p = interp1d(np.array(flows)[::-1],
                          np.array(probs)[::-1],
                          bounds_error=False,
                          fill_value=100)
       
    # Estimate min flow threshold for generation and associated exceedence %
    min_flow = opt_flow*turb_min_pct/100.
    min_p = float(interp_q2p(min_flow))
      
    # Calculate areas
    rec_area = opt_flow*turb_cap_pct
    
    # Get arrays for the part touching the curve
    flow_vals = np.array(flows[np.logical_and(probs>turb_cap_pct, 
                                              probs<min_p)])
    p_vals = np.array(probs[np.logical_and(probs>turb_cap_pct,
                                           probs<min_p)])
                                    
    # Add exact start and end points
    flow_vals = np.insert(flow_vals, 0, opt_flow)
    p_vals = np.insert(p_vals, 0, turb_cap_pct)
    
    flow_vals = np.append(flow_vals, min_flow)
    p_vals = np.append(p_vals, min_p)
    
    # Area of second part of curve                                
    curve_area = trapz(flow_vals, p_vals)
    
    # Area of HOF
    tot_area = rec_area + curve_area
    
    # Calculate average effective flow
    eff_flow = tot_area/100.
    eff_cap = 9.81*eff_flow*head*plant_eff_fac/100. # in kW
    
    # Estimate energy output in this season
    days_dict = {'annual':365,
                 'spring':92,
                 'summer':92, 
                 'autumn':91,
                 'winter':90}
    
    # Get days for this season             
    days = days_dict[season.lower()]             
    
    # Energy generated this season
    eff_energy = days*24*eff_cap/1000. # in MWh
    
    # Estimate load factor
    load_fac = 100.*eff_cap/turb_cap
    
    return eff_energy, load_fac

def process_ff_data(flows_df, probs, pct, turb_min_pct, season):
    """ Loops over each simulation in a FF data frame, estimating energies and
        load factors for the specified exceedance percentage.
        
    Args:
        flows_df:      Data frame of FF data
        probs:         Exceedence percentile
        pct:           Exceedence percentile for turbine capacity
        turb_min_pct:  Exceedence percentile for turbine cut-out
        season:        Season of interest
        
    Returns:
        List of lists [[Capacities, Energy Output (MWh), Load Factors (%)]]
    """ 
    cap_list = []
    en_list = []
    lf_list = []
    for col in flows_df.columns:
        opt_q, turb_cap = select_turbine(flows_df[col], 
                                         probs,
                                         pct)
                                         
        energy, lf = estimate_energy_output(opt_q, 
                                            turb_cap,
                                            pct,
                                            turb_min_pct,
                                            flows_df[col], 
                                            probs,
                                            season)
        cap_list.append(turb_cap)
        en_list.append(energy)
        lf_list.append(lf)
    
    return [cap_list, en_list, lf_list]

def interp_en_lf(cap_df, en_df, lf_df):
    """ Takes data frames containing capacities, energies and load factors for
        percentiles running from 5 to 95. Identifies a common capacity scale
        and interpolates values for energy and load factors for each point
        along this scale. Returns data frames of energies and load factors
        where the index is the identified capacity scale.
    
    Args:
        cap_df: Data frame of capacities
        en_df:  Data frame of energy outputs
        lf_df:  Data frame of load factors
    
    Returns:
        Data frames showing energy and load factor as a function of capacity.
    """
    # Identify min and max capacities for scale
    # Rounded up/down as appropriate
    c_max = int(cap_df.min(axis=1).ix[5] - 1)
    c_min = int(cap_df.max(axis=1).ix[95] + 1)
    # If (c_max - c_min) is small, there's not much point in considering power 
    # potential. Only consider sites where range of possible capacities exceeds
    # 5 kW
    if (c_max - c_min)<5:
        print "    Range of capacities is < 5 kW. Too small for meaningful comparison."
        return (None, None)
    else:
        cap_range = np.arange(c_min, c_max, 1)
    
        # Interpolate
        en_dict = {}
        lf_dict = {}
        for col in cap_df.columns:
            caps = np.array(cap_df[col])[::-1]
            ens = np.array(en_df[col])[::-1]
            lfs = np.array(lf_df[col])[::-1]
            
            # Build interpolators
            # Energy
            interp_c2e = interp1d(caps, ens)
            en_dict[col] = interp_c2e(cap_range)
            
            # Load factors
            interp_c2l = interp1d(caps, lfs)
            lf_dict[col] = interp_c2l(cap_range)
        cap_en_df = pd.DataFrame(en_dict, index=cap_range)
        cap_lf_df = pd.DataFrame(lf_dict, index=cap_range)
        
        return (cap_en_df, cap_lf_df)

def print_outputs_for_turbine_of_interest(base_cap_interest,
                                          obs_cap_df,
                                          obs_en_df,
                                          obs_lf_df,
                                          base_en_df,
                                          base_lf_df,
                                          fut_en_df,
                                          fut_lf_df):
    """ Print outputs for a turbine of the specified capacity.
    
    Args:
        base_cap_interest: Specified capacity of interest (kW)
        obs_cap_df:        Capacity data frame from observed data
        obs_en_df:         Energy data frame from observed data
        obs_lf_df:         Load factor data frame from observed data
        base_en_df:        Energy data frame from FF baseline data      
        base_lf_df:        Load factor data frame from FF baseline data
        fut_en_df:         Energy data frame from FF future data
        fut_lf_df:         Load factor data frame from FF future data
        
    Returns:
        Prints energy output and load factor for observed and FF baselines and
        FF future periods.
        Prints -1 if specified capacity is outside the range of flow 
        percentiles considered by the script (5 to 95).
    """             
    # Observed baseline
    interp_obs_en = interp1d(obs_cap_df['Cap'][::-1], 
                             obs_en_df['En'][::-1],
                             bounds_error=False,
                             fill_value=-1)
    interp_obs_lf = interp1d(obs_cap_df['Cap'][::-1], 
                             obs_lf_df['LF'][::-1],
                             bounds_error=False,
                             fill_value=-1)
    obs_en = float(interp_obs_en(base_cap_interest))
    obs_lf = float(interp_obs_lf(base_cap_interest))

    # FF baseline
    interp_base_en = interp1d(base_en_df.index, 
                              base_en_df['50%'],
                              bounds_error=False,
                              fill_value=-1)
    interp_base_lf = interp1d(base_lf_df.index, 
                              base_lf_df['50%'],
                              bounds_error=False,
                              fill_value=-1)
    base_en = float(interp_base_en(base_cap_interest))
    base_lf = float(interp_base_lf(base_cap_interest))

    # FF future
    interp_fut_en = interp1d(fut_en_df.index, 
                             fut_en_df['50%'],
                             bounds_error=False,
                             fill_value=-1)
    interp_fut_lf = interp1d(fut_lf_df.index, 
                             fut_lf_df['50%'],
                             bounds_error=False,
                             fill_value=-1)
    fut_en = float(interp_fut_en(base_cap_interest))
    fut_lf = float(interp_fut_lf(base_cap_interest))
    
    print ('      Observed baseline: energy output %.2f MWh; '
           'load factor %.0f%%' % (obs_en, obs_lf))
    print ('      FF baseline:       energy output %.2f MWh; '
           'load factor %.0f%%' % (base_en, base_lf))
    print ('      FF future:         energy output %.2f MWh; '
           'load factor %.0f%%' % (fut_en, fut_lf))
    
                              
# #############################################################################
# User input
obs_fold = r'D:\Flow_Duration_Curves\FF_Catchment_Level_Data\NRFA_FF_Obs_Data'
ff_fold = r'D:\Flow_Duration_Curves\FF_Catchment_Level_Data\NRFA_FF_TS_Data'
ff_stns_csv = r'D:\Flow_Duration_Curves\FF_Catchment_Level_Data\NRFA_FF_Stations.csv'

out_fold = r'D:\Flow_Duration_Curves\Plots\Working2'

freq = 'D'  # Calc FDC with this frequency

# Define baseline and future time periods (in time frequency of raw data)
# If you choose to include raw data, the same baseline time period will be used
# for that as well
base_st = '1961-01-01'
base_end = '1990-12-31'

fut_st = '2041-01-01'
fut_end = '2070-12-31'

# Hydropower parameters
head = 25           # Plant head in m
catch_area = 10     # Plant catchment area in km2
plant_eff_fac = 70  # Overall % efficiency of the plant (turbine and generator). 
                    # HEC says ~85%; british-hydro.org says ~70%

turb_min_pct = 10   # Percentage of the optimum turbine capacity below which 
                    # generation stops

hof_pct = 95        # Hands off flow percentile

pct_over_hof = 50   # Pct of flow above the HOF available to plant

scale_area = False  # Whether to scale flows based on catch_area 

base_cap_interest = 160 # If you're interested in a turbine with a particular
                        # capacity, enter it here (in kW). The script will then
                        # print the annual and season energy output and load
                        # factors for this turbine 
# #############################################################################

# Validate user input
validate_inputs(plant_eff_fac, turb_min_pct, hof_pct, pct_over_hof)

# Get list of stations to process
stn_df = read_station_data(ff_stns_csv, base_st, base_end, scale_area)

# Get lists of paths to obs and ff datasets
search_path = os.path.join(obs_fold, '*.csv')
obs_paths = glob.glob(search_path)

search_path = os.path.join(ff_fold, '*.csv')
ff_paths = glob.glob(search_path)
    
# Loop over sites
for stn_id in stn_df.index[:3]:  
    # Get station properties
    riv = stn_df.ix[stn_id]['River']
    loc = stn_df.ix[stn_id]['Location']
    area = stn_df.ix[stn_id]['Area_km2']
    area_fac = catch_area/area

    # Print progress
    print 'Currently processing: %s at %s.' % (riv, loc)
    
    # Get the time series for this station
    obs_csv = [i for i in obs_paths if 
               (os.path.split(i)[1].split('_')[1]=='%s' % stn_id)][0]
    ff_csvs = [i for i in ff_paths if 
               (os.path.split(i)[1].split('-')[2]=='%05d' % stn_id)]
    
    # Get the number of models used by FF at this site
    num_models = len(ff_csvs)

    # Read the observed data
    obs_df_full = read_obs(obs_csv, scale_area, area_fac)

    # Read the FF data
    base_df_full, fut_df_full = read_ff(ff_csvs, scale_area, area_fac)
    
    # Prepare to write output multi-page PDF
    out_pdf = os.path.join(out_fold, '%s_%05d.pdf' % (riv, stn_id))
    pdf = PdfPages(out_pdf)
       
    # Loop over seasons
    for season in ['Annual', 'Spring', 'Summer', 'Autumn', 'Winter']:
        print '    %s.' % season
        if season == 'Annual':
            obs_df = obs_df_full.copy()
            base_df = base_df_full.copy()
            fut_df = fut_df_full.copy()
        else:
            obs_df = select_season(obs_df_full.copy(), season)
            base_df = select_season(base_df_full.copy(), season)
            fut_df = select_season(fut_df_full.copy(), season)
        
        # Calculate raw FDCs
        obs_p_orig, obs_q_orig = fdc_from_obs(obs_df)
        
        (base_ff_p_orig, base_ff_q_orig, 
         fut_ff_p_orig, fut_ff_q_orig) = fdc_from_ff(base_df, fut_df)
            
        # Calc obs FDC as available for the plant
        obs_p, obs_q = calc_plant_obs_fdc(obs_df,
                                          obs_q_orig['Flow'],
                                          obs_p_orig)
        
        # Calc future FDCs as available for the plant
        (base_ff_p, base_ff_q, 
         fut_ff_p, fut_ff_q) = calc_plant_ff_fdc(base_df,
                                                 base_ff_q_orig, 
                                                 base_ff_p_orig, 
                                                 fut_df,
                                                 fut_ff_q_orig,
                                                 fut_ff_p_orig)
        
        # Turbine capacity percentages. Try all turbine sizes between 5th  
        # and 95th exceedance percentiles of available flow
        turb_cap_pcts = np.arange(5, 100, 5)
        
        # Empty lists to store data
        obs_cap_list = []
        base_cap_dict = {}
        fut_cap_dict = {}
        
        obs_en_list = []
        base_en_dict = {}
        fut_en_dict = {}
    
        obs_lf_list = []
        base_lf_dict = {}
        fut_lf_dict = {}        
        
        # Loop over turbine capacity percentages
        for pct in turb_cap_pcts:
            # Estimate optimum turbine capacity
            obs_opt_q, obs_turb_cap = select_turbine(obs_q['Flow'], 
                                                     obs_p,
                                                     pct)
            
            # Estimate annual energy output and load factor
            obs_energy, obs_lf = estimate_energy_output(obs_opt_q, 
                                                        obs_turb_cap,
                                                        pct,
                                                        turb_min_pct,
                                                        obs_q['Flow'],
                                                        obs_p,
                                                        season)
                
            # Add to dict
            obs_cap_list.append(obs_turb_cap)
            obs_en_list.append(obs_energy)
            obs_lf_list.append(obs_lf)
            
            # Process the modelled data                    
            # Get turbine caps, energies and load facs for FF baseline
            ff_results = process_ff_data(base_ff_q, 
                                         base_ff_p, 
                                         pct, 
                                         turb_min_pct,
                                         season)

            base_cap_dict[pct] = ff_results[0]
            base_en_dict[pct] = ff_results[1]
            base_lf_dict[pct] = ff_results[2]
                
            # Get turbine caps, energies and load facs for FF future
            ff_results = process_ff_data(fut_ff_q, 
                                         fut_ff_p, 
                                         pct, 
                                         turb_min_pct,
                                         season)

            fut_cap_dict[pct] = ff_results[0]
            fut_en_dict[pct] = ff_results[1]
            fut_lf_dict[pct] = ff_results[2]
            
        # Build dfs
        # Observed
        obs_cap_df = pd.DataFrame({'Pct':turb_cap_pcts,
                                   'Cap':obs_cap_list})
        obs_cap_df.index = obs_cap_df['Pct']
        del obs_cap_df['Pct']
        
        obs_en_df = pd.DataFrame({'Pct':turb_cap_pcts,
                                  'En':obs_en_list})
        obs_en_df.index = obs_en_df['Pct']
        del obs_en_df['Pct']
        
        obs_lf_df = pd.DataFrame({'Pct':turb_cap_pcts,
                                  'LF':obs_lf_list})
        obs_lf_df.index = obs_lf_df['Pct']
        del obs_lf_df['Pct']
        
        # FF baseline
        base_cap_df = pd.DataFrame(base_cap_dict).T
        base_en_df = pd.DataFrame(base_en_dict).T
        base_lf_df = pd.DataFrame(base_lf_dict).T
        
        # FF future
        fut_cap_df = pd.DataFrame(fut_cap_dict).T
        fut_en_df = pd.DataFrame(fut_en_dict).T
        fut_lf_df = pd.DataFrame(fut_lf_dict).T
            
        # Interpolate results onto suitable capacity scale and get 
        # percentiles
        base_en_df, base_lf_df = interp_en_lf(base_cap_df, 
                                              base_en_df, 
                                              base_lf_df)
                                              
        fut_en_df, fut_lf_df = interp_en_lf(fut_cap_df,
                                            fut_en_df,
                                            fut_lf_df)
        
        # Only continue if base_en_df and fut_en_df are not none
        if isinstance(base_en_df, 
                      pd.DataFrame) and isinstance(fut_en_df, 
                                                   pd.DataFrame):
            # FF baseline
            base_en_df = base_en_df.T.describe(
                         percentiles=[0.05, 0.5, 0.95]).T[['5%', 
                                                           '50%',
                                                           '95%']]
            base_lf_df = base_lf_df.T.describe(
                         percentiles=[0.05, 0.5, 0.95]).T[['5%', 
                                                           '50%',
                                                           '95%']]
            
            # FF future
            fut_en_df = fut_en_df.T.describe(
                        percentiles=[0.05, 0.5, 0.95]).T[['5%',
                                                          '50%',
                                                          '95%']]
            fut_lf_df = fut_lf_df.T.describe(
                        percentiles=[0.05, 0.5, 0.95]).T[['5%',
                                                          '50%',
                                                          '95%']]    

            # Print results for capacity of interest if specified
            if base_cap_interest:
                print_outputs_for_turbine_of_interest(base_cap_interest,
                                                      obs_cap_df,
                                                      obs_en_df,
                                                      obs_lf_df,
                                                      base_en_df,
                                                      base_lf_df,
                                                      fut_en_df,
                                                      fut_lf_df)
                                                      
            # Plot
            # Plot FDCs
            # Instead of plotting all 11 traces for baseline and future, 
            # calculate the 5th, 50th and 95th percentiles for each, then plot 
            # just these lines
            base_ff_q_orig = base_ff_q_orig.T.describe(
                             percentiles=[0.05, 0.5, 0.95]).T[['5%',
                                                               '50%',
                                                               '95%']]
            fut_ff_q_orig = fut_ff_q_orig.T.describe(
                            percentiles=[0.05, 0.5, 0.95]).T[['5%',
                                                              '50%',
                                                              '95%']]
            
            # Get canvas
            fig = plt.figure(figsize=(8, 11.5))
            ax1 = plt.subplot2grid((4,2), (0,0), colspan=2, rowspan=2)
                
            # Baseline
            ax1.fill_between(base_ff_p_orig, 
                             base_ff_q_orig['5%'].values,
                             base_ff_q_orig['95%'].values, 
                             alpha=0.2,
                             color='k')
                              
            ax1.plot(base_ff_p_orig,
                     base_ff_q_orig['50%'].values,
                     'k-', 
                     lw=1,
                     label='FF Baseline median')
                
            # Future
            ax1.fill_between(fut_ff_p_orig,
                             fut_ff_q_orig['5%'].values,
                             fut_ff_q_orig['95%'].values,
                             alpha=0.2,
                             color='r')
                              
            ax1.plot(fut_ff_p_orig,
                     fut_ff_q_orig['50%'].values,
                     'r-',
                     lw=1, 
                     label='FF Future median')
                      
            # Observed
            ax1.plot(obs_p_orig,
                     obs_q_orig['Flow'].values,
                     'b--',
                     lw=1, 
                     label='Observed')
                
            # Labelling    
            ax1.set_xlabel('Exceedance probability (%)')
            ax1.set_ylabel('Discharge ($m^3/s$)')
            
            # Create patches for legend
            p1 = plt.Rectangle((0, 0), 1, 1, fc='k', alpha=0.2)
            p2 = plt.Rectangle((0, 0), 1, 1, fc='r', alpha=0.2)
            handles, labels = ax1.get_legend_handles_labels()
            handles += [p1, p2]
            labels += ['FF Baseline 5th to 95th percentiles',
                       'FF Future 5th to 95th percentiles']
            ax1.legend(handles, labels, loc='best', fontsize=10)
            ax1.grid(True)
            ax1.set_yscale('log')
            ax1.set_title('Flow duration curves', fontsize=14)
            
            # Plot energy outputs  
            # Baseline
            ax2 = plt.subplot2grid((4,2), (2,0), colspan=1)
            ax2.fill_between(base_en_df.index,
                             base_en_df['5%'].values,
                             base_en_df['95%'].values,
                             alpha=0.2,
                             color='r')
            ax2.plot(obs_cap_df['Cap'], obs_en_df['En'], 'b--', lw=1, 
                     label='Observed')
            ax2.plot(base_en_df.index, base_en_df['50%'], 'r-', lw=1, 
                     label='FF median')
            ax2.set_title('Baseline energy output', fontsize=14)
            ax2.set_xlabel('Turbine capacity (kW)')
            ax2.set_ylabel('Energy (MWh)')
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax2.grid(True)
            
            # Future
            ax3 = plt.subplot2grid((4,2), (2,1), colspan=1, sharey=ax2, 
                                   sharex=ax2)
            ax3.fill_between(fut_en_df.index,
                             fut_en_df['5%'].values,
                             fut_en_df['95%'].values,
                             alpha=0.2,
                             color='r')
            ax3.plot(fut_en_df.index, fut_en_df['50%'], 'r-', lw=1, 
                     label='FF median')    
            ax3.set_title('Future energy output', fontsize=14)
            ax3.set_xlabel('Turbine capacity (kW)')
            plt.setp(ax3.get_yticklabels(), visible=False)
            ax3.grid(True)
            
            # Plot load factors
            # Baseline
            ax4 = plt.subplot2grid((4,2), (3,0), colspan=1, sharex=ax2)
            ax4.fill_between(base_lf_df.index,
                             base_lf_df['5%'].values,
                             base_lf_df['95%'].values,
                             alpha=0.2,
                             color='r')
            ax4.plot(obs_cap_df['Cap'], obs_lf_df['LF'], 'b--', lw=1, 
                     label='Observed')
            ax4.plot(base_lf_df.index, base_lf_df['50%'], 'r-', lw=1, 
                     label='FF median')    
            ax4.set_title('Baseline load factors', fontsize=14)
            ax4.set_ylabel('Load factor (%)')
            ax4.set_xlabel('Turbine capacity (kW)')
            ax4.yaxis.set_ticks(np.arange(0, 101, 20))
            ax4.grid(True)
            
            # Future
            ax5 = plt.subplot2grid((4,2), (3,1), colspan=1, sharey=ax4, 
                                   sharex=ax2)
            ax5.fill_between(fut_lf_df.index,
                             fut_lf_df['5%'].values,
                             fut_lf_df['95%'].values,
                             alpha=0.2,
                             color='r')
            ax5.plot(fut_lf_df.index, fut_lf_df['50%'], 'r-', lw=1, 
                     label='FF median')     
            ax5.set_title('Future load factors', fontsize=14)
            ax5.set_xlabel('Turbine capacity (kW)')
            plt.setp(ax5.get_yticklabels(), visible=False)
            ax5.grid(True)
            
            if scale_area == True:
                plt.suptitle('%s at %s (%s; area-scaled to %s km$^2$)'
                             '\nBaseline %s to %s; future %s to %s'
                             '\nBased on data from %s hydrological model(s)' 
                             % (riv,
                                loc,
                                season,
                                catch_area,
                                base_st[:4],
                                base_end[:4],
                                fut_st[:4],
                                fut_end[:4],
                                num_models),
                                fontsize=14)        
            else:
                plt.suptitle('%s at %s (%s)'
                             '\nBaseline %s to %s; future %s to %s'
                             '\nBased on data from %s hydrological model(s)' 
                             % (riv,
                                loc,
                                season,
                                base_st[:4],
                                base_end[:4],
                                fut_st[:4],
                                fut_end[:4],
                                num_models),
                                fontsize=14)
        
            # Tidy up
            plt.subplots_adjust(wspace=0.15, hspace=0.75, left=0.18, 
                                bottom=0.07, top=0.87)
            
            # Save to output
            pdf.savefig(fig)
            plt.close(fig)           
    pdf.close()