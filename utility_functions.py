#Import packages
from typing import Union, Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, lognorm 
import math
from typing import Tuple
pd.options.mode.chained_assignment = None  # default='warn'
import os.path as path
import unittest

# Function to set default values if any of the NBI data is not given
#Set to default values if any of the required NBI data is not available
def NBI_Defaults(df_NBI_data):
    if math.isnan(df_NBI_data.loc[0,'structure_kind_043a']):
        df_NBI_data['structure_kind_043a'] = 5  
    if math.isnan(df_NBI_data.loc[0,'structure_type_043b']):
        df_NBI_data['structure_type_043b'] = 1 
    if math.isnan(df_NBI_data.loc[0,'structure_len_mt_049']):
        df_NBI_data['structure_len_mt_049'] = 50 
    if math.isnan(df_NBI_data.loc[0,'max_span_len_mt_048']):
        df_NBI_data['max_span_len_mt_048'] = 20 
    if math.isnan(df_NBI_data.loc[0,'year_built_027']):
        df_NBI_data['year_built_027'] = 1990
    if math.isnan(df_NBI_data.loc[0,'degrees_skew_034']):
        df_NBI_data['degrees_skew_034'] = 0 
    if math.isnan(df_NBI_data.loc[0,'main_unit_spans_045']):
        df_NBI_data['main_unit_spans_045'] = 3 
        
    return df_NBI_data

# Get Hazus bridge class
#Function adopted from the resilience repo
def get_hazus_class(row: pd.Series) -> str:
    """
    Given one row of the NBI data, this function returns the Hazus highway bridge class
    Arguments
      row (Tuple) -- row of the dataframe containing nbi bridge information
    Returns:
      Tuple -- row with hazus class which corresponds to fragilities info
    """
    nbi_class = 100 * row.structure_kind_043a + row.structure_type_043b
 

    # pre-1975 (for ca) and pre-1990 (for other states) bridges
    if (row.state_code_001 == 6 and row.year_built_027 < 1975) or (
        row.state_code_001 != 6 and row.year_built_027 < 1990
    ):
        if nbi_class >= 101 and nbi_class <= 106:
            if row.state_code_001 != 6:
                hazus_class = "HWB5"
            else:
                hazus_class = "HWB6"

        elif (nbi_class == 205 or nbi_class == 206) and row.state_code_001 == 6:
            hazus_class = "HWB8"

        elif nbi_class >= 201 and nbi_class <= 206:
            hazus_class = "HWB10"

        elif nbi_class >= 301 and nbi_class <= 306:
            if row.state_code_001 != 6:
                if row.structure_len_mt_049 > 20:
                    hazus_class = "HWB12"
                else:
                    hazus_class = "HWB24"
            else:
                if row.structure_len_mt_049 > 20:
                    hazus_class = "HWB13"
                else:
                    hazus_class = "HWB25"

        elif nbi_class >= 402 and nbi_class <= 410:
            if row.structure_len_mt_049 > 20:
                hazus_class = "HWB15"
            else:
                if row.state_code_001 != 6:
                    hazus_class = "HWB26"
                else:
                    hazus_class = "HWB27"

        elif nbi_class >= 501 and nbi_class <= 506:

            if row.state_code_001 != 6:
                hazus_class = "HWB17"
            else:
                hazus_class = "HWB18"

        elif (nbi_class == 605 or nbi_class == 606) and row.state_code_001 == 6:
            hazus_class = "HWB20"

        elif nbi_class >= 601 and nbi_class <= 607:
            hazus_class = "HWB22"

        else:
            if row.max_span_len_mt_048 > 150:
                hazus_class = "HWB1"

            elif row.main_unit_spans_045 == 1:
                hazus_class = "HWB3"

            else:
                hazus_class = "HWB28"

    # post-1975 (for ca) and post-1990 (for other states) bridges
    else:
        if nbi_class >= 101 and nbi_class <= 106:
            hazus_class = "HWB7"

        elif (nbi_class == 205 or nbi_class == 206) and row.state_code_001 == 6:
            hazus_class = "HWB9"

        elif nbi_class >= 201 and nbi_class <= 206:
            hazus_class = "HWB11"

        elif nbi_class >= 301 and nbi_class <= 306:
            hazus_class = "HWB14"

        elif nbi_class >= 402 and nbi_class <= 410:
            hazus_class = "HWB16"

        elif nbi_class >= 501 and nbi_class <= 506:
            hazus_class = "HWB19"

        elif (nbi_class == 605 or nbi_class == 606) and row.state_code_001 == 6:
            hazus_class = "HWB21"

        elif nbi_class >= 601 and nbi_class <= 607:
            hazus_class = "HWB23"

        else:
            if row.max_span_len_mt_048 > 150:
                hazus_class = "HWB2"

            elif row.main_unit_spans_045 == 1:
                hazus_class = "HWB4"

            else:
                hazus_class = "HWB28"

    return hazus_class

# Obtain fragility function for the Hazus class
def get_fragilities(df_NBI_data: pd.DataFrame, bridge_df_imported: pd.DataFrame) -> pd.DataFrame:
    """
    This function selects fragility function paramaters for the bridge class and combines them with NBI attributes
    Arguments
      df_NBI_data -- dataframe of NBI bridge data
      bridge_df_imported --  imported bridge fragility functions from CSV file
    Returns:
      Bridge data frame with all the required parameters including the NBI database parameters
    """
    # Get the row of the NBI data corresponding to the bridge
    df_NBI_data = NBI_Defaults(df_NBI_data) #set default values if some of the NBI data is not available
    classes = []
    
    for i, row in df_NBI_data.iterrows():
        #Obtain hazus bridge fragilities
        hazus_class = get_hazus_class(row)
        classes.append(hazus_class)
        
    df_NBI_data['HAZUS_CLASS'] = classes
    df_NBI_data = df_NBI_data.merge(bridge_df_imported)
    
    return df_NBI_data

def adjust_fragilities(bridge_df):
    
    #Limit the skew angle to 45 degrees (see documentation for reasoning)
    if bridge_df.loc[0,"degrees_skew_034"] > 45:
        bridge_df["degrees_skew_034"] = 45
    
    #Convert angles from degrees into radians
    bridge_df["skew_angle_radian"] = bridge_df["degrees_skew_034"] * (np.pi / 180.0)  # in radians
    Ninty_radian = 90*np.pi / 180 #convert 90 degrees to radian

    #Compute Kskew parameter
    bridge_df["K_skew"] = np.sqrt(np.sin(Ninty_radian-bridge_df["skew_angle_radian"]))
 
    
    #Compute k3d parameter
    bridge_df["K_3d"] = 1.0
    span_b_diff_nz = bridge_df["main_unit_spans_045"] - bridge_df["B"] != 0 #N-B i.e., number of spans - B coeff.  
    bridge_df.loc[span_b_diff_nz, "K_3d"] = 1 + bridge_df.loc[span_b_diff_nz, "A"] / (
        bridge_df.loc[span_b_diff_nz, "main_unit_spans_045"] - bridge_df.loc[span_b_diff_nz, "B"]
    ).astype("float")
            
    #Compute Kshape parameter
    bridge_df["K_shape"] =  2.5*bridge_df["Sa10"]/bridge_df["Sa03"]
    
    #Modify fragility median values for slight damage
    if bridge_df.loc[0,"I_shape"] == 1:
        bridge_df["median_slight_Sa10"] = bridge_df["median_slight_Sa10"]*bridge_df["K_shape"]
        bridge_df["median_slight_Sa10"] = min(1,bridge_df["median_slight_Sa10"].to_numpy())

    #Modify fragility median values for higher than slight damage state
    median_sa_cols = ["median_moderate_Sa10", "median_extensive_Sa10", "median_complete_Sa10"]
    bridge_df[median_sa_cols] = bridge_df[median_sa_cols].multiply(
        bridge_df["K_skew"] * bridge_df["K_3d"], axis="index"
    )
    
    return bridge_df


# Compute fragility functions given parameters
def bridge_frag(bridge_df, damage_state: int)\
-> Union[float, np.ndarray]:
        
    if damage_state == 1:
        Median = bridge_df['median_slight_Sa10']
        Beta = bridge_df['beta_slight_Sa10']
    elif damage_state == 2:
        Median = bridge_df['median_moderate_Sa10']
        Beta = bridge_df['beta_moderate_Sa10']
    elif damage_state == 3:
        Median = bridge_df['median_extensive_Sa10']
        Beta = bridge_df['beta_extensive_Sa10']
    else:
        Median = bridge_df['median_complete_Sa10']
        Beta = bridge_df['beta_complete_Sa10']       
    
    Sa10 = np.array(bridge_df['Sa10'])
    
    #Generate the fragility curve
    Pf = norm.cdf(np.log(Sa10), np.log(Median),Beta)
    
    #Return probability of failure
    return Pf
