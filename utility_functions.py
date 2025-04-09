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
import networkx as nx

RECOVERY_TIME_STEPS = np.insert(np.round(np.logspace(start=np.log10(0.1), stop=np.log10(1000), num=99), 2), 0, 0)

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

# Compute recovery given failure 
def Recov_given_failure(damage_state: int, time: Union[List[float], np.ndarray]= RECOVERY_TIME_STEPS) -> Union[float,np.ndarray]:
    """
    This function computes recovery given failure using Hazus parameters
    
    Arguments:
      damage_state  -- damage state. Valid inputs are 1, 2, or 3(one damage state at a time). \
                        1: "moderate damage"; 2: "extensive damage"; 3: "complete damage"
      time  -- time in days following earthquake
    Returns:
      Probability of recovery given failure 
    
    """
        
    #Hazus recovery parameters
    dict_param = {'Damage States': ['DS1','DS2','DS3','DS4'],
                  'Mean': [0.6, 2.5, 75.0, 230.0],
                  'SD': [0.6, 2.7, 42.0, 110.0]
                  }
    #Hazus recovery parameters
    # dict_param = {'Damage States': ['DS1','DS2','DS3'],
    #               'Mean': [2.5, 75.0, 230.0],
    #               'SD': [2.7, 42.0, 110.0]
    #               }

    
    #Convert dictionary to dataframes
    df_param = pd.DataFrame(dict_param)
    df_param.set_index('Damage States',inplace=True)  

    #Get paramaters corresponding to each damage state
    Mean =  df_param.iloc[damage_state-1,0]
    SD =  df_param.iloc[damage_state-1,1]

    #Compute probability of recovery
    Pr = norm.cdf(time,Mean,SD)
    
    #Return probability of recovery
    return Pr

# Compute Recovery Given Sa
def get_recov_curve(bridge_df: pd.DataFrame, 
                    Sa10: float, 
                    Sa03:float, 
                    time: Union[List[float], np.ndarray] = RECOVERY_TIME_STEPS
)-> Tuple[np.ndarray, np.ndarray]:
    
    """
    This function compute recovery given Sa10 values.
    
    Arguments:
       df_NBI_data -- dataframe of NBI bridge data
       bridge_df_imported --  imported bridge fragility functions from CSV file
       Sa10 -- Spectral acceleration value at 1.0 s in units of g
       Sa03 -- spectral acceleration at 0.3 s in g
       time  -- time in days following earthquake
    Returns:
      time and functionality varying from 0 to 1
    """
    
    DSs = [1,2,3,4] #All damage states
    Pf = [0]*5 #presizing including the no damage state
    Pf[0] = 1-bridge_frag(bridge_df,1) #No damage
    Pf[1] = bridge_frag(bridge_df,1)-bridge_frag(bridge_df,2) #slight 
    Pf[2] = bridge_frag(bridge_df,2)-bridge_frag(bridge_df,3) #moderate 
    Pf[3] = bridge_frag(bridge_df,3)-bridge_frag(bridge_df,4) #extensive
    Pf[4] = bridge_frag(bridge_df,4)                                     #complete
    Pfn = Pf[0] 
    for ds in DSs:
        Pfn = Pfn+Recov_given_failure(ds,time)*Pf[ds]

    #reset the variable names so that they are easy to be understood by the ENG team
    time_days = time
    functionality = Pfn

    return time_days, functionality 

# Compute mean and standard deviation of downtime 
def get_downtime_mean_std(bridge_df: pd.DataFrame, 
                          time: Union[List[float], np.ndarray]= RECOVERY_TIME_STEPS
) -> Tuple[float,float]:
    """
    This function computes mean and standard deviation of downtime given Sa at 1.0 s
       
    Arguments:
      df_NBI_data -- dataframe of NBI bridge data
      bridge_df_imported --  imported bridge fragility functions from CSV file
      Sa10 -- Spectral acceleration value at 1.0 s in units of g
      Sa03 -- spectral acceleration at 0.3 s in g
      time  -- time in days following earthquake (optional)
    Returns:
      bridge_downtime_days --  mean downtime in days
      stdev_bridge_downtime_days -- standard deviation of downtime in days
    
    """
    
    #Compute the functionality curve
    _,Pfn = get_recov_curve(bridge_df,time)
    
    #Compute mean downtime
    expected_downtime = np.trapz(1-Pfn,time)  
    
    #Compute standard deviation of downtime
    expected_square_downtime = np.trapz(2*time*(1-Pfn), time)  
    variance_downtime = expected_square_downtime - expected_downtime ** 2
    standard_deviation = np.sqrt(variance_downtime) 
    
    #Final output in days (not converted to hours unlike WS and floods)
    mean_downtime_days = expected_downtime
    stdev_downtime_days = standard_deviation
       
    return mean_downtime_days, stdev_downtime_days


def create_networkx_graph(nodes_gdf, 
                          edges_gdf,
                          fromnode = 'fromnode',
                          tonode = 'tonode',
                          node_id = 'nodenwid'
                         
):
    # Ensure node_id is present in nodes_gdf
    if node_id not in nodes_gdf.columns:
        raise ValueError("Ensure that the correct Node ID Column in the input nodes geodataframe is used")

    # Ensure source and target are present in edges_gdf
    if fromnode not in edges_gdf.columns or tonode not in edges_gdf.columns:
        raise ValueError("Ensure that the correct fromnode and tonode columns in the input nodes geodataframe is used")

    # Create a directed graph (change to nx.Graph() for undirected)
    # G = nx.DiGraph()
    G = nx.Graph()

    # Add nodes with attributes
    for _, row in nodes_gdf.iterrows():
        G.add_node(
            row[node_id],
            **row.drop(['geometry', node_id]).to_dict(),  # Add all columns except geometry and node_id
            geometry=row['geometry']  # Add geometry separately
        )

    # Add edges with attributes
    for _, row in edges_gdf.iterrows():
        G.add_edge(
            row[fromnode],  # Source node
            row[tonode],  # Target node
            **row.drop(['geometry', fromnode, tonode]).to_dict(),  # Add all columns except geometry, source, and target
            geometry=row['geometry']  # Add geometry separately
        )

    return G


