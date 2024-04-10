"""
This module contains helper functions from Yan.

Functions:
- add(a, b): Adds two numbers together and returns the result.

Usage:
import yan_helper_functions as yhf
"""

import bikeability_functions as bf
import dem_functions as df
import numpy as np


def dem_features(dem):
    """_summary_

    Args:
        dem (_type_): _description_
    """
    dem["frac_pop_nonwhite"] = (dem["total_pop"] - dem["total_pop_white"]) / dem[
        "total_pop"
    ]
    dem["frac_below_poverty"] = dem["below_poverty"] / dem["total_pop"]
    dem["frac_no_car"] = dem["zero_vehicles"] / dem["total_workers"]


def quint_curves(G, quant_of_int, var, betas=np.linspace(0.01, 10, 200)):
    """_summary_

    Args:
        G (): _description_
        quant_of_int (_type_): _description_
        var (_type_): _description_
        betas (_type_, optional): _description_. Defaults to np.linspace(0.01, 10, 200).

    Returns:
        _type_: _description_
    """
    var_low = bf.compute_bikeability_curves(G, quant_of_int[var][0])
    var_high = bf.compute_bikeability_curves(G, quant_of_int[var][1])

    curve_1 = bf.network_wide_bikeability_curve(betas, var_low)
    curve_2 = bf.network_wide_bikeability_curve(betas, var_high)

    return curve_1, curve_2

def load_sf_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    acs_vars = {'B02001_001E':'total_pop','B02001_002E':'total_pop_white',
                'B19013_001E':'median_hh_income','B06012_002E':'below_poverty',
                'B08014_001E':'total_workers','B08014_002E':'zero_vehicles'}

    graph_sf = bf.load_graph('data/V2_SF_coarse_graph_cluster')
    dem_sf = df.get_dem('San Francisco,CA',acs_vars)

    dem_features(dem_sf)

    return graph_sf, dem_sf
