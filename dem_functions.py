import cenpy
import osmnx
import contextily
import networkx as nx

def get_dem(place,acs_vars):
    acs = cenpy.products.ACS()
    dem = acs.from_place(place,level='tract',place_type='Incorporated Place',variables=list(acs_vars.keys()))
    dem = dem.rename(acs_vars,axis='columns')
    return dem

def quant_thres(dem,var,quant=[0.2,0.8]):
    '''Return the value of quantile thresholds, but not the actual tracts'''
    quant_vals = list(dem[var].quantile(quant))
    return quant_vals

def extract_quant(dem,var,quant=[0.2,0.8]):
    '''Return list of tracts that fall within the lowest and highest quantiles for a given metric'''
    quant_vals = list(dem[var].quantile(quant))
    
    below = list(dem[dem[var]<=quant_vals[0]]['tract'])
    above = list(dem[dem[var]>=quant_vals[1]]['tract'])
    return below,above

def all_quants(keys,dem):
    quant_of_int = {}
    # values in dictionary are of format [tracts below 0.2 quantile],[tracts above 0.8 quantile]
    for key in keys:
        quant_of_int[key] = extract_quant(dem,key) 
    return quant_of_int