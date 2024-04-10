import cenpy
import osmnx
import contextily
import networkx as nx
import pandas as pd
import pickle
import geopandas
import numpy as np
import heapq
import random
import geopy.distance
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
from pyproj import Transformer
from kneed import KneeLocator

def add_discomfort(G,scale='linear'):
    '''adds discomfort attribute to each edge as a function of its distance and layer'''
    distances = nx.get_edge_attributes(G,'distance')
    discomforts = {}
    for edge in G.edges:
        key = (edge[0],edge[1],0)
        beta = [None]*4
        if scale=='linear':
            beta[0] = 0
            beta[1] = distances[key][1]*0.33
            beta[2] = distances[key][2]*0.66
            beta[3] = distances[key][3]
        elif scale=='threshold':
            beta[0] = 0
            beta[1] = distances[key][1]*0.33
            beta[2] = distances[key][2]
            beta[3] = distances[key][3]*100
        discomforts[key] = {'discomfort':beta}
    nx.set_edge_attributes(G,discomforts)
    return G

def load_graph(file_name,discomfort_scale='linear'):
    '''load networkx graph from data and do basic cleaning'''
    with open(file_name, 'rb') as f:
        G = pickle.load(f)
        
    # only retain giant component (in case there are isolated nodes)
    GC = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(GC)
        
    # incorporate discomfort function
    G=add_discomfort(G,scale=discomfort_scale)
    
    return G

def updateParetoFront(new_length, new_risk, best_length_risk_states):
    '''helper function that determines if path currently being explored is part of the pareto front
    (i.e. if there is no other OD path with shorter length AND lower risk) and updates if warranted'''
    # Find indices before and after where this path would fit in the sorted length axis of pareto front
    after_idx = best_length_risk_states.bisect_right(new_length)
    before_idx = after_idx - 1
    after_risk = -1
    before_risk = -1 
    
    # Find associated risks
    keys = best_length_risk_states.keys()
    if after_idx < len(best_length_risk_states):
        after_risk = best_length_risk_states.get(keys[after_idx])
    if before_idx >= 0:
        before_risk = best_length_risk_states.get(keys[before_idx])
       
    # Exclude new path if the shorter path has same or less risk
    if before_risk <= new_risk and not before_risk == -1:
        return False

    # Exclude longer or equal paths with higher risks:
    final_exclude_idx = after_idx
    while(final_exclude_idx < len(best_length_risk_states)):
        risk = best_length_risk_states.get(keys[final_exclude_idx])
        if risk < new_risk:
            break
        final_exclude_idx = final_exclude_idx + 1
    del best_length_risk_states.keys()[after_idx:final_exclude_idx]
    best_length_risk_states[new_length] = new_risk
    
    return True

def compute_pareto_fronts(G, u):
    '''given a network graph and the osmid (key) of an origin node u, returns a dictionary that maps each node 
    in the network to the pareto fronts from u to v '''
    
    states_to_explore = []
    best_length_risk_states_for_node = {u : SortedDict({0: 0})}
    cnt = 0
    # keep track of paths that need to be explored
    heapq.heappush(states_to_explore, (0, 0, cnt, {'distance': 0, 'risk': 0, 'prevstate': None, 'node': u}))
    
    while len(states_to_explore) > 0:
        current_distance, current_risk, dummy, current_state = heapq.heappop(states_to_explore)
        current_node = current_state.get('node')
        
        # Check if the explored path is still part of the optimal length risk curve.
        # Otherwise, discard.        
        best_length_risk_states = best_length_risk_states_for_node.get(current_node)
        if best_length_risk_states.get(current_distance, -1) != current_risk:
            continue
        
        # Extend path in all directions possible from current node
        for tt in G.out_edges(current_node):
            next_node = tt[1]
            edge = G[current_state.get('node')][next_node][0]
            
            # Get pareto fronts from origin to this node
            best_length_risk_states = best_length_risk_states_for_node.get(next_node, None)
            if best_length_risk_states is None:
                best_length_risk_states = SortedDict()
                best_length_risk_states_for_node[next_node] = best_length_risk_states
            
            # Loop over different network layer choices for link
            for network_i in range(len(edge['distance'])):
                
                # Compute new length
                edge_length = edge['distance'][network_i]
                if np.isnan(edge_length):
                    continue
                new_length = current_distance + edge_length
                
                # Compute new discomfort
                edge_risk = edge['discomfort'][network_i]
                new_risk = current_risk + edge_risk
                
                # True if this path is pareto optimal; false otherwise (if there is a path of shorter length and less risk)
                inserted = updateParetoFront(new_length, new_risk, best_length_risk_states)
                if not inserted:
                    continue
                
                new_state = {'distance': new_length, 'risk': new_risk, 'prevstate': current_state, 'node': next_node}
                
                heapq.heappush(states_to_explore, (new_length, new_risk, cnt, new_state))
                cnt = cnt + 1
                
    return best_length_risk_states_for_node

def euclidean_distance(u,v):
    '''helper function that calculates Euclidean distance between two nodes (census tract centroids)'''
    inProj = 'epsg:3857' # Web mercator coordinate system
    outProj = 'epsg:4326' # lat long coordinate system
    
    transformer = Transformer.from_crs(inProj, outProj)

    ux_proj,uy_proj = transformer.transform(u[0],u[1])
    vx_proj,vy_proj = transformer.transform(v[0],v[1])

    euclid_dist=geopy.distance.geodesic((ux_proj,uy_proj), (vx_proj,vy_proj)).m
    return euclid_dist

def normalise_and_add_bikeability_curve(G, u, v, curve, output_data, color='royalblue'): 
    '''helper function that appends curve for one OD pair to a running list'''
    # normalize based on shortest available route 
    rel_dist=np.array(curve.keys())/min(curve.keys())
    rel_risk=np.array(curve.values())/min(curve.keys())
    euclid_dist = euclidean_distance((G.nodes[u]['x'],G.nodes[u]['y']),(G.nodes[v]['x'],G.nodes[v]['y']))
    
    output_data.append({'x': rel_dist, 'y': rel_risk, 'dist': euclid_dist})
    #output_data.append(go.Scatter(x = rel_dist, y= rel_risk, dist = euclid_dist,
    #                            marker=dict(color=color,size=7,opacity=0.9),
    #                            opacity=0.5,
    #                            line=dict(color= color,width=1.3), 
    #                            mode = 'markers+lines',showlegend=False))   
    return output_data

def compute_bikeability_curves(G,node_set,dest='subset'):
    '''given a street network, calculates bikeability curves (pareto fronts) for OD pairs in a specified 
       subset of all possible points. if dest='subset' destination nodes must be in the same subset as
       origin nodes. if dest='all' destination nodes are all other nodes in the network.'''
    bike_curves=[]
    for u in node_set:
        fronts = compute_pareto_fronts(G,u)
        if dest=='subset':
            for v in node_set:
                if u!=v and nx.has_path(G, u, v):
                    bike_curves=normalise_and_add_bikeability_curve(G,u,v,fronts[v],output_data=bike_curves)
        elif dest=='all':
            for v in G.nodes:
                if u!=v and nx.has_path(G, u, v):
                    bike_curves=normalise_and_add_bikeability_curve(G,u,v,fronts[v],output_data=bike_curves)
    
    return bike_curves

def utility(risk,dist,alpha): 
    # utility function of the users
    return alpha*risk+dist

def network_wide_bikeability_curve(betas, bike_curves,weighted=True):
    '''given individual bikeability curves for all desired origin-destination pairs, constructs an aggregate
       (network-wide) bikeability curve by finding the expected distance and discomfort of optimal paths for
       a spectrum of user preferences represented by beta (weight of discomfort relative to distance in a 
       linear disutility function)'''
    risk_list=[list(i['y']) for i in bike_curves]
    dist_list=[list(i['x']) for i in bike_curves]
    if weighted:
        weight_list=[1/i['dist'] for i in bike_curves]
    else:
        weight_list = [1 for i in bike_curves]
    #risk_list=[list(i.y) for i in bike_curves]
    #dist_list=[list(i.x) for i in bike_curves]
    user_dict={}

    for u in betas: 
    # for each user preference we fill in the dictionary[alpha:[optimas, optima_dist,optima_risk]
        optima=[]
        opt_dist=[]
        opt_risk=[]
        for i in np.arange(len(risk_list)):
            a=[u]*len(risk_list[i])
            result = min(list(map(utility,risk_list[i],dist_list[i],a)))
            best_index=np.argmin(list(map(utility,risk_list[i],dist_list[i],a)))

            optima.append(result)
            opt_dist.append(dist_list[i][best_index])
            opt_risk.append(risk_list[i][best_index])
        user_dict[u]=[np.average(optima,weights=weight_list),
                  np.average(opt_dist,weights=weight_list), np.average(opt_risk,weights=weight_list)]
    return user_dict

def compare_curves(user_dict_1,user_dict_2,layout):
    '''old function for comparing two bikeability curves'''
    data_plot=[]
    data_plot.append(go.Scatter(x= np.array([data[1] for _,data in user_dict_1.items()]),
                                y=np.array([data[2] for _,data in user_dict_1.items()]),
                                marker=dict(color='black',size=10), 
                                mode = 'markers',showlegend=False))
    data_plot.append(go.Scatter(x= np.array([data[1] for _,data in user_dict_2.items()]),
                                y=np.array([data[2] for _,data in user_dict_2.items()]),
                                marker=dict(color='blue',size=10), 
                                mode = 'markers',showlegend=False))

    iplot({"data":data_plot,"layout":layout})
    
def calc_elbow(user_dict):
    '''given a bikeability curve, finds the elbow point of maximum curvature and calculates proximity 
       (inverse distance) to the point (1,0) in normalized (distance,discomfort) space'''
    x = np.array(list(user_dict.values()))[:,1]
    y = np.array(list(user_dict.values()))[:,2]

    kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='decreasing')
    dist =((kneedle.elbow-1)**2 + kneedle.elbow_y**2)**0.5

    elbow_prox = 1/dist
        
    return elbow_prox