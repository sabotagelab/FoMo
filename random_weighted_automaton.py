# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:43:53 2020

@author: Colin
"""

from tqdm import trange
from igraph import *
import numpy as np


def generateGraph(num_vertices, prob_edges, min_weight, max_weight):
    # make random graph
    g = Graph.Erdos_Renyi(num_vertices, prob_edges, directed=True, loops=True)
    
    # min and max weight values
    w_min = min_weight
    w_max = max_weight
    
    # find the number of edges, and assign weights
    g.ecount()
    # change to np.random.random for non-integer weights
    weights = np.random.randint(w_min, w_max, g.ecount())
    g.es["weight"] = weights
    g.es["label"] = weights
    
    # plot(g)
    return g



def generateHistories(graph, num_histories, history_len, discount_factor):
    # generate some histories
    histories = []
    weights = graph.es["weight"]
    for i in trange(num_histories):
        # perform random walk on given graph, starting at node "1"
        walk = graph.random_walk(1, history_len)
        # compute accumulated value of the walk
        value = 0
        for depth, vertex in enumerate(walk):
            factor = discount_factor ** depth
            edge_weight = weights[vertex-1]
            value += factor * edge_weight
            
        # add walk and value to histories
        histories.append((walk, value))
        
    return histories
