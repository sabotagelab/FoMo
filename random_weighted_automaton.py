# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:43:53 2020

@author: Colin
"""

from tqdm import trange, tqdm
from igraph import *
import numpy as np
import cvxpy as cp


def generateGraph(num_vertices, prob_edges, min_weight, max_weight):
    # make random graph
    g = Graph.Erdos_Renyi(num_vertices, prob_edges, directed=True, loops=True)
        
    # remove terminal vertices from graph

    while len(g.vs.select(_degree_lt=2)) > 0:
        for vertex in g.vs.select(_degree_lt=2):
            vertex["current"] = True
            # connect a random vertex to this one and this one to another one
            from_vert = np.random.choice(g.vs.select(current_ne=True))
            g.add_edge(from_vert, vertex)
            to_vert = np.random.choice(g.vs.select(current_ne=True))
            g.add_edge(vertex, to_vert)
            vertex["current"] = False
        
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
    # TODO: parallelize this...
    for i in trange(num_histories):
        # perform random walk on given graph, starting at node "1"
        walk = graph.random_walk(0, history_len)
        edges = []
        # compute accumulated value of the walk
        value = 0
        for depth in range(len(walk)-1):
            from_vertex = walk[depth]
            to_vertex = walk[depth + 1]
            edge_tuple = ([from_vertex], [to_vertex])
            edge = graph.es.find(_between=edge_tuple)
            factor = discount_factor ** depth
            edge_weight = weights[edge.index]
            value += factor * edge_weight
            edges.append(edge.index)
            
        # add walk and value to histories
        histories.append((edges, value))
        
    return histories
    
    
def solveWeights(graph, histories, discount_factor):
    # make list of edges, whose weights will be solved for
    weights = cp.Variable(graph.ecount())
    
    # create value equations for each history
    values = np.zeros((len(histories), graph.ecount()))
    for h, history in tqdm(enumerate(histories)):
        word = history[0]
        value = []
        for depth, edge in enumerate(word):
            factor = discount_factor ** depth
#            factor = cp.power(discount_factor, depth)
            values[h, edge] += factor
            value.append(cp.multiply(weights[edge], factor))
#        values.append(cp.sum(value))
    
    vals = np.array(values)
    sums = np.array(histories)[:, 1]
#    cost = cp.sum_squares(vals - sums)
    val_w = values @ weights
    cost = cp.sum_squares(val_w - sums)
    
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(weights.value)
    print("The norm of the residual is ", cp.norm(vals - sums, p=2).value)
    print(prob.status)

#g = generateGraph(5, 0.5, -5.0, 5.0)
#h = generateHistories(g, 100, 1000, 0.5)
solveWeights(g, h, 0.5)
