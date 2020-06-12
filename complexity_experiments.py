"""
Created on Fri May 15 13:43:53 2020

@author: Colin
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from copy import deepcopy
from tqdm import trange, tqdm
from joblib import Parallel, delayed
from random_weighted_automaton import *


def experiment(g_size, e_prob, w_min, w_max, h_num, h_len, factor):
    g = generateGraph(int(g_size), e_prob, w_min, w_max)
    h = generateHistories(g, h_num, h_len, factor)
    p = solveWeights(g, h, factor)
    return p["norm"]


def singleGraphExperiment(g, h_num, h_len, factor):
    h = generateHistories(g, h_num, h_len, factor)
    p = solveWeights(g, h, factor)
    return p["norm"]


def matrixExperiment(g, h_num, h_len, factor):
    h = generateHistories(g, h_num, h_len, factor)
    p = solveWeights(g, h, factor)
    return p


def compareExperiment(g, h_num, h_len, factor):
    h = generateHistories(g, h_num, h_len, factor)
    p = solveWeights(g, h, factor)
    weights2 = p["weights"]
    weights1 = g.es["weight"]

    walk = g.random_walk(0, h_len)
    # perform random walk on given graph, starting from random node
    # walk = graph.random_walk(np.random.randint(0, graph.vcount()))
    edges = []
    # compute accumulated value of the walk
    value1 = 0
    value2 = 0
    for depth in range(len(walk) - 1):
        from_vertex = walk[depth]
        to_vertex = walk[depth + 1]
        edge_tuple = ([from_vertex], [to_vertex])
        edge = g.es.find(_between=edge_tuple)
        factor = factor ** depth
        edge_weight1 = weights1[edge.index]
        edge_weight2 = weights2[edge.index]
        value1 += factor * edge_weight1
        value2 += factor * edge_weight2
        edges.append(edge.index)
    return np.abs(value1 - value2)


if __name__ == "__main__":

    num_cores = multiprocessing.cpu_count() - 1
    # g = generateGraph(100, 0.5, -5.0, 5.0)
    g = generateGraph(25, 0.5, -5.0, 5.0)
    # iterate over graph sizes
    # norms = Parallel(n_jobs=num_cores,verbose=5)(delayed(experiment)(s, 0.5, -5.0, 5.0, 1000, 1000, 0.5) for s in tqdm(np.logspace(1, 10, base=2.0)))

    # iterate over number of histories
    # norms = Parallel(n_jobs=num_cores,verbose=5)(delayed(singleGraphExperiment)(g, int(s), 10, 0.99) for s in tqdm(np.logspace(1, 4, base=10)))

    # iterate over size of histories
    # norms = Parallel(n_jobs=num_cores,verbose=5)(delayed(singleGraphExperiment)(g, 1000, int(s), 0.5) for s in tqdm(np.logspace(1, 4, base=10)))

    # iterate over discount factor
    # norms = Parallel(n_jobs=num_cores, verbose=5)(delayed(singleGraphExperiment)(g, 1000, 1000, s) for s in tqdm(np.linspace(0.01, 0.99)))

    # iterate over range of weights
    # norms = Parallel(n_jobs=num_cores,verbose=5)(delayed(experiment)(100, 0.5, s*-1.0, s, 1000, 1000, 0.5) for s in tqdm(np.logspace(1, 10, base=2.0)))

    # compare the value of a history on the true graph to a history on the graph
    # with estimated weights
    mean_diffs = []
    for s in np.linspace(0.01, 0.99):
        diffs = Parallel(n_jobs=num_cores, verbose=5)(delayed(compareExperiment)(g, g.ecount(), 100, s) for i in range(100))
        mean_diff = np.mean(diffs)
        mean_diffs.append(mean_diff)

    # check for problems with precision
    # norms = Parallel(n_jobs=num_cores, verbose=5)(
    #     delayed(experiment)(3, 0.5, -1.0*s, s, 1000, 10, 0.9) for s in
    #     tqdm(np.linspace(0.0001, 10)))

    # copmare matrix rank to condition number...
    # iterate over length of histories
    # g = generateGraph(2, 0.5, -5.0, 5.0)
    # problems = Parallel(n_jobs=num_cores, verbose=5)(
    #     delayed(matrixExperiment)(g, g.ecount(), int(s), 0.5) for s in
    #     tqdm(np.logspace(1, 4, base=10)))

    # average the iterate over number of histories
    # problem_sets = []
    # for t in trange(10):
    #     g = generateGraph(25, 0.1, -5.0, 5.0)
    #     problems = Parallel(n_jobs=num_cores, verbose=5)(
    #         delayed(matrixExperiment)(g, int(s), int(g.ecount() * 1.5), 0.99)
    #         for s in np.logspace(1, 4, base=10))
    #     problem_sets.append(problems)

    # average the iterate over range of discount factors
    # problem_sets = []
    # for i in trange(10):
    #     problems = Parallel(n_jobs=num_cores, verbose=5)(
    #         delayed(matrixExperiment)(g, g.ecount(), 25, s) for s in
    #         tqdm(np.linspace(0.01, 0.99)))
    #     problem_sets.append(problems)

    # norm_sets = []
    # rank_sets = []
    # for problems in problem_sets:
    #     norms = []
    #     ranks = []
    #     for problem in problems:
    #         norms.append(problem["norm"])
    #         ranks.append(np.linalg.matrix_rank(problem["values"]))
    #     norm_sets.append(norms)
    #     rank_sets.append(ranks)
    #
    # mean_norm = np.mean(norm_sets, axis=0)
    # mean_rank = np.mean(rank_sets, axis=0)

    # mats = []
    # norms = []
    # ranks = []
    # conds = []
    # for problem in problems:
    #     norms.append(problem["norm"])
    #     mats.append(problem["values"])
    #     ranks.append(np.linalg.matrix_rank(problem["values"]))
    #     conds.append(np.linalg.cond(problem["values"], 'fro'))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('number of histories')
    # ax.plot(np.logspace(1, 4, base=10), mean_norm, color='red')
    # ax.set_xscale('log')
    # ax.set_ylabel('residual norm', color='red')
    # ax.tick_params(axis='y', labelcolor='red')
    # ax2 = ax.twinx()
    # ax2.plot(np.logspace(1, 4, base=10), mean_rank, color='blue')
    # ax2.set_xscale('log')
    # ax2.set_ylabel('matrix rank', color='blue')
    # ax2.tick_params(axis='y', labelcolor='blue')
    # plt.savefig('mean_norms_and_ranks_by_number_of_histories_p10.pdf')
