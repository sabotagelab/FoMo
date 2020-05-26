"""
Created on Fri May 15 13:43:53 2020

@author: Colin
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import trange, tqdm
from joblib import Parallel, delayed
from random_weighted_automaton import *


def experiment(g_size, e_prob, w_min, w_max, h_num, h_len, factor):
    g = generateGraph(int(g_size), e_prob, w_min, w_max)
    h = generateHistories(g, h_num, h_len, factor)
    p = solveWeights(g, h, factor)
    return p["norm"]


if __name__ == "__main__":

    # iterate over graph sizes
    num_cores = multiprocessing.cpu_count() - 1
    # norms = Parallel(n_jobs=num_cores,verbose=5)(delayed(experiment)(s, 0.5, -5.0, 5.0, 1000, 1000, 0.5) for s in tqdm(np.logspace(1, 10, base=2.0)))
    # norms = Parallel(n_jobs=num_cores,verbose=5)(delayed(experiment)(100, 0.5, -5.0, 5.0, int(s), 1000, 0.5) for s in tqdm(np.logspace(1, 4, base=10)))
    norms = Parallel(n_jobs=num_cores,verbose=5)(delayed(experiment)(100, 0.5, -5.0, 5.0, 1000, int(s), 0.5) for s in tqdm(np.logspace(1, 4, base=10)))


    plt.plot(norms)