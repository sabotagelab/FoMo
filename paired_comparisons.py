# -*- coding: utf-8 -*-
"""
Created Jan 2021

@author: Colin
"""


import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import choix
import mdptoolbox
from model_check import Automaton
from collections import Counter
from copy import deepcopy
from tqdm import trange, tqdm
from joblib import Parallel, delayed
from random_weighted_automaton import *
from igraph import Graph

def tournament(elements, weights, population, std=2.5, normal=True, free_cores=1):
    # given a set of weighted trajectories, return a tournament
    val_dict = dict(zip(elements, weights))
    matches = itertools.combinations(elements, 2)
    n = len(elements)
    tot = int(np.ceil(0.5*(n-1)*n))
    # TODO: make it faster
    num_cores = multiprocessing.cpu_count() - free_cores
    results = Parallel(n_jobs=num_cores, verbose=5)(
        delayed(_per_match)(match, val_dict, population, std, normal) for match in
        tqdm(matches, total=tot))
    return list(itertools.chain.from_iterable(results))


def _per_match(match, val_dict, population, std, normal=True):
    val1 = val_dict[match[0]]
    val2 = val_dict[match[1]]
    if normal:
        util1 = np.random.normal(val1, scale=std, size=population)
        util2 = np.random.normal(val2, scale=std, size=population)
    else:
        l1 = val1 - 2 * std
        h1 = val1 + 2 * std
        l2 = val2 - 2 * std
        h2 = val2 + 2 * std
        util1 = np.random.uniform(l1, h1, size=population)
        util2 = np.random.uniform(l2, h2, size=population)
    win = (match[0], match[1])
    loss = (match[1], match[0])
    match_results = [win if u1 > u2 else loss for u1, u2 in zip(util1, util2)]
    return match_results


def comparison_scores(tournament, method):
    # given a tournament, return a score for each element
    score_dict = method(tournament)
    return score_dict.items()


# TODO: fix any bugs!
# TODO: normalize weight vectors for better comparison
# TODO: empirical sensitivity analysis
# TODO: multiple runs per experimental configuration and average results
# TODO: include scale of utility distribution in experimental configurations


def d_compare(tournament):
    # implementation of deterministic custom method of paired comparison
    # turn the tournament into a DAG, where each vertex is a "competitor" from
    # the tournament, and each edge from vertex u to v indicates that u "beat"
    # v. That edge's weight indicates how many time u beat v. A vertex's ID in
    # the DAG is the ID of the trajectory it represents.
    dag = tourn_to_dag(tournament)
    # find the diameter of the graph
    l = dag.diameter(unconn=False)
    # using the diameter, determine the change (d) in weight between "layers"
    d = 1.0/l
    # sort the graph topologically
    topo = dag.topological_sorting()
    # for each vertex, if it's a root its weight is one, else its weight is
    # min(weight(parents))-d
    dag.vs["utility"] = np.zeros(len(topo))
    utility = {}
    for v in dag.vs[topo]:
        if v.indegree() == 0:
            utility[v.index] = v["utility"] = 1.0
        else:
            pred_util = [pred["utility"] for pred in v.predecessors()]
            utility[v.index] = v["utility"] = np.min(pred_util) - d

    return utility


def s_compare(tournament):
    # implementation of stochastic custom method of paired comparison
    # turn the tournament into a DAG, where each vertex is a "competitor" from
    # the tournament, and each edge from vertex u to v indicates that u "beat"
    # v. That edge's weight indicates how many time u beat v. A vertex's ID in
    # the DAG is the ID of the trajectory it represents.
    dag = tourn_to_dag(tournament)
    # find the diameter of the graph, this is the maximum number of layers
    l = dag.diameter(unconn=False)
    # generate a utility value for each layer
    utilities = []
    high = 1.0
    for layer in range(l):
        util = np.random.uniform(0.0, high)
        utilities.append(util)
        high = util
    # topologically sort the graph
    topo = dag.topological_sorting()
    # for each vertex, if it's a root, its layer is zero, else its layer is
    # max(layer(parents))+1
    utility = {}
    for v in dag.vs[topo]:
        if v.indegree() == 0:
            utility[v.index] = v["utility"] = utilities[0]
            v["layer"] = 0
        else:
            pred_layer = [pred["layer"] for pred in v.predecessors()]
            v["layer"] = np.max(pred_layer) + 1
            utility[v.index] = v["utility"] = utilities[v["layer"]]

    return utility


def borda(tournament):
    c = Counter()
    for comp in tournament:
        c[comp[0]] += 1.0
    max_c = max(c.values())
    c = dict(c)
    for key in c:
        c[key] = c[key]/max_c
    return c


def bradley_terry(tournament):
    e_count = Counter(tournament)
    l = len({item[1] for item in list(e_count)})
    # generate adjacency matrix
    g = Graph(edges=list(e_count), directed=True)
    g.es["weight"] = list(e_count.values())
    mat = np.array(g.get_adjacency(attribute='weight').data)
    params = choix.lsr_pairwise_dense(mat, alpha=0.0001)
    # params = choix.mm_pairwise(l, tournament, alpha=0.0001)
    return dict(zip(np.arange(l), params))


def tourn_to_dag(tournament):
    # given a tournament, return a DAG
    e_count = Counter(tournament)
    dag = Graph(edges=list(e_count), directed=True)
    dag.es["weight"] = list(e_count.values())
    fas = dag.feedback_arc_set(weights="weight")
    dag.delete_edges(fas)
    return dag


def calc_policy(elements, scores, automaton, discount_factor):
    # given trajectories, their scores, and the generating automaton...
    # find the weights on the transitions in the automaton, return a policy
    # solve transition weights on automaton
    histories = [[elements[s[0]], s[1]] for s in scores]

    wt_sol = solveWeights(automaton, histories, discount_factor)
    new_auto = automaton.copy()
    new_auto.es["weight"] = wt_sol["weights"]
    # convert automaton graph to Automaton object
    policy = get_policy(new_auto, discount_factor)
    return (policy, wt_sol["norm"])


def get_policy(automaton, discount_factor, iterations=50):
    # convert automaton graph to Automaton object
    actions = {}
    for i in range(len(automaton.es)):
        actions[i] = [i]
    auto = Automaton(automaton, actions)
    # learn policy via backwards induction
    tr = auto.to_mdp()
    mdp = mdptoolbox.mdp.FiniteHorizon(tr[0], tr[1], discount_factor,
                                       N=iterations)
    mdp.run()
    return mdp.policy[:, -1]


def policy_dist(policy1, policy2, distance):
    # given two policies, return a distance
    # variations based on distance metric, or take distance function as param
    # for each state in the policies determine the difference between them
    distances = []
    for state in range(len(policy1)):
        state1 = None
        state2 = None
        try:
            iter(policy1[state])
            state1 = policy1[state]
        except:
            state1 = [policy1[state]]
        try:
            iter(policy2[state])
            state2 = policy2[state]
        except:
            state2 = [policy2[state]]
        distances.append(distance(state1, state2))
    return np.linalg.norm(distances)


def jaccard_dist(list1, list2):
    # take the jaccard set distance between two iterables
    # incredibly neither scikit-learn or scipy have this version implemented!
    intersection = len(list(set(list1).intersection(list2)))
    union = len(list1) + len(list2) - intersection
    return (union-intersection)/union


def experiment(g_size, e_prob, w_min=0, w_max=10, factor=0.9, pop=1000,
               normal=True, n_h=None, l_h=None, val_scale=2.5, d_b=None):
    # driver code to generate automata and run analysis on each
    g = generateGraph(g_size, e_prob, w_min, w_max)
    # h_num and h_len should be functions of g_size and e_prob
    # TODO: is h_num something reasonable now?
    # Kinda... h_num is probably bigger than needed
    if n_h is None:
        h_num = len(g.es)
    else:
        h_num = n_h

    if l_h is None:
        h_len = int(1.5*g_size)
    else:
        h_len = l_h
    # h[k] is [edge_list, value]
    h = generateHistories(g, h_num, h_len, factor)
    # NOTE: consider assigning new values to each history drawn at random
    true_values = []
    trajectories = []
    traj_id = []
    for i in range(len(h)):
        true_values.append(h[i][1])
        trajectories.append(h[i][0])
        traj_id.append(i)

    population = pop
    tourn = tournament(traj_id, true_values, population, normal=normal, std=val_scale)

    methods = [d_compare, s_compare, bradley_terry, borda]
    policies = []
    wgt_norms = []

    # determine policy for original (no re-weighting)
    wgt_norms.append(0.0)
    policies.append(get_policy(g, factor))

    # determine policy and re-weighting error for original
    true_scores = dict(zip(traj_id, true_values)).items()
    result = calc_policy(trajectories, true_scores, g, factor)
    policies.append(result[0])
    wgt_norms.append(result[1])

    for method in methods:
        print("scoring...")
        scores = comparison_scores(tourn, method)
        print("calculating policy...")
        result = calc_policy(trajectories, scores, g, factor)
        policies.append(result[0])
        wgt_norms.append(result[1])

    if d_b is not None:
        d = np.random.normal(size=len(h))
        n = d_b ** (1 / len(h))
        l = np.linalg.norm(d)
        x = d * (n / l)
        false_values = true_values + x
        perturbed_tourn = tournament(traj_id, false_values, population,
                                     normal=normal, std=val_scale)
        perturbed_policies = []
        policies = policies[1:]
        # determine policy and re-weighting error for original
        false_scores = dict(zip(traj_id, false_values)).items()
        result = calc_policy(trajectories, false_scores, g, factor)
        perturbed_policies.append(result[0])
        for method in methods:
            print("scoring...")
            scores = comparison_scores(perturbed_tourn, method)
            print("calculating policy...")
            result = calc_policy(trajectories, scores, g, factor)
            perturbed_policies.append(result[0])

        perturbation = []
        dist = jaccard_dist
        for pair in zip(policies, perturbed_policies):
            perturbation.append(policy_dist(pair[0], pair[1], dist))

        return perturbation


    d_mat = np.zeros((len(policies), len(policies)))
    dist = jaccard_dist
    for j in range(len(policies)):
        for k in range(len(policies)):
            d_mat[j, k] = policy_dist(policies[j], policies[k], dist)

    return (d_mat, wgt_norms)

if __name__ == "__main__":
    size = []
    # size_range = range(5, 50, 2)
    # size_range = [5, 10, 15, 20, 25, 30, 35, 40]
    # prob_range = range(3, 9, 1)
    batch_num = range(5)
    size_range = [25]
    # prob_range = [3, 5, 7]
    # prob_range = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    prob_range = [0.1, 0.5, 1, 2, 3, 4, 5, 6]
    # prob_range = [5]
    # for sz in size_range:
    #     prob = []
    #     for pr in prob_range:
    #         p = pr/10.0
    #         dmats = []
    #         wgtns = []
    #         for batch in batches:
    #             print("Current Graph Size is", sz)
    #             print("Current Connection P is", pr)
    #             print("Current Batch is", batch)
    #             result = experiment(sz, p, pop=500, normal=False)
    #             dmats.append(result[0])
    #             wgtns.append(result[1])
    #         # TODO: variance as well as mean?
    #         dmat = np.array(dmats).mean(0)
    #         wgtn = np.array(wgtns).mean(0)
    #         prob.append((dmat, wgtn))
    #     size.append(prob)

    batches = []
    # pops = np.linspace(2, 500, 10, dtype=int)
    # for pop in pops:
    #     dmats = []
    #     wgtns = []
    #     for batch in batch_num:
    #         print("Current pop is", pop)
    #         print("Current batch is", batch)
    #         result = experiment(25, 0.6, pop=pop, normal=False)
    #         dmats.append(result[0])
    #         wgtns.append(result[1])
    #     dmat = np.array(dmats).mean(0)
    #     wgtn = np.array(wgtns).mean(0)
    #     batches.append((dmat, wgtn))

    d_bs = np.linspace(0, 50, 10)
    for d_b in d_bs:
        diffs = []
        for batch in batch_num:
            print("Current d_b is", d_b)
            print("Current batch is", batch)
            result = experiment(25, 0.6, pop=500, d_b=d_b)
            diffs.append(result)
        mean_diff = np.array(diffs).mean(0)
        vars_diff = np.array(diffs).var(0)
        batches.append((mean_diff, vars_diff))