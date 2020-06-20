import numpy as np
import pickle
import mdptoolbox.mdp as mdp
from copy import deepcopy
from igraph import *
from random_weighted_automaton import *


class Automaton(object):
    def __init__(self, graph, actions, q0=0):
        """
        graph is an igraph graph object
        actions is a dictionary that maps edges to actions
        key(action), values(list of edges)
        e.g. {0:[0, 1, 3], 1:[2, 4], 2:[5], ... }
        """

        self.graph = graph
        self.graph.vs["label"] = [str(v.index) for v in self.graph.vs]
        self.num_clones = 0
        self.q0 = q0
        self.q_previous = []
        for action in actions:
            for edge in actions[action]:
                edge = graph.es[edge]
                edge["action"] = action


    def k(self, i):
        """ get all actions available from vertex i """
        es = self.graph.es.select(_source=i)
        actions = []
        for edge in es:
            action = edge["action"]
            if action not in actions:
                actions.append(action)
        return actions


    def forceKn(self, kn, source=0):
        """
        delete all edges from source vertex with edges in kn that are not
        themselves in kn
        """
        # find all edges with this source
        candidates = self.graph.es.select(_source=source)
        # find all candidates not in kn
        selections = candidates.select(action_ne=kn)
        # remove these edges from the graph
        self.graph.delete_edges(selections)
        return self


    def forceQn(self, qn, source=0):
        """
        delete all edges from source vertex with edges that do not lead to given
        vertex.

        :param qn:
        :param source:
        :return:
        """
        # find all edges with this source
        candidates = self.graph.es.select(_source=source)
        # find all candidates not in qn
        selections = candidates.select(_target_ne=qn)
        # remove these edges from the graph
        self.graph.delete_edges(selections)
        return self


    def union(self, g, target=0):
        """
        modify this automaton such that transitions in itself to its initial
        state are replaced with transitions to automaton g
        """
        # recall certain properties of the given graphs
        v_mod = self.graph.vcount() + target % g.graph.vcount()

        # find the transitions to the initial state not from previous state
        es = self.graph.es.select(_target=target, _source_notin=self.q_previous)
        if not es:
            return self
        else:
            self.num_clones += 1

        labels = self.graph.vs["label"] + [label + "-" + str(self.num_clones)
                                           for label in g.graph.vs["label"]]
        weights = self.graph.es["weight"] + g.graph.es["weight"]
        actions = self.graph.es["action"] + g.graph.es["action"]
        # take the disjoint union of this graph and the given graph
        self.graph = self.graph.disjoint_union(g.graph)
        # reinstate edge and vertex attributes
        self.graph.vs["label"] = labels
        self.graph.es["weight"] = weights
        self.graph.es["action"] = actions
        properties = [(e.source, e["action"], e["weight"]) for e in es]
        # for each edge, make a replacement edge to new graph
        for edge in es:
            new_edge = self.graph.add_edge(edge.source, self.graph.vs[v_mod])
            new_edge["action"] = edge["action"]
            new_edge["weight"] = edge["weight"]
        # delete the edges
        self.graph.delete_edges(_target=target, _source_notin=self.q_previous)

        return self


    def checkCTL(self, f):
        # convert graph to nuXmv file and call nuXmv to check if it satisfies f
        #TODO: actually convert automaton and call nuXmv
        return True


    def optimal(self, discount, best=True):
        # solve graph as MDP for most optimal strategy and return value
        vcount = self.graph.vcount()
        ecount = self.graph.ecount()
        t = np.zeros((ecount, vcount, vcount))
        # punishment for the MDP choosing to stay
        # if there are negative weights, the MDP seems to want to stay
        # unless it gets punished for doing so, since 0 reward is better than -1
        punish = -1000
        r = np.full((ecount, vcount, vcount), punish)
        mod = 1
        if not best:
            mod = -1
        #TODO: documentation - before the indexing knowledge leaks out
        for i, edge in enumerate(self.graph.es):
            tup = edge.tuple
            for j in range(vcount):
                if j == tup[0]:
                    t[i, tup[0], tup[1]] = 1
                else:
                    t[i, j, j] = 1

            r[i, tup[0], tup[1]] = edge["weight"] * mod
        sol = mdp.ValueIteration(t, r, discount)
        sol.run()
        # TODO: is negated value the same as value of worst policy?
        return sol.V[0] * mod


def checkObligation(g, a):
    # return checkConditional with trivial condition params
    pass


def checkConditional(g, a, x, t):
    root = 0
    choices = g.k(0)
    intervals = []
    l = len(choices)
    for n in np.arange(l):
        kn = choices[n]
        gn = deepcopy(g)
        gn = gn.forceKn(kn)
        gnr = deepcopy(gn)
        gnp = gnr.union(g)
        # get a list of automata whose first action is kn, and have one history
        # up to depth t, and that history satisfies X, and after that it behaves
        # like g
        gns = generateFragments(gnp, g, 0, x, t)
        lows = []
        highs = []
        for gf in gns:
            lows.append(gf.optimal(0.5, best=False))
            highs.append(gf.optimal(0.5, best=True))
        interval = [np.max(lows), np.max(highs)]
        intervals.append(interval)

    # find all un-dominated intervals
    optimal = []
    inf = np.max(np.min(intervals, axis=1))
    for i, range in enumerate(intervals):
        if range[1] >= inf:
            optimal.append(choices[i])
    return optimal


def generateFragments(g, g0, q0, x, t):
    #TODO: documentation - I won't know how this works in a few days...
    f = "[]" + x
    g.q0 = q0
    systems = [g]
    for i in range(t):
        new_systems = []
        for system in systems:
            possible_states = system.graph.neighbors(system.q0, mode=OUT)
            for state in possible_states:
                sys_n = deepcopy(system)
                sys_n = sys_n.forceQn(state, source=system.q0)
                sys_n_ren = deepcopy(sys_n)
                sys_n_prime = sys_n_ren.union(g0, system.q0)

                if sys_n_prime.checkCTL(f):
                    sys_n_prime.q_previous.append(sys_n_prime.q0)
                    sys_n_prime.q0 = state
                    new_systems.append(sys_n_prime)
        systems = new_systems
    return systems


if __name__ == "__main__":
    graph = pickle.load(open("example.pkl", "rb"))
    k = {0: [3], 1: [0], 2: [6], 3: [1, 4], 4: [7, 5], 5: [2]}
    g = Automaton(graph, k)
    kn = 1
    gn = deepcopy(g)
    gn = gn.forceKn(kn)
    gnr = deepcopy(gn)
    gnp = gnr.union(g)
    gns = generateFragments(gnp, g, 0, "E0", 3)
    sol = g.optimal(0.5)
    checkConditional(g, '2', '1', 3)