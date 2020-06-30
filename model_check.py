"""
Created June 2020

@author: Colin
"""

import pickle
import subprocess
import numpy as np
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
        e.g. {0:[0, 1, 3], 1:[2, 4], 2:[5], ... }.

        :param graph:
        :param actions:
        :param q0:
        """

        self.graph = graph
        self.graph.vs["label"] = [str(v.index) for v in self.graph.vs]
        self.graph.vs["name"] = [str(v.index) for v in self.graph.vs]
        self.num_clones = 0
        self.q0 = q0
        self.qn = q0
        self.q_previous = []
        self.counter = False
        for action in actions:
            for edge in actions[action]:
                edge = graph.es[edge]
                edge["action"] = action


    def k(self, i):
        """
        get all actions available from vertex i.

        :param i:
        :return:
        """
        es = self.graph.es.select(_source=i)
        actions = []
        for edge in es:
            action = edge["action"]
            if action not in actions:
                actions.append(action)
        return actions


    def setCounter(self, var_name='c', start=0):
        """
        create a simple counter in this automaton

        :param var_name:
        :return:
        """
        self.counter = (var_name, start)


    def forceKn(self, kn, source=0):
        """
        delete all edges from source vertex with edges in kn that are not
        themselves in kn.

        :param kn:
        :param source:
        :return:
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
        state are replaced with transitions to automaton g.

        :param g:
        :param target:
        :return:
        """
        # recall certain properties of the given graphs
        v_mod = self.graph.vcount() + target % g.graph.vcount()

        # find the transitions to the initial state not from previous state
        es = self.graph.es.select(_target=target, _source_notin=self.q_previous)
        # if no
        if not es:
            return self
        else:
            self.num_clones += 1

        labels = self.graph.vs["label"] + [label + "-" + str(self.num_clones)
                                           for label in g.graph.vs["label"]]
        names = self.graph.vs["name"] + g.graph.vs["name"]
        weights = self.graph.es["weight"] + g.graph.es["weight"]
        actions = self.graph.es["action"] + g.graph.es["action"]
        # take the disjoint union of this graph and the given graph
        self.graph = self.graph.disjoint_union(g.graph)
        # reinstate edge and vertex attributes
        self.graph.vs["label"] = labels
        self.graph.vs["name"] = names
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


    def optimal(self, discount, best=True, punish=-1000, steps=100):
        """
        solve graph as MDP for most (or least) optimal strategy and return value

        :param discount:
        :param best:
        :param punish:
        :param steps:
        :return:
        """
        vcount = self.graph.vcount()
        ecount = self.graph.ecount()
        # t represents the transition probabilities for each "action" from one
        # "state" to another "state, where every action is associated with a
        # transition, and every state is represented by a vertex.
        # The matrix may be considered, along the "action" vertex, as specifying
        # the probability that action has of moving the process from state A
        # to state B. As we are treating each transition as a sure thing, all
        # probabilities are 1. E.g. if edge 2 in the graph points from vertex 3
        # to vertex 1, then the entry t[2, 3, 1] = 1. The t matrix must be row
        # stochastic, so there must be an entry of 1 in each row; i.e. for each
        # source state, there must be some effect on the process if the process
        # takes that edge - even if the edge is not connected to the source.
        # In the given example, t[2, 3, j] = 0 where j != 1, since it is clear
        # that taking edge 2 from 3 to any edge other than 1 is impossible,
        # however t[2, j, j] = 1 where j != 3, since there *must* be an action
        # for each edge-state pair. Due to this discrepancy in representations,
        # the only reasonable choice is to say that trying to take edges that do
        # not begin from the current vertex leaves you in the current vertex.
        t = np.zeros((ecount, vcount, vcount))
        # Letting the process "wait" by "taking" an edge not connected to the
        # current state can be problematic when there are negative weights.
        # If there are negative weights, the MDP seems to want to wait unless
        # it gets punished for doing so, since 0 reward is better than -x.
        # To prevent this behavior, rewards for taking actions that correspond
        # with moving on edges not connected to the current vertex are deeply
        # negative. If the range of rewards for the automaton are lower than
        # this value, it will have to be changed, so it's a parameter with
        # default of -1000.
        # The r matrix, itself, has the same shape as the t matrix, and each
        # entry in r provides the reward for taking the transition in t that has
        # the same index. E.g. if moving from vertex 3 to vertex 1 via edge 2
        # has the reward of 5, then r[2, 3, 1] = 5. For "wait" actions, the
        # reward is equal to the punish value, e.g. r[2, j, j] = -1000 where
        # j != 1. NOTE: Currently, all elements of r are set to a value (either
        # a valid weight, or the punish value). Rewards associated with elements
        # of t where the transition probability is 0 may also be set to 0 if we
        # want to switch to sparse matrix representations.
        r = np.full((ecount, vcount, vcount), punish)
        # mod negates the weights of the system if we're looking for the worst
        # possible execution (except punishment weights, otherwise the system
        # would do nothing at all).
        mod = 1
        if not best:
            mod = -1
        # This loop iterates through the edges in the graph so each transition
        # matrix can be provided for every edge.
        # for each edge...
        for i, edge in enumerate(self.graph.es):
            tup = edge.tuple
            # ... for each vertex considered as source...
            for j in range(vcount):
                # ... if this vertex actually is the source of this edge...
                if j == tup[0]:
                    # ... the transition probability from source to target is 1
                    t[i, tup[0], tup[1]] = 1
                else:
                    # ... otherwise, taking this edge is a "wait" action.
                    t[i, j, j] = 1
            # ... change the reward corresponding to actually taking the edge.
            r[i, tup[0], tup[1]] = edge["weight"] * mod
        sol = mdp.FiniteHorizon(t, r, discount, N=steps)
        sol.run()
        # TODO: is negated value the same as value of worst policy?
        return sol.V[0] * mod


    def checkCTL(self, file, x):
        # convert graph to nuXmv model
        self.convertToNuXmv(file, x)
        # nuxmv = "nuXmv"
        nuxmv = "E:\\Programs\\nuXmv-2.0.0-win64\\bin\\nuXmv.exe"

        # with open("cmd.txt", 'w') as f:
        #     f.write("read_model -i " + file + "\n")
        #     f.write("flatten_hierarchy\n")
        #     f.write("encode_variables\n")
        #     f.write("build_model\n")
        #     f.write("check_ctlspec -p \"" + x + "\"")

        # out = subprocess.run([nuxmv, "-source", "cmd.txt", file], shell=True, stdout=subprocess.PIPE)
        out = subprocess.run([nuxmv, file], shell=True, stdout=subprocess.PIPE)
        check = "true" in str(out.stdout)
        return check


    def convertToNuXmv(self, file, x, lang="CTL"):
        """
        Produces a NuXmv input file specifying this automaton.
        :param file:
        :return:
        """
        with open(file, 'w') as f:
            f.write("MODULE main\n\n")
            self._writeStates(f)

            self._writeNames(f)

            self._writeVars(f)

            # begin ASSIGN constraint for state and name transitions
            f.write("ASSIGN\n")

            # States:
            self._writeStateTrans(f)

            # Names:
            self._writeNameTrans(f)

            # Properties:
            self._writePropTrans(f)

            # Specification
            f.write(lang.upper() + "SPEC " + x + ";")
            f.write("\n")

    def _writeStates(self, f):
        sep = ', '
        # include each vertex as a state in the model
        states = [str(v.index) for v in self.graph.vs]
        states = sep.join(states)
        f.write("VAR state: {" + states + "};\n\n")


    def _writeNames(self, f):
        sep = ', '
        # since multiple states can be associated with the same state of a
        # smaller original automaton, we want to track what that original
        # state is with a name variable
        names = self.graph.vs["name"]
        # remove duplicates from names
        names = list(set(names))
        # add names variable to model
        names = sep.join(names)
        f.write("VAR name: {" + names + "};\n\n")


    def _writeStateTrans(self, f):
        sep = ', '
        # set initial state
        f.write(" init(state) := " + str(self.q0) + ";\n")
        # define state transitions
        f.write(" next(state) :=\n")
        f.write("  case\n")
        # for each vertex...
        for v in self.graph.vs:
            # ... get a string representation of all the vertex's successors
            next = [str(vx.index) for vx in v.neighbors(mode=OUT)]
            next = sep.join(next)
            # and a string rep of this vertex
            state = str(v.index)
            # and write out the transitions to the case
            f.write("   state = " + state + " : {" + next + "};\n")
        # default case
        f.write("   TRUE : state;\n")
        f.write("  esac;\n")
        f.write("\n")


    def _writeNameTrans(self, f):
        # set initial name
        init_name = self.graph.vs["name"][self.q0]
        f.write(" init(name) := " + str(init_name) + ";\n")
        # define name transitions
        f.write(" next(name) :=\n")
        f.write("  case\n")
        # for each vertex...
        for v in self.graph.vs:
            # ... get that vertex's name
            name = v["name"]
            # and a string rep of this vertex
            state = str(v.index)
            # and write out the transitions to the case based on next state
            f.write("   next(state) = " + state + " : " + name + ";\n")
        # default case
        f.write("   TRUE : name;\n")
        f.write("  esac;\n")
        f.write("\n")


    def _writeVars(self, f):
        # if auto has a counter
        if self.counter:
            # ... then write the counter var
            f.write("VAR " + str(self.counter[0]) + " : integer;\n\n")


    def _writePropTrans(self, f):
        # if auto has a counter
        if self.counter:
            # ... then write the counter transitions
            c = str(self.counter[0])
            f.write(" init(" + c + ") := " + str(self.counter[1]) + ";\n")
            f.write(" next(" + c + ") := (TRUE)?(" + c + "+1):(" + c + ");\n\n")


class Obligation(object):
    """
    Contains an obligation in Dominance Act Utilitarian deontic logic
    """
    def __init__(self, phi, is_ctls, is_neg):
        """
        Creates an Obligation object

        :param phi:
        :param is_ctls:
        :param is_neg:
        """
        self.phi = phi
        self.is_ctls = is_ctls
        self.is_neg = is_neg
        self.is_stit = not is_ctls

    @classmethod
    def fromCTL(cls, phi):
        """
        Creates an Obligation object from a CTL string

        :param phi:
        :return:
        """
        return cls(phi, True, False)

    def isCTLS(self):
        """
        Checks if obligation is a well formed CTL* formula

        :return:
        """
        return self.is_ctls

    def isSTIT(self):
        """
        Checks if obligation is a well formed dstit statement

        :return:
        """
        return self.is_stit

    def isNegSTIT(self):
        """
        Checks if obligation is of the form ![alpha dstit: phi]

        :return:
        """
        return (self.is_stit and self.is_neg)

    def getPhi(self):
        """
        Gets the inner formula of the obligation

        :return:
        """
        return self.phi



def checkObligation(g, a):
    # return checkConditional with trivial condition params
    pass


def checkConditional(g, a, x, t):
    """
    Check an automaton for if it has a given obligation under a given condition.

    :param g:
    :param a:
    :param x:
    :param t:
    :return:
    """
    root = g.q0
    choices = g.k(root)
    intervals = []
    gnps = []
    l = len(choices)
    # for each choice available from start...
    for n in np.arange(l):
        kn = choices[n]
        gn = deepcopy(g)
        gn = gn.forceKn(kn)
        gnr = deepcopy(gn)
        gnp = gnr.union(g)
        gnps.append(gnp)
        # get a list of automata whose first action is kn, and have one history
        # up to depth t, and that history satisfies X, and after that it behaves
        # like g
        gns = generateFragments(gnp, g, root, x, t)
        print(len(gns))
        lows = []
        highs = []
        if gns:
            for gf in gns:
                lows.append(gf.optimal(0.5, best=False))
                highs.append(gf.optimal(0.5, best=True))
            interval = [np.max(lows), np.max(highs)]
            intervals.append(interval)

    # find all un-dominated intervals
    # optimal carries tuples containing an optimal action and an automaton
    # whose first action is that optimal action.
    optimal = []
    inf = np.max(np.min(intervals, axis=1))
    for i, range in enumerate(intervals):
        if range[1] >= inf:
            optimal.append((choices[i], gnps[i]))

    for m in optimal:
        if a.isCTLS():
            return m[1].checkCTL('temp.smv', 'A' + a.getPhi())
        elif a.isSTIT():
            phi = a.getPhi()
            if not a.isNegSTIT():
                delib = not g.checkCTL('temp.smv', phi)
                guaranteed = m[1].checkCTL('temp.smv', phi)
                return delib and guaranteed
            else:
                not_delib = g.checkCTL('temp.smv', phi)
                guaranteed = m[1].checkCTL('temp.smv', phi)
                return not_delib or not guaranteed
        else:
            raise ValueError(
                'The given obligation was not a well formed CTL* formula, ' +
                'nor a well formed deliberative STIT statement.',
                a)



def generateFragments(gn, g0, q0, x, t):
    """
    Given an Automaton g, a prototype Automaton g0, a starting state q0,
    a finite horizon condition x, and the length of that horizon t, generate
    a list of all Automata that start from q0 and have only one history up to
    depth t, that history satisfies x, and after t the Automaton behaves like
    g0.

    :param gn:
    :param g0:
    :param q0:
    :param x:
    :param t:
    :return:
    """

    g = deepcopy(gn)
    # set a clock on the automaton so the condition can be horizon limited
    # g.setCounter(var_name="fragmentc")
    # set up the condition to be checked in each step
    # f = "E [ (E" + x + ") U " + "(fragmentc = " + str(t) + ")]"
    f = 'E' + x
    # make sure we start from the right state
    g.qn = q0
    # initialize the list of systems with the given system
    systems = [g]
    # until we reach the given horizon...
    for i in range(t):
        new_systems = []
        # ... for every system we have so far...
        for system in systems:
            # ... get each possible next state for that system...
            possible_states = system.graph.neighbors(system.qn, mode=OUT)
            # ... and for each possible state...
            for state in possible_states:
                # copy the system
                sys_n = deepcopy(system)
                # make the possible next state the only next state
                sys_n = sys_n.forceQn(state, source=system.qn)
                sys_n_ren = deepcopy(sys_n)
                # tack the prototype system onto the end
                sys_n_prime = sys_n_ren.union(g0, system.qn)

                # if this new system satisfies the condition...
                if sys_n_prime.checkCTL("temp.smv", f):
                    # ... update the list of previous states
                    sys_n_prime.q_previous.append(sys_n_prime.qn)
                    # set the system's current state to the only possible next
                    # state
                    sys_n_prime.qn = state
                    # and add the system to our list of systems.
                    new_systems.append(sys_n_prime)
        # all systems have been stepped through, and the satisfactory systems
        # get to make it to the next round.
        systems = new_systems
    # now that all the systems in our list are deterministic to depth t
    # the list can be returned
    return systems


if __name__ == "__main__":
    # graph = pickle.load(open("example.pkl", "rb"))
    # k = {0: [3], 1: [0], 2: [6], 3: [1, 4], 4: [7, 5], 5: [2]}
    # g = Automaton(graph, k)
    # kn = 1
    # gn = deepcopy(g)
    # gn = gn.forceKn(kn)
    # gnr = deepcopy(gn)
    # gnp = gnr.union(g)
    # gns = generateFragments(gnp, g, 0, "F (name = 2 | name = 0)", 3)
    # sol = g.optimal(0.5)
    # checkConditional(g, '2', 'F (name = 2)', 3)
    # gns[0].graph.write_svg("graph1.svg")
    # obl = Obligation.fromCTL("AF (state = 2)")
    # out = gns[0].checkCTL("model.smv", obl.phi)

    # graph = pickle.load(open("control_graph.pkl", "rb"))
    # graph.add_edge(8, 0)
    # graph.es["weight"] = [2, 1, 2, 1, 3, 1, 2, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    # k = {0:[0], 1:[17], 2:[2], 3:[1], 4:[16], 5:[4], 6:[3], 7:[15], 8:[5],
    #      9:[7], 10:[14], 11:[6], 12:[13], 13:[8], 14:[12], 15:[9], 16:[11],
    #      17:[10], 18:[18]}
    # g = Automaton(graph, k)
    # g.setCounter(var_name='x')
    # obl = Obligation.fromCTL("X (name = 6)")
    # checkConditional(g, obl, "(x = ")

    graph = Graph(n=3, edges=[(0, 0), (0, 1), (1, 1), (0, 2), (2, 2)],
                  directed=True)
    graph.es["weight"] = [1, 3, 2, 4, 2]
    k = {0: [0, 1], 1: [3], 2: [2], 3: [4]}
    g = Automaton(graph, k)
    obl = Obligation.fromCTL("X (name = 0 | name = 1)")
    out = checkConditional(g, obl, "G (name = 0)", 2)
