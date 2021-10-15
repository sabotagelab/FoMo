"""
Created July 2020

@author: Colin
"""

from random_weighted_automaton import *
from model_check import *
import pickle

def simpleExperiment():
    graph = pickle.load(open("example.pkl", "rb"))
    k = {0: [3], 1: [0], 2: [6], 3: [1, 4], 4: [7, 5], 5: [2]}
    g = Automaton(graph, k)
    kn = 1
    gn = deepcopy(g)
    gn = gn.forceKn(kn)
    gnr = deepcopy(gn)
    gnp = gnr.union(g)
    gns = generateFragments(gnp, g, 0, "F (name = 2 | name = 0)", 3)
    sol = g.optimal(0.5)
    print("optimal actions: ", sol)
    checkConditional(g, '2', 'F (name = 2)', 3)
    gns[0].graph.write_svg("graph1.svg")
    obl = Obligation.fromCTL("AF (state = 2)")
    out = gns[0].checkCTL("model.smv", obl.phi)
    print("AF (state = 2): ", out)


def originalObligations():
    ograph1 = Graph(n=10, edges=[(0, 1), (1, 2), (1, 3), (3, 2), (3, 4), (2, 2),
                                 (4, 5), (4, 6), (5, 4), (9, 9), (6, 7), (7, 8),
                                 (8, 8), (0, 9), (1, 9), (3, 9), (4, 9), (5, 9),
                                 (6, 9), (7, 9), (8, 9)],
                    directed=True)
    ograph1.es["weight"] = [5, 2, 5, 2, 7, 0, 5, 5, 3, 0, 7, 10, 5, 0, 0, 0, 0,
                            0, 0, 0, 0]
    k0 = {0: [0, 13], 1: [1, 2, 14], 2: [3, 4, 15], 3: [6, 7, 16], 4: [8, 17],
          5: [10, 18], 6: [11, 19], 7: [12, 20], 8: [9], 9: [5]}
    og = Automaton(ograph1, k0)

    col_mission0 = og.checkCTL("temp.smv", "EG !(name = 9)")
    print("T0: Collision mission (EG !collision) = ", str(col_mission0))

    exit_mission0 = og.checkCTL("temp.smv", "EF (name = 7)")
    print("T0: Exit mission (EF safe_to_exit) = ", str(exit_mission0))

    hwy_mission0 = og.checkCTL("temp.smv", "EF (name = 4)")
    print("T0: Highway mission (EF on_highway) = ", str(hwy_mission0))

    safe_obl = Obligation.fromCTL("X !(name = 9)")

    og.q0 = 5
    has_safe0 = checkObligation(og, safe_obl)
    print("T0: Safety obligation (O[a cstit: X !collision]) = ", str(has_safe0))

    fast_obl = Obligation.fromCTL(" [TRUE U (name=6 & c<=4)]")
    fast_obl.phi_neg = True

    og.q0 = 0
    og.setCounter()
    has_fast0 = not checkObligation(og, fast_obl)
    print("T0: Fast obligation (!O[a cstit: !(True U reach_exit & c<=4)]) = ",
          str(has_fast0))

    ast_obl = Obligation(" [! (name=4) U (name=6 | name=9)]", False, False)
    ast_obl.phi_neg = True

    check_state = 5

    og.q0 = check_state
    has_ast0 = not checkObligation(og, ast_obl)
    print("T0: Assertive obligation (!O[a cstit: [a dstit: !(!g U p)]]) = ",
          str(has_ast0))

    agg_obl = Obligation(" [! (name=4) U (name=6 | name=9)]", False,
                         True)
    agg_obl.phi_neg = False

    og.q0 = check_state
    has_agg0 = not checkObligation(og, agg_obl)
    print("T0: Aggressive obligation (!O[a cstit: ![a dstit: (!g U p)]]) = ",
          str(has_agg0))


def modifiedObligations(safe=True, verbose=False):
    graph3 = Graph(n=13, edges=[(0, 1), (1, 2), (2, 2), (1, 3), (3, 2), (3, 2),
                                (3, 4), (4, 4), (4, 5), (4, 9), (5, 4), (5, 9),
                                (5, 12), (12, 12), (3, 6), (6, 6), (6, 7),
                                (7, 6), (6, 9), (9, 10), (10, 11), (11, 11)],
                   directed=True)

    if(safe):
        # not assertive, not aggressive, safe:
        graph3.es["weight"] = [5, 1, 0, 5, 1, 1, 1, 5, 5, 5, 14, 5, 0, 0, 2, 5,
                               5, 5, 0, 10, 10, 5]
    else:
        # assertive, aggressive, unsafe:
        graph3.es["weight"] = [5, 1, 0, 5, 1, 1, 1, 5, 5, 5, 4, 5, 5, 0, 2, 5,
                               5, 1, 5, 10, 10, 5]

    # assertive, not aggressive,
    # k3 = {0: [0], 1: [1, 3], 2: [2], 3: [4, 6], 5: [7, 8], 7: [10, 11, 12],
    #       8: [13], 9: [14, 5], 10: [15, 16], 11: [17], 12: [18], 13: [19],
    #       14: [20], 15: [21]}

    k3 = {0: [0], 1: [1, 3], 2: [2], 3: [4, 6], 5: [7, 8], 6: [10], 7: [11, 12],
          8: [13], 9: [14, 5], 10: [15, 16], 11: [17], 12: [18], 13: [19],
          14: [20], 15: [21]}

    g3 = Automaton(graph3, k3)
    col_mission3 = g3.checkCTL("temp.smv", "EG !(name = 12)")
    print("T3: Collision mission (EG !collision) = ", str(col_mission3))

    g3.setCounter()
    exit_mission3 = g3.checkCTL("temp.smv", "E [TRUE U (name=9 & c<=4)]")
    print("T3: Exit mission (EF{<=4} safe_to_exit) = ", str(exit_mission3))

    hwy_mission3 = g3.checkCTL("temp.smv", "EF (name = 4 | name = 6)")
    print("T3: Highway mission (EF on_highway) = ", str(hwy_mission3))

    safe_obl_re = Obligation.fromCTL("G !(name = 12)")

    g3.q0 = 6
    has_safe3 = checkObligation(g3, safe_obl_re)
    print("T3: Safety obligation (O[a cstit: G !collision]) = ", str(has_safe3))

    fast_obl_re = Obligation.fromCTL(" [TRUE U (name=9 & c<=1)]")
    fast_obl_re.phi_neg = True

    g3.q0 = 6
    g3.setCounter()
    has_fast3 = not checkObligation(g3, fast_obl_re)
    print("T3: Fast obligation (!O[a cstit: !(True U reach_exit & c<=1)]) = ",
          str(has_fast3))

    ast_obl_re = Obligation(" [! (name=4 | name=6) U (name=9 | name=12)]",
                            False, False)
    ast_obl_re.phi_neg = True

    g3.q0 = 5
    has_ast3 = not checkObligation(g3, ast_obl_re, verbose=verbose)
    print("T3: Assertive obligation (!O[a cstit: [a dstit: !(!g U p)]]) = ",
          str(has_ast3))

    agg_obl_re = Obligation(" [! (name=4 | name=6) U (name=9 | name=12)]",
                            False, True)
    agg_obl_re.phi_neg = False

    g3.counter = False
    g3.q0 = 5
    has_agg3 = not checkObligation(g3, agg_obl_re, verbose=verbose)
    print("T3: Aggressive obligation (!O[a cstit: ![a dstit: (!g U p)]]) = ",
          str(has_agg3))


if __name__ == "__main__":
    # originalObligations()
    modifiedObligations(verbose=True)
