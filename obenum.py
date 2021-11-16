"""
Created October 2021

@author: Colin

Enumerate the obligations of a stit automaton from a state in an automaton
Should this be a function of a stit automaton? Not for now, in case we want it to generalize to pstit automata.
based on "Interpretable Classification of Time-Series Data Using Efficient Enumerative Techniques"
by Mohammadinejad, Deshmukh, Puranic, Vazquez-Chanlatte, and Donze
"""
# TODO: experiment with the expressiveness of the grammar
# TODO: a class that is a grammar, and a subclass that is CTL? A class for unary operators and binary operators?
# TODO: signature or logical equivalence optimization
# TODO: enumerate obligations instead of CTL formulas; use checkObligation or checkConditional instead
# ^ maybe a different set of functions? That way we can include stit statements and nested obligations
# TODO: conditional obligation checking
# ^ need to get optimal fragments w.r.t. the condition.
# TODO: rename file, because we're gonna do proper searching soon

# TODO: function to step through the grammar tree of a given language
# ^ maybe not deterministically
# TODO: function to estimate the reward for a given equation
# TODO: function to return the best reward given db, vdb

from model_check import get_choice_fragments, get_optimal_automata, get_choice_automata, choose_optimal_automata
from tqdm import trange, tqdm


def enum(auto, l_max, horizon, atoms, condition="TRUE", unary_ops=[], binary_ops=[]):

    if not unary_ops:
        # set unary_ops to CTL ops
        paren = lambda x: "(" + x + ")"
        neg = lambda x: "!" + x
        eg = lambda x: "EG " + x
        ex = lambda x: "EX " + x
        ef = lambda x: "EF " + x
        ag = lambda x: "AG " + x
        ax = lambda x: "AX " + x
        af = lambda x: "AF " + x
        unary_ops = [paren, neg, eg, ex, ef, ag, ax, af]

    if not binary_ops:
        # set binary_ops to CTL ops
        cand = lambda x, y: x + " & " + y
        cor = lambda x, y: x + " | " + y
        # cxor = lambda x, y: x + " xor " + y
        # cxnor = lambda x, y: x + " xnor " + y
        cimp = lambda x, y: x + " -> " + y
        # ceq = lambda x, y: x + " <-> " + y
        eu = lambda x, y: "E[" + x + " U " + y + "]"
        au = lambda x, y: "A[" + x + " U " + y + "]"
        # binary_ops = [cand, cor, cxor, cxnor, cimp, ceq, eu, au]
        binary_ops = [cand, cor, cimp, eu, au]

    db = []
    vdb = []
    for l in trange(l_max):
        if l == 0:
            db.append(atoms)
            # vdb.append([error(atom, auto, horizon) for atom in atoms])
        else:
            db_l = []
            vdb_l = []
            # apply unary ops
            for unary_op in unary_ops:
                for arg in db[l - 1]:
                    phi = unary_op(arg)
                    db_l.append(phi)
                    # vdb_l.append(error(phi, auto, horizon))

            if l > 1:
                # apply binary ops
                for binary_op in binary_ops:
                    for i in range(l - 2):
                        for lhs in db[i]:
                            for rhs in db[l - i - 1]:
                                phi = binary_op(lhs, rhs)
                                db_l.append(phi)
                                # vdb_l.append(error(phi, auto, horizon))

            # add the new formulas to the database
            db.append(db_l)
            # vdb.append(vdb_l)
    # check error of all the formulas at once, this way we generate fragments once
    vdb = error(db, auto, horizon, condition)
    return db, vdb


# TODO: check error at the end? Checking each formula as it's created is super expensive.
# TODO: break error into chunks so I *can* check error for just one formula
# ^ inner function should take the fragments as parameters
def error(db, auto, horizon, condition="TRUE"):
    # get all the fragments
    choice_automata, frags = get_choice_automata(auto, horizon, condition, True)
    # get just the optimal action fragments
    optimal = choose_optimal_automata(choice_automata)
    optimal_actions = [opt[0] for opt in optimal]
    vdb = []

    # set up progress bar
    size = 0
    for s in db:
        size += len(s)

    with tqdm(total=size) as pbar:
        for l in db:
            vl = []
            for phi in l:
                tp = 0
                fn = 0
                tn = 0
                fp = 0
                for entry in optimal:
                    if not entry[0] in optimal_actions:
                        optimal_actions.append(entry[0])
                    check = entry[1].checkCTL("temp.smv", phi)
                    if check:
                        tp += 1
                    else:
                        fn += 1

                for i, frag in enumerate(frags):
                    if not frag[0] in optimal_actions:
                        check = frag.checkCTL("temp.smv", phi)
                        if check:
                            fp += 1
                        else:
                            tn += 1
                mcr = (fp + fn) / (tp + tn + fp + fn)
                vl.append(mcr)
                pbar.update(1)
            vdb.append(vl)
    # TODO: experiments with different error scores?

    return vdb


if __name__ == "__main__":
    pass
