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

from model_check import Automaton, checkConditional, checkObligation
from tqdm import trange


def enum(auto, l_max, atoms, unary_ops=[], binary_ops=[]):
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
        cxor = lambda x, y: x + " xor " + y
        cxnor = lambda x, y: x + " xnor " + y
        cimp = lambda x, y: x + " -> " + y
        ceq = lambda x, y: x + " <-> " + y
        eu = lambda x, y: "E[" + x + " U " + y + "]"
        au = lambda x, y: "A[" + x + " U " + y + "]"
        binary_ops = [cand, cor, cxor, cxnor, cimp, ceq, eu, au]

    db = []
    vdb = []
    for l in trange(l_max):
        if l == 0:
            db.append(atoms)
            vdb.append([error(atom, auto) for atom in atoms])
        else:
            db_l = []
            vdb_l = []
            # apply unary ops
            for unary_op in unary_ops:
                for arg in db[l-1]:
                    phi = unary_op(arg)
                    db_l.append(phi)
                    vdb_l.append(error(phi, auto))

            if l > 1:
                # apply binary ops
                for binary_op in binary_ops:
                    for i in range(l-2):
                        for lhs in db[i]:
                            for rhs in db[l-i-1]:
                                phi = binary_op(lhs, rhs)
                                db_l.append(phi)
                                vdb_l.append(error(phi, auto))

            # add the new formulas to the database
            db.append(db_l)
            vdb.append(vdb_l)

    return db, vdb

# TODO: find the MCR of a formula
def error(phi, auto):
    return 0


if __name__ == "__main__":
    db, vdb = enum(None, 5, ['A', 'B', 'C', 'D', 'E'])
    print(db)
