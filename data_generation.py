"""
Created May 2022

@author: Colin

Generate sets of data for training learning algorithms at tasks in formal methods.
Right now it's limited to LTL.

Generate system models as nondeterministic, finite-state automata (NFA).
Generate traces from a NFA.
Generate satisfying formulas for a NFA.
Generate unsatisfying formulas for a NFA.
"""

import multiprocessing

from tqdm import trange, tqdm
from joblib import Parallel, delayed

from emukit.core import ParameterSpace
from emukit.core.initial_designs import RandomDesign

from boss.code.CFG.CFG import Grammar
from boss.code.parameters.cfg_parameter import CFGParameter
from boss.code.parameters.cfg_parameter import unparse

from random_weighted_automaton import generateGraph, generateHistory
from bayes_opt import reformat
from model_check import Automaton

grammar_str = """
    S -> NOT LB S RB | LB S RB | GLB S | NXT S | FTR S | S AND S | S OR S | S IMPLIES S | S UNTIL S | T
    NOT -> "!"
    GLB -> "G"
    NXT  -> "X"
    FTR -> "F"
    LB -> "lb"
    RB -> "rb"
    AND -> "&"
    OR -> "|"
    IMPLIES -> "->"
    UNTIL -> "U"
    T -> """


def generate_automaton(num_vertices, prob_edges, symbols=None, max_symbols=1):
    # get a random graph that's automaton shaped
    graph = generateGraph(num_vertices, prob_edges, 0, 1, symbols, max_symbols)
    return Automaton(graph, 0, symbols)


def generate_traces(graph, num_traces, trace_len):
    return generateHistory(graph, graph.es["weight"], trace_len, 0)


def generate_formula(automaton, grammar, max_formula_length, satisfying=True):
    # TODO: consider replacing sampling with uniform random
    space = ParameterSpace([CFGParameter("grammar", grammar, max_length=max_formula_length, min_length=0)])
    valid_formula = None
    invalid_formulas = []
    random_design = RandomDesign(space)
    while not valid_formula:
        candidate_formula = random_design.get_samples(1)
        candidate_formula = unparse(candidate_formula)
        if candidate_formula not in invalid_formulas:
            # TODO: check the indexing here
            formula = reformat(candidate_formula[0][0])
            if not satisfying:
                formula = "! ( " + formula + " )"
            validity = automaton.checkLTL("temp.smv", formula)
            if validity:
                valid_formula = candidate_formula[0][0]
            else:
                invalid_formulas.append(candidate_formula)
    return reformat(valid_formula)


if __name__ == "__main__":
    propositions = ['p', 'q', 'r', 's']
    model_propositions = []
    for prop in propositions:
        model_propositions.append(str(prop))

    prop_str = " | ".join(['"' + proposition + '"' for proposition in model_propositions])
    grammar_str = grammar_str + prop_str
    gram = Grammar.fromstring(grammar_str)
    auto = generate_automaton(20, 0.3, symbols=propositions, max_symbols=4)
    traces = generate_traces(auto.graph, 10, 10)
    sat_formula = generate_formula(auto, gram, 20)
    unsat_formula = generate_formula(auto, gram, 20, False)
    print(traces)
    print(sat_formula)
    print(unsat_formula)
