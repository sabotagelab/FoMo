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
import csv
import random
import os
import math

from collections import defaultdict

import matplotlib.pyplot as plt
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from tqdm import trange, tqdm
from joblib import Parallel, delayed, parallel_backend

from emukit.core import ParameterSpace
from emukit.core.initial_designs import RandomDesign

from boss.code.CFG.CFG import Grammar
from boss.code.parameters.cfg_parameter import CFGParameter
from boss.code.parameters.cfg_parameter import unparse

from random_weighted_automaton import generateGraph, generateHistory
from bayes_opt import reformat
from model_check import Automaton

spot_imported = True
try:
    import spot
except ImportError:
    spot_imported = False

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


def generate_trace(graph, trace_len):
    return generateHistory(graph, graph.es["weight"], trace_len, 0)[2]


def generate_formula(automaton, grammar, max_formula_length, satisfying=True, smv_file="temp.smv"):
    # grammar should be a Grammar object if spot_imported is False, and a string of propositions otherwise.
    # TODO: consider replacing sampling with uniform random
    valid_formula = None
    invalid_formulas = []
    formula_size = random.randint(1, max_formula_length)
    if not spot_imported:
        cfactor = max(math.exp(-100/formula_size), 10**-1)
        while not valid_formula:
            candidate_formula = grammar.sampler_restricted(1, formula_size, cfactor, max_formula_length)
            candidate_formula = unparse(candidate_formula)
            if candidate_formula not in invalid_formulas:
                formula = reformat(candidate_formula[0])
                validity = automaton.checkLTL(smv_file, formula)
                if (validity and satisfying) or (not validity and not satisfying):
                    valid_formula = reformat(candidate_formula[0])
                else:
                    invalid_formulas.append(candidate_formula)
    else:
        seed = int.from_bytes(os.urandom(2), byteorder="big")
        ltl_properties = 'false=0,true=0,equiv=0,R=0,W=0,M=0'
        formula_generator = spot.randltl(grammar, seed=seed, tree_size=(formula_size, max_formula_length), simplify=0,
                                         ltl_properties=ltl_properties)
        while not valid_formula:
            try:
                candidate_formula = str(next(formula_generator))
            except StopIteration:
                seed = int.from_bytes(os.urandom(2), byteorder="big")
                formula_generator = spot.randltl(grammar, seed=seed, tree_size=(formula_size, max_formula_length),
                                                 simplify=0, ltl_properties=ltl_properties)
                candidate_formula = str(next(formula_generator))

            # formula_is_short = len(candidate_formula) <= 3
            formula_is_short = len(candidate_formula.replace(" ", "")) <= formula_size
            # formula_shouldnt_be_short = formula_size > 3
            # formula_is_wrong_length = len(candidate_formula) != formula_size
            # if formula_is_short and formula_shouldnt_be_short:
            # if formula_is_wrong_length:
            # check desired properties, sometimes equiv, R, W, and M show up
            equiv = "<->" in candidate_formula
            release = "R" in candidate_formula
            weak_release = "M" in candidate_formula
            weak_until = "W" in candidate_formula
            prop_violation = equiv or release or weak_release or weak_until
            if formula_is_short or prop_violation:
                # try again
                continue
            else:
                # add space after temporal operators
                candidate_formula_as_list = list(candidate_formula)
                for i, char in enumerate(candidate_formula_as_list):
                    if char.isupper():
                        if candidate_formula_as_list[i+1].isalpha():
                            candidate_formula_as_list.insert(i+1, ' ')
                candidate_formula = "".join(candidate_formula_as_list)
                validity = automaton.checkLTL(smv_file, candidate_formula)
                if (validity and satisfying) or (not validity and not satisfying):
                    valid_formula = candidate_formula
    return valid_formula


def generate_mfl_entry(props, grammar, auto_size, auto_connect, max_symbols, formula_length, model_file):
    coin = [True, False]
    auto = generate_automaton(auto_size, auto_connect, symbols=props, max_symbols=max_symbols)
    model = auto.convertToNuXmv(model_file, x=None, lang="LTL", return_string=True)
    formula = ""
    label = ""
    flip = random.choice(coin)
    if flip:
        label = "1"
        formula = generate_formula(auto, grammar, formula_length, smv_file=model_file)
    else:
        label = "0"
        formula = generate_formula(auto, grammar, formula_length, False, smv_file=model_file)
    # delete model file (model has been checked, and content is in the model string
    os.remove(model_file)
    return [model, formula, label]


if __name__ == "__main__":
    propositions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'p', 'q', 'r', 's']
    model_propositions = []
    for prop in propositions:
        model_propositions.append(str(prop))

    prop_str = " | ".join(['"' + proposition + '"' for proposition in model_propositions])
    grammar_str = grammar_str + prop_str
    if spot_imported:
        gram = propositions
    else:
        gram = Grammar.fromstring(grammar_str)
    # test_auto = generate_automaton(20, 0.3, symbols=propositions, max_symbols=4)
    # test_formula = generate_formula(test_auto, gram, 20)

    cluster = SLURMCluster(
        queue='eecs',
        project='eecs',
        cores=10,
        memory='48GB',
        shebang="#!/bin/bash",
        n_workers=40,
        walltime='3-00:00:00',
        job_extra=['-o generate_data.out', '-e generate_data.err', '--mail-user=sheablyc@oregonstate.edu', '--mail-type=ALL'],
    )
    print(cluster.dashboard_link)
    client = Client(cluster.scheduler_address)

    data_size = 2**20
    data_file = "data/deep_verify_train_data.csv"
    # entries = []
    # for _ in trange(data_size):
    #     entry = generate_mfl_entry(propositions, gram, 20, 0.3, 4, 20, "temp.smv")
    #     entries.append(entry)
    num_cores = multiprocessing.cpu_count()
    # here I use a new model file for each job to avoid race conditions. There's probably a better way of doing this.
    with parallel_backend('dask', wait_for_workers_timeout=120):
        entries = Parallel()(delayed(generate_mfl_entry)(propositions, gram, 20, 0.3, 11, 20, "model_files/temp"+str(i)+".smv") for i in trange(data_size))

    with open(data_file, 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        headwriter = csv.DictWriter(csvfile, fieldnames=["sys", "phi", "label"])
        headwriter.writeheader()
        datawriter.writerows(entries)

    count_dict = {}
    count_dict = defaultdict(lambda: 0, count_dict)
    with open(data_file, 'r', newline='') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            phi = row[1].replace(" ", "")
            phi_len = len(phi)
            if count_dict[phi_len]:
                count_dict[phi_len] += 1
            else:
                count_dict[phi_len] = 1

    plt.bar(count_dict.keys(), count_dict.values())
    plt.ylabel("Number of formulas")
    plt.xlabel("Formula length")
    plt.savefig(f"data/formula_distribution")
