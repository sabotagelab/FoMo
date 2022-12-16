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
import re

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


def generate_automaton(num_vertices, prob_edges, symbols=None, max_symbols=1,
                       smv_file="temp.smv", formula=None, satisfy=True, min_weight=0, max_weight=1):
    # get a random graph that's automaton shaped
    graph = generateGraph(num_vertices, prob_edges, min_weight, max_weight, symbols, max_symbols)
    if formula:
        # reject the automaton if it doesn't satisfy the formula
        auto = Automaton(graph, 0, symbols)
        valid_auto = None
        while not valid_auto:
            validity = auto.checkLTL(smv_file, formula)
            if (validity and satisfy) or (not validity and not satisfy):
                valid_auto = auto
            else:
                graph = generateGraph(num_vertices, prob_edges, 0, 1, symbols, max_symbols)
                auto = Automaton(graph, 0, symbols)
        return valid_auto

    return Automaton(graph, 0, symbols)


def generate_auto_mat(num_vertices, prob_edges, symbols=None, max_symbols=1,
                      smv_file="temp.smv", formula=None, satisfy=True, min_weight=0, max_weight=1):
    auto = generate_automaton(num_vertices, prob_edges, symbols, max_symbols, smv_file, formula, satisfy, min_weight,
                              max_weight)
    return auto.convertToMatrix()


def generate_trace(graph, trace_len):
    return generateHistory(graph, graph.es["weight"], trace_len, 0)[2]


def generate_formula(automaton, grammar, max_formula_length, satisfying=True, smv_file="temp.smv"):
    # grammar should be a Grammar object if spot_imported is False, and a string of propositions otherwise.
    # TODO: consider replacing sampling with uniform random
    # TODO: refactor so automaton is optional
    valid_formula = None
    invalid_formulas = []
    formula_size = random.randint(1, max_formula_length)
    if not spot_imported:
        cfactor = max(math.exp(-100 / formula_size), 10 ** -1)
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

            bad_formula = not check_formula(candidate_formula, formula_size)
            if bad_formula:
                # try again
                continue
            else:
                candidate_formula = expand_spot_formula(candidate_formula)
                validity = automaton.checkLTL(smv_file, candidate_formula)
                if (validity and satisfying) or (not validity and not satisfying):
                    valid_formula = candidate_formula
    return valid_formula


# TODO: use rewriting rules of LTL to generate equivalent formulas instead of using random generation.
def generate_equiv_formulas(formula, number, grammar, formula_size, max_formula_length):
    seed = int.from_bytes(os.urandom(2), byteorder="big")
    ltl_properties = 'false=0,true=0,equiv=0,R=0,W=0,M=0'
    formula_generator = spot.randltl(grammar, seed=seed, tree_size=(formula_size, max_formula_length), simplify=0,
                                     ltl_properties=ltl_properties)
    checker = spot.language_containment_checker()
    equiv_formulas = []
    while len(equiv_formulas) < number:
        try:
            candidate_formula = str(next(formula_generator))
        except StopIteration:
            seed = int.from_bytes(os.urandom(2), byteorder="big")
            formula_generator = spot.randltl(grammar, seed=seed, tree_size=(formula_size, max_formula_length),
                                             simplify=0, ltl_properties=ltl_properties)
            candidate_formula = str(next(formula_generator))
        bad_formula = not check_formula(candidate_formula, formula_size)
        if bad_formula:
            # try again
            continue
        elif checker.equal(formula, candidate_formula):
            equiv_formulas.append(candidate_formula)
    return equiv_formulas


def expand_spot_formula(candidate_formula):
    # add space after temporal operators
    candidate_formula_as_list = list(candidate_formula)
    for i, char in enumerate(candidate_formula_as_list):
        if char.isupper():
            if candidate_formula_as_list[i + 1].isalpha():
                candidate_formula_as_list.insert(i + 1, ' ')
    candidate_formula = "".join(candidate_formula_as_list)
    return candidate_formula


def check_formula(candidate_formula, formula_size):
    formula_is_short = len(candidate_formula.replace(" ", "")) <= formula_size
    equiv = "<->" in candidate_formula
    release = "R" in candidate_formula
    weak_release = "M" in candidate_formula
    weak_until = "W" in candidate_formula
    true_false = "1" in candidate_formula or "0" in candidate_formula
    prop_violation = equiv or release or weak_release or weak_until or true_false
    bad_formula = formula_is_short or prop_violation
    return not bad_formula


def generate_mfl_entry(props, grammar, auto_size, auto_connect, max_symbols, formula_length, model_file):
    coin = [True, False]
    auto = generate_automaton(auto_size, auto_connect, symbols=props, max_symbols=max_symbols)
    # model = auto.convertToNuXmv(model_file, x=None, lang="LTL", return_string=True)
    model = strip_auto_mat(auto.convertToMatrix())
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


def generate_contrastive_mfl_entry(props, grammar, auto_size, auto_connect, max_symbols, formula_length, model_file,
                                   negative_examples):
    auto = generate_automaton(auto_size, auto_connect, symbols=props, max_symbols=max_symbols)
    pos_model = strip_auto_mat(auto.convertToMatrix())
    formula = generate_formula(auto, grammar, formula_length, True, smv_file=model_file)
    negative_models = []
    while len(negative_models) < negative_examples:
        auto_mat = generate_automaton(auto_size, auto_connect, symbols=props, max_symbols=max_symbols,
                                      smv_file=model_file, formula=formula, satisfy=False).convertToMatrix()
        negative_models.append(strip_auto_mat(auto_mat))

    return [pos_model, formula] + negative_models


def strip_auto_mat(auto_mat):
    auto_str = re.sub(r'\W', '', str(auto_mat))
    return auto_str


def _init_default_cluster(cores, workers):
    cluster = SLURMCluster(
        queue='share',
        project='eecs',
        cores=cores,
        memory='48GB',
        shebang="#!/bin/bash",
        n_workers=workers,
        walltime='3-00:00:00',
        job_extra=['-o generate_data.out', '-e generate_data.err', '--mail-user=sheablyc@oregonstate.edu',
                   '--mail-type=ALL', '--nodelist=cn-a-7'],
    )
    return cluster


def _hpc_data_gen(propositions, grammar, n_entries, n_contrast, states, e_prob, max_labels, phi_len, cores, workers,
                  out_file):
    model_propositions = []
    for prop in propositions:
        model_propositions.append(str(prop))

    prop_str = " | ".join(['"' + proposition + '"' for proposition in model_propositions])
    grammar = grammar + prop_str
    if spot_imported:
        gram = propositions
    else:
        gram = Grammar.fromstring(grammar)
    # test_auto = generate_automaton(20, 0.3, symbols=propositions, max_symbols=4)
    # test_formula = generate_formula(test_auto, gram, 20)

    cluster = _init_default_cluster(cores, workers)
    print(cluster.dashboard_link)
    client = Client(cluster.scheduler_address)

    data_size = n_entries
    num_cores = multiprocessing.cpu_count()
    # here I use a new model file for each job to avoid race conditions. There's probably a better way of doing this.
    with parallel_backend('dask', wait_for_workers_timeout=120):
        if n_contrast:
            entries = Parallel()(
                delayed(generate_contrastive_mfl_entry)(propositions, gram, states, e_prob, max_labels, phi_len,
                                                        "model_files/temp" + str(i) + ".smv", 5)
                                                        for i in trange(data_size))
        else:
            entries = Parallel()(
                delayed(generate_mfl_entry)(propositions, gram, states, e_prob, max_labels, phi_len,
                                            "model_files/temp" + str(i) + ".smv") for i in trange(data_size))

    with open(out_file, 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        if n_contrast:
            headwriter = csv.DictWriter(csvfile,
                                        fieldnames=["sys", "phi"] + ["nex" + str(k) for k in range(n_contrast)])
        else:
            headwriter = csv.DictWriter(csvfile, fieldnames=["sys", "phi"])
        headwriter.writeheader()
        datawriter.writerows(entries)


def _time_gen_phi_from_sys(sys_auto, propositions, phi_len):
    # cluster = _init_default_cluster(1, 1)
    # client = Client(cluster.scheduler_address)
    with parallel_backend('dask', wait_for_workers_timeout=120):
        entries = Parallel()(
            delayed(generate_formula)(sys_auto, propositions, phi_len, True, "model_files/temp" + str(i) + ".smv")
            for i in trange(10000))
    with open("data/sat_formula_test.csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        headwriter = csv.DictWriter(csvfile, fieldnames=["formula"])
        headwriter.writeheader()
        datawriter.writerows(entries)


def _time_gen_trace_from_sys(sys_auto):
    # cluster = _init_default_cluster(1, 1)
    # client = Client(cluster.scheduler_address)
    with parallel_backend('dask', wait_for_workers_timeout=120):
        entries = Parallel()(delayed(generateHistory)(sys_auto.graph, sys_auto.graph.es["weights"], 1000, 0.9)
                             for _ in trange(10000))
    with open("data/trace_test.csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        headwriter = csv.DictWriter(csvfile, fieldnames=["transitions", "value", "labels"])
        headwriter.writeheader()
        datawriter.writerows(entries)


def _time_gen_sys_from_phi(phi_str, states, e_prob, propositions, max_symbols):
    # cluster = _init_default_cluster(1, 1)
    # client = Client(cluster.scheduler_address)
    with parallel_backend('dask', wait_for_workers_timeout=120):
        entries = Parallel()(
            delayed(generate_auto_mat)(states, e_prob, propositions, max_symbols, "model_files/temp" + str(i) + ".smv",
                                       phi_str) for i in trange(10000))
    with open("data/sat_model_test.csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        headwriter = csv.DictWriter(csvfile, fieldnames=["model"])
        headwriter.writeheader()
        datawriter.writerows(entries)


def _time_gen_equiv_phi(phi_str, propositions, formula_size):
    print("Generating equivalent formulas")
    entries = generate_equiv_formulas(phi_str, 10000, propositions, 0, formula_size)
    with open("data/equiv_formula_test.csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        headwriter = csv.DictWriter(csvfile, fieldnames=["formula"])
        headwriter.writeheader()
        datawriter.writerows(entries)


if __name__ == "__main__":
    # propos = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'p', 'q', 'r', 's']
    # propos = ['a', 'b']
    # file = "data/deep_verify_train_data_contrastive_small.csv"
    # _hpc_data_gen(propositions=propos, grammar=grammar_str, n_entries=10000, n_contrast=5, states=5, e_prob=0.8,
    #               max_labels=2, phi_len=4, cores=1, workers=1, out_file=file)

    adj_mat = [[0, 0, 0, 0, 1, 1, 0],
               [0, 0, 0, 0, 0, 1, 1],
               [1, 1, 0, 0, 1, 0, 0],
               [0, 1, 1, 0, 0, 0, 1],
               [0, 1, 0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 0]]
    ap = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    label = ['e', 'b', 'e, b, g', 'f, c, d', 'c', 'a, c, b', 'g, f']
    weight = [-3, 0, -3, 2, -3, 2, -2, 0, 2, -1, -1, -1, 0, 0, -2, -1, 0, 2]
    ex_auto = Automaton.from_matrix(adj_mat, ap, label, weight)
    ex_phi = "( ! f ) U c"
    assert ex_auto.checkLTL("model_files/temp.smv", ex_phi)
    cluster = _init_default_cluster(1, 1)
    client = Client(cluster.scheduler_address)
    # _time_gen_phi_from_sys(ex_auto, ap, 7)
    _time_gen_trace_from_sys(ex_auto)
    # _time_gen_sys_from_phi(ex_phi, 7, 0.3, ap, 3)
    _time_gen_equiv_phi(ex_phi, ap, 14)
