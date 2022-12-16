"""
Created September 2022

@author: Colin

Analyze sets of data for training learning algorithms at tasks in formal methods.
Right now it's limited to LTL.

Collect statistics from a dataset of LTL formulas.
    Number of operators
    Number of atomic propositions
    Number of production rules
    Number of a specific operator/AP
"""

import multiprocessing
import csv
import re

from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm
from joblib import Parallel, delayed
from nltk import ChartParser

from boss.code.CFG.CFG import Grammar

from bayes_opt import deformat


def num_prod_rules(formula_trees):
    prods_parse = []
    for parse_tree in formula_trees:
        prods_parse.append(len(parse_tree.productions()))
    return min(prods_parse)


def get_operators(formula_tokens, non_ops):
    op_tokens = [token for token in formula_tokens if token not in non_ops]
    return op_tokens


def get_props(formula_tokens, props):
    prop_tokens = [token for token in formula_tokens if token in props]
    return prop_tokens


def plot_count(count_dict, ylabel, xlabel, save_path):
    plt.figure(figsize=(5, 3))
    plt.bar(count_dict.keys(), count_dict.values())
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(save_path+".pdf", format="pdf")


def plot_counts(count_dicts, names, ylabel, xlabel, save_path):
    plt.figure(figsize=(5, 3))
    for i, count_dict in enumerate(count_dicts):
        plt.bar(count_dict.keys(), count_dict.values(), alpha=0.5, label=names[i])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+".pdf", format="pdf")


def count_to_list(count_dict):
    count_list = []
    for key in count_dict.keys():
        count_list.extend([key] * count_dict[key])
    return count_list


if __name__ == "__main__":
    grammar_str = """
        S -> NOT LB S RB | LB S RB | GLB S | NXT S | FTR S | S AND S | S XOR S | S OR S | S IMPLIES S | S UNTIL S | T
        NOT -> "!"
        GLB -> "G"
        NXT  -> "X"
        FTR -> "F"
        LB -> "lb"
        RB -> "rb"
        AND -> "&"
        XOR -> "xor"
        OR -> "|"
        IMPLIES -> "->"
        UNTIL -> "U"
        T -> """
    propositions = ['p', 'q', 'r', 's', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
    non_operators = propositions + ['lb', 'rb']
    prop_str = " | ".join(['"' + proposition + '"' for proposition in propositions])
    grammar_str = grammar_str + prop_str
    ltl_grammar = Grammar.fromstring(grammar_str)
    ltl_parser = ChartParser(ltl_grammar)
    csvpath = f"/home/colin/Documents/GitHub/deep_verify/data/deep_verify_eval_data.csv"
    each_terms_base = {"lb": 0, "rb": 0, "G": 0, "F": 0, "X": 0, "U": 0, "!": 0, "&": 0, "|": 0, "->": 0, "xor": 0,
                       "a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0, "g": 0, "p": 0, "q": 0, "r": 0, "s": 0}

    terminals_count = defaultdict(lambda: 0, {})
    prod_rules_count = defaultdict(lambda: 0, {})
    operators_count = defaultdict(lambda: 0, {})
    props_count = defaultdict(lambda: 0, {})
    each_terms_count = Counter(each_terms_base)

    sat_terminals_count = defaultdict(lambda: 0, {})
    sat_prod_rules_count = defaultdict(lambda: 0, {})
    sat_operators_count = defaultdict(lambda: 0, {})
    sat_props_count = defaultdict(lambda: 0, {})
    sat_each_terms_count = Counter(each_terms_base)

    unsat_terminals_count = defaultdict(lambda: 0, {})
    unsat_prod_rules_count = defaultdict(lambda: 0, {})
    unsat_operators_count = defaultdict(lambda: 0, {})
    unsat_props_count = defaultdict(lambda: 0, {})
    unsat_each_terms_count = Counter(each_terms_base)

    with open(csvpath, 'r', newline='') as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in tqdm(datareader):
            phi = row["phi"]
            label = row["label"]
            phi_tokens = re.split(r"\s|([()!])", phi)
            phi_tokens = [deformat(c) for c in phi_tokens if c]
            phi_trees = [tree for tree in ltl_parser.parse(phi_tokens)]
            # count the number of terminals in this formula
            phi_term_count = len(phi_tokens)
            terminals_count[phi_term_count] += 1
            # get the number of production rules for this formula
            if phi_trees:
                phi_prod_rules_count = num_prod_rules(phi_trees)
                prod_rules_count[phi_prod_rules_count] += 1
            # get the number of operators
            phi_operators = get_operators(phi_tokens, non_operators)
            phi_operators_count = len(phi_operators)
            operators_count[phi_operators_count] += 1
            # get the number of atomic propositions
            phi_props = get_props(phi_tokens, propositions)
            phi_props_count = len(phi_props)
            props_count[phi_props_count] += 1
            # count the number of each terminal
            phi_terms_count = Counter(phi_tokens)
            each_terms_count.update(phi_terms_count)
            if label == "1":
                sat_terminals_count[phi_term_count] += 1
                if phi_trees:
                    sat_prod_rules_count[phi_prod_rules_count] += 1
                sat_operators_count[phi_operators_count] += 1
                sat_props_count[phi_props_count] += 1
                sat_each_terms_count.update(phi_terms_count)
            else:
                unsat_terminals_count[phi_term_count] += 1
                if phi_trees:
                    unsat_prod_rules_count[phi_prod_rules_count] += 1
                unsat_operators_count[phi_operators_count] += 1
                unsat_props_count[phi_props_count] += 1
                unsat_each_terms_count.update(phi_terms_count)

    # plot data
    fig_dir = f"data/"
    dict_names = ['Satisfied', 'Unsatisfied']
    plot_count(prod_rules_count, "Number of formulas", "Number of production rules", fig_dir + "prod_rules_count")
    plot_counts([sat_prod_rules_count, unsat_prod_rules_count], dict_names, "Number of formulas",
                "Number of production rules", fig_dir + "compare_prod_rules_count")
    plot_count(terminals_count, "Number of formulas", "Formula length", fig_dir + "terminals_count")
    plot_counts([sat_terminals_count, unsat_terminals_count], dict_names, "Number of formulas", "Formula length",
                fig_dir + "compare_terminals_count")
    plot_count(operators_count, "Number of formulas", "Number of operators", fig_dir + "operators_count")
    plot_counts([sat_operators_count, unsat_operators_count], dict_names, "Number of formulas", "Number of operators",
                fig_dir + "compare_operators_count")
    plot_count(props_count, "Number of formulas", "Number of propositions", fig_dir + "propositions_count")
    plot_counts([sat_props_count, unsat_props_count], dict_names, "Number of formulas", "Number of propositions",
                fig_dir + "compare_propositions_count")
    plot_count(each_terms_count, "Number of each terminal", "Terminal symbol", fig_dir + "each_terms_count")
    plot_counts([sat_each_terms_count, unsat_each_terms_count], dict_names, "Number of each terminal",
                "Terminal symbol", fig_dir + "compare_each_terms_count")

    # calculate summary statistics
    prod_rules_count_list = count_to_list(prod_rules_count)
    sat_prod_rules_count_list = count_to_list(sat_prod_rules_count)
    unsat_prod_rules_count_list = count_to_list(unsat_prod_rules_count)
    print("Mean number of production rules per formula: " + str(np.mean(prod_rules_count_list)) +
          "; STD:" + str(np.std(prod_rules_count_list)))
    print("Mean number of production rules per satisfied formula: " + str(np.mean(sat_prod_rules_count_list)) +
          "; STD:" + str(np.std(sat_prod_rules_count_list)))
    print("Mean number of production rules per unsatisfied formula: " + str(np.mean(unsat_prod_rules_count_list)) +
          "; STD:" + str(np.std(unsat_prod_rules_count_list)))

    terminals_count_list = count_to_list(terminals_count)
    sat_terminals_count_list = count_to_list(sat_terminals_count)
    unsat_terminals_count_list = count_to_list(unsat_terminals_count)
    print("Mean number of terminals per formula: " + str(np.mean(terminals_count_list)) +
          "; STD:" + str(np.std(terminals_count_list)))
    print("Mean number of terminals per satisfied formula: " + str(np.mean(sat_terminals_count_list)) +
          "; STD:" + str(np.std(sat_terminals_count_list)))
    print("Mean number of terminals per unsatisfied formula: " + str(np.mean(unsat_terminals_count_list)) +
          "; STD:" + str(np.std(unsat_terminals_count_list)))

    operators_count_list = count_to_list(operators_count)
    sat_operators_count_list = count_to_list(sat_operators_count)
    unsat_operators_count_list = count_to_list(unsat_operators_count)
    print("Mean number of operators per formula: " + str(np.mean(operators_count_list)) +
          "; STD:" + str(np.std(operators_count_list)))
    print("Mean number of operators per satisfied formula: " + str(np.mean(sat_operators_count_list)) +
          "; STD:" + str(np.std(sat_operators_count_list)))
    print("Mean number of operators per unsatisfied formula: " + str(np.mean(unsat_operators_count_list)) +
          "; STD:" + str(np.std(unsat_operators_count_list)))

    props_count_list = count_to_list(props_count)
    sat_props_count_list = count_to_list(sat_props_count)
    unsat_props_count_list = count_to_list(unsat_props_count)
    print("Mean number of atomic propositions per formula: " + str(np.mean(props_count_list)) +
          "; STD:" + str(np.std(props_count_list)))
    print("Mean number of atomic propositions per satisfied formula: " + str(np.mean(sat_props_count_list)) +
          "; STD:" + str(np.std(sat_props_count_list)))
    print("Mean number of atomic propositions per unsatisfied formula: " + str(np.mean(unsat_props_count_list)) +
          "; STD:" + str(np.std(unsat_props_count_list)))

    print("done")
