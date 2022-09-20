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

import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from joblib import Parallel, delayed
from nltk import ChartParser

from boss.code.CFG.CFG import Grammar

from bayes_opt import deformat


def num_prod_rules(grammar, csvfile):
    datareader = csv.reader(csvfile)
    parser = ChartParser(grammar)
    for row in datareader:
        phi = row[1]
        phi_tokens = re.split(r"\s|([()!])", phi)
        phi_tokens = [deformat(c) for c in phi_tokens if c]
        phi_tree = parser.parse(phi_tokens)
        # TODO: work with that tree


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
    prop_str = " | ".join(['"' + proposition + '"' for proposition in propositions])
    grammar_str = grammar_str + prop_str
    ltl_grammar = Grammar.fromstring(grammar_str)

