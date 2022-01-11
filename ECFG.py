import re
import types
import os.path
import numpy as np
from collections import defaultdict
from boss.code.CFG.CFG import Grammar, choice
from nltk import ChartParser


class EnumerativeGrammar(Grammar):
    def __init__(self, *args, **kwargs):
        super(Grammar, self).__init__(*args, **kwargs)
        # store some extra quantities needed later
        non_unique_nonterminals = [str(prod.lhs()) for prod in self.productions()]
        self.nonterminals = list(set(non_unique_nonterminals))
        self.terminals = list(
            set([str(individual) for prod in self.productions() for individual in prod.rhs()]) - set(self.nonterminals))
        # collect nonterminals that are worth swapping when doing genetic operations
        # (i.e not those with a single production that leads to a terminal)
        self.swappable_nonterminals = list(
            set([i for i in non_unique_nonterminals if non_unique_nonterminals.count(i) > 1]))

        self._sample_space = []
        self._space_size = 0
        self._space_file = None
        self.sample_length = 8

    def set_file(self, fname):
        self._space_file = fname

    def _convergent_sampler(self, cfactor, symbol=None, pcount=defaultdict(int)):
        # sampler that down-weights the probability of selcting the same production many times
        # ensuring that the sampled trees are not 'too' long (size to be controlled by cfactor)
        #
        # :pcount: storage for the productions used in the current branch
        # TODO: fix this so it actually converges
        # TODO: make an attribute that sets the maximum depth of a formula.
        # init the sequence
        tree = "(" + str(symbol)
        # init counter of tree depth and number of production rules
        depth, num_prod = 1, 1
        sym_list = [symbol]
        pcounts = [pcount]
        pcount_dir = {0: 0}
        while not all_strings(sym_list):
            new_syms = []
            new_pcounts = []
            new_pcount_dir = {}
            for i, sym in enumerate(sym_list):
                if isinstance(sym, str):
                    continue
                # collect possible productions from the current symbol
                productions = self.productions(lhs=symbol)
                # init sampling weights
                weights = []
                # calc weights for the possible productions
                # pcount_idx = pcount_dir[i]
                # pcount = pcounts[pcount_idx]
                for prod in productions:
                    if prod in pcount:
                        # if production already occurred in branch then downweight
                        weights.append(cfactor ** (pcount[prod]))
                    else:
                        # otherwise, set to be 1
                        weights.append(1.0)
                # normalize weights to get probabilities
                norm = sum(weights)
                probs = [weight / norm for weight in weights]
                # sample
                production = choice(productions, probs)
                # update counts
                pcount[production] += 1
                # pcounts[pcount_idx] = pcount
                sym_list[i] = production.rhs()
                for psym in production.rhs():
                    if isinstance(psym, str):
                        # if terminal then add string to sequence
                        tree = tree + " " + psym
                    else:
                        # otherwise, keep generating the sequence
                        num_prod += 1
                # update depth
                depth += 1
            # update counts and flatten list
            for j, sym_set in enumerate(sym_list):
                if isinstance(sym_set, str):
                    new_syms.append(sym_set)
                else:
                    for new_sym in sym_set:
                        old_pcount_idx = pcount_dir[j]
                        old_pcount = pcounts[old_pcount_idx]
                        new_syms.append(new_sym)
                        new_pcounts.append(old_pcount)
                        new_pcount_dir[len(new_syms)-1] = len(new_pcounts)-1
            pcounts = new_pcounts
            pcount_dir = new_pcount_dir
            sym_list = new_syms
        parser = ChartParser(self)
        parsed_sym = parser.parse(sym_list)
        if isinstance(parsed_sym, types.GeneratorType):
            for tree in parsed_sym:
                tree = re.sub(r"\n\s*", " ", str(tree))
                break
        else:
            tree = re.sub(r"\n\s*", " ", str(parsed_sym))
        return tree, depth, num_prod

    def _make_space(self, n):
        # TODO: save strings to file so we can use them later, and offload RAM usage to disk?
        if self._space_file:
            if os.path.isfile(self._space_file):
                # this space already exists
                return
            with open(self._space_file, 'w') as f:
                enum_parser = ChartParser(self)
                iter = self._generate(depth=n)
                for phi in iter:
                    self._space_size += 1
                    parsed_phi = enum_parser.parse(phi)
                    if isinstance(parsed_phi, types.GeneratorType):
                        for psi in parsed_phi:
                            f.write(re.sub(r"\n\s*", " ", str(psi)) + "\n")
                            break
                    else:
                        f.write(re.sub(r"\n\s*", " ", str(phi)) + "\n")
        else:
            enum_parser = ChartParser(self)
            iter = self._generate(depth=n)
            for phi in iter:
                self._space_size += 1
                parsed_phi = enum_parser.parse(phi)
                if isinstance(parsed_phi, types.GeneratorType):
                    for psi in parsed_phi:
                        self._sample_space.append(re.sub(r"\n\s*", " ", str(psi)))
                        break
                else:
                    self._sample_space.append(re.sub(r"\n\s*", " ", str(phi)))
            self._sample_space = np.array(self._sample_space)


def get_lines(fp, line_numbers):
    return (x for i, x in enumerate(fp) if i in line_numbers)


def all_strings(symbol_list):
    is_str = [isinstance(sym, str) for sym in symbol_list]
    return np.all(is_str)
