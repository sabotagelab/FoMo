"""
Created September 2020

@author: Colin
"""

import subprocess
import tqdm
import numpy as np
from copy import deepcopy
from igraph import *
from random_weighted_automaton import *

def learnLabels(graph, strings, q0=0):
    labels = {vert.index: None for vert in graph.vs}
    labels[q0] = strings[0][0]
    data = []
    pos_labels = [labels]
    for string in tqdm(strings):
        # account for new labelings after a string is read
        for labeling in pos_labels:
            data.append([labeling, graph.neighbors(q0, mode=OUT)])
            if labeling[q0] != string[0]:
                raise ValueError('Strings do not all begin with same symbol',
                                 string)
        for symbol in string[1:]:
            new_data = []
            for entry in data:
                curr_labels = entry[0]
                curr_states = entry[1]
                for state in curr_states:
                    if not curr_labels[state]:
                        new_labels = deepcopy(curr_labels)
                        new_labels[state] = symbol
                        new_data.append([new_labels,
                                         graph.neighbors(state, mode=OUT)])
                        # if curr_labels in pos_labels:
                        #     pos_labels.remove(curr_labels)
                        # pos_labels.append(new_labels)
                    elif curr_labels[state] != symbol:
                        continue
                    else:
                        new_data.append([curr_labels,
                                         graph.neighbors(state, mode=OUT)])
            data = new_data
        new_pos_labels = []
        for entry in data:
            if entry[0] not in new_pos_labels:
                new_pos_labels.append(entry[0])
        pos_labels = new_pos_labels
    return pos_labels

if __name__ == "__main__":
    graph = Graph(n=5,
                  edges=[(0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (3, 1), (3, 2),
                         (3, 4), (4, 2)], directed=True)
    strings = ["acaacba", "acbbaca", "acbacba", "acbbbaa"]
    # strings = ["acaacba"]
    labels = learnLabels(graph, strings)