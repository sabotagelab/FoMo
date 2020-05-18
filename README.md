# autonomous-learning

Repository related to learning behaviors exhibited by autonomous systems.

random_weighted_automaton.py contains provides a function to generate a weighted directed graph via Erdos-Renyi, and another that uses random walks to generate histories. The former returns a graph object, and the latter returns a list of 2-tuples (a list of vertices visited, the accumulated value of the history).

This code relies on 3 packages: igraph, tqdm, and numpy.
I'm sure you already have numpy; igraph (https://igraph.org/python/) is used for graph operations, and tqdm (https://pypi.org/project/tqdm/) just adds a CLI progress bar.

