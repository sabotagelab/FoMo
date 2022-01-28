"""
Created December 2021

@author: Colin

Perform symbolic regression of a formula to meet an objective that is latent in a user.
Interact with the user to receive feedback on proposed formulas.
Model the user's objective over formulas as a Gaussian process with a string similarity kernel.
Find the string most likely to improve the performance of the model using an evolutionary model.
Once preferences have been suitably learned, search only for formulas that are satisfied by a given model.

Based on, and using code from "BOSS: Bayesian Optimization over String Spaces"
by Henry B. Moss, Daniel Beck, Javier Gonzalez, David S. Leslie, Paul Rayson
"""

import pickle
import time
import timeit
import numpy as np

from emukit.core import ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.loop import FixedIterationsStoppingCondition

from nltk import ChartParser

# import BOSS code
from boss.code.CFG.CFG import Grammar
from boss.code.parameters.cfg_parameter import CFGParameter
from boss.code.parameters.cfg_parameter import unparse
from boss.code.optimizers.GrammarGeneticAlgorithmAcquisitionOptimizer import GrammarGeneticProgrammingOptimizer
from boss.code.emukit_models.emukit_ssk_model import SSK_model

# import our code
from ECFG import EnumerativeGrammar
from GrammarAcquisitionOptimizer import GrammarGeneticValidityOptimizer
from model_check import get_choice_fragments, get_optimal_automata, get_choice_automata, choose_optimal_automata


# TODO: more parameters for more control over optimization
# TODO: make grammar strings/objects their own thing
def explore_formulas(auto, propositions=None, init_query_size=10, online_query_size=60):
    if not propositions:
        propositions = ['p', 'q', 'a', 'b']
    prop_str = " | ".join(['"' + proposition + '"' for proposition in propositions])

    # grammar_str = """
    #     S -> NOT S | LB S RB | EGLB S | ENXT S | EFTR S | AGBL S | ANXT S | AFTR S | S AND S | S OR S | S IMPLIES S | EB S UNTIL S CB | AB S UNTIL S CB | T
    #     NOT -> "!"
    #     EGLB -> "EG"
    #     ENXT  -> "EX"
    #     EFTR -> "EF"
    #     AGBL -> "AG"
    #     ANXT -> "AX"
    #     AFTR -> "AF"
    #     LB -> "lb"
    #     RB -> "rb"
    #     AND -> "&"
    #     OR -> "|"
    #     IMPLIES -> "->"
    #     UNTIL -> "U"
    #     EB -> "E["
    #     AB -> "A["
    #     CB -> "]"
    #     T -> """

    # grammar_str = """
    #         S -> S AND T | S OR T | S IMPLIES T | EB S UNTIL T CB | AB S UNTIL T CB | T
    #         T -> NOT S | LB S RB | EGLB S | ENXT S | EFTR S | AGBL S | ANXT S | AFTR S
    #         NOT -> "!"
    #         EGLB -> "EG"
    #         ENXT  -> "EX"
    #         EFTR -> "EF"
    #         AGBL -> "AG"
    #         ANXT -> "AX"
    #         AFTR -> "AF"
    #         LB -> "lb"
    #         RB -> "rb"
    #         AND -> "&"
    #         OR -> "|"
    #         IMPLIES -> "->"
    #         UNTIL -> "U"
    #         EB -> "E["
    #         AB -> "A["
    #         CB -> "]"
    #         T -> """

    # grammar_str = """
    #     S -> S ADD T | S TIMES T | S DIVIDE T | T
    #     T -> LB S RB | SIN S RB | EXP S RB
    #     ADD -> "+"
    #     TIMES -> "*"
    #     DIVIDE -> "/"
    #     LB -> "lb"
    #     RB -> "rb"
    #     SIN -> "sin"
    #     EXP -> "exp"
    #     T -> "x" | "1" | "2" | "3"
    # """

    # ctl_grammar = Grammar.fromstring("""
    #     S -> "!" S | "(" S ")" | Q"G" S | Q"X" S | Q"F" S | S "&" S | S "|" S | S "->" S | Q"[" S "U" S "]" | T
    #     Q -> "E" | "A"
    #     T -> "p" | "q"
    # """)

    # grammar_str = """
    #         S -> "!" S | "lb" S "rb" | "EG" S | "EX" S | "EF" S | "AG" S | "AX" S | "AF" S | S "&" S | S "|" S | S "->" S | "A[" S "U" S "]" | "E[" S "U" S "]" | T
    #         T -> """

    # grammar_str = """
    #         S -> S "&" R | S "|" R | S "->" R | "!" S | "lb" S "rb" | P | T
    #         P -> "EG" S | "EX" S | "EF" S | "AG" S | "AX" S | "AF" S | "A[" S "U" R "]" | "E[" S "U" R "]"
    #         R -> S
    #         T -> """

    # grammar_str = """
    #         S -> S AND R | S OR R | S IMPLIES R | NOT S | LB S RB | P | LB T RB
    #         P -> EGLB S | ENXT S | EFTR S | AGBL S | ANXT S | AFTR S | AB S UNTIL R CB | EB S UNTIL R CB
    #         R -> S
    #         NOT -> "!"
    #         AND -> "&"
    #         OR -> "|"
    #         IMPLIES -> "->"
    #         UNTIL -> "U"
    #         EGLB -> "EG"
    #         ENXT  -> "EX"
    #         EFTR -> "EF"
    #         AGBL -> "AG"
    #         ANXT -> "AX"
    #         AFTR -> "AF"
    #         LB -> "lb"
    #         RB -> "rb"
    #         AB -> "A["
    #         EB -> "E["
    #         CB -> "]"
    #         T -> """

    grammar_str = """
                N -> PROB COMP PCT OB Q CB | NOT PROB COMP PCT OB Q CB
                Q -> N | NOT LB Q RB | Q AND R | Q OR R | Q IMPLIES R | LB S RB | T
                R -> Q
                COMP -> LEQ | LT | GEQ | GT
                PCT -> ZERO | ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT | NINE | TEN
                S -> GLB O | NXT O | FTR O | O UNTIL V
                O -> S | Q
                V -> S | Q
                LEQ -> "<="
                LT -> "<"
                GEQ -> ">="
                GT -> ">"
                PROB -> "P"
                ZERO -> "0"
                ONE -> "0.1"
                TWO -> "0.2"
                THREE -> "0.3"
                FOUR -> "0.4"
                FIVE -> "0.5"
                SIX -> "0.6"
                SEVEN -> "0.7"
                EIGHT -> "0.8"
                NINE -> "0.9"
                TEN -> "1.0"
                NOT -> "!"
                AND -> "&"
                OR -> "|"
                IMPLIES -> "=>"
                UNTIL -> "U"
                GLB -> "G"
                NXT  -> "X"
                FTR -> "F"
                LB -> "lb"
                RB -> "rb"
                OB -> "["
                CB -> "]"
                T -> """

    grammar_str = grammar_str + prop_str

    ctl_grammar = Grammar.fromstring(grammar_str)
    # ctl_grammar = EnumerativeGrammar.fromstring(grammar_str)
    # ctl_grammar.set_file("11_state_grammar.txt")
    ctl_parser = ChartParser(ctl_grammar)

    # space = ParameterSpace([CFGParameter("grammar", ctl_grammar, max_length=10, min_length=3)])
    space = ParameterSpace([CFGParameter("grammar", ctl_grammar, max_length=16, min_length=0)])
    random_design = RandomDesign(space)
    # X_init = random_design.get_samples(5)
    # X_init_strings = unparse(X_init)

    # get initial formulas and interests
    # xs, ys = init_query(init_query_size, ctl_parser)
    # for testing, set up dummy inputs and outputs
    # xs = ["EG name=7", "EF AX name=7", "AG name=7", "AF EX name=7", "EX name=0", "EX name=1", "EF AG name=11",
    #       "name=0", "A[ ! name=11 U name=0 ]", "A[ name=5 U name=0 ]"]
    xs = ["P >= 0.4 [ ( F G name=7 ) ]", "P >= 0 [ ( F name=7 ) ]", "P <= 0.7 [ ( F name=11 ) ]",
          "! P <= 0.7 [ ( F name=10 ) => ( F name=11 ) ]", "P >= 1.0 [ name=0 ]", "P <= 0 [ name=1 ]",
          "P >= 0.7 [ ( X name=4 ) ]", "P <= 0 [ ( F name=11 ) & ( F name=7 ) ]",
          "P <= 1.0 [ ( ! ( name=11 ) U name=0 ) ]", "P <= 1.0 [ ( name=5 U name=0 ) ]"]
    xs = [deformat(x) for x in xs]
    ys = [1 - 78 / 100, 1 - 70 / 100, 1 - 90 / 100, 1 - 74 / 100, 1 - 0, 1 - 0, 1 - 58 / 100, 1 - 0, 1 - 0, 1 - 0]

    xs = np.array(xs).reshape(-1, 1)
    ys = np.array(ys).reshape(-1, 1)
    # build BO loop with SSK model
    # model = SSK_model(space, xs, ys, max_subsequence_length=5, n_restarts=3)
    # np.random.seed(123)
    # TODO: exorcise demons from the sampling method.
    model = SSK_model(space, xs, ys, max_subsequence_length=8, n_restarts=5, observation_noise=True)
    # load acquisition function
    # TODO: include some jitter in EI?
    expected_improvement = ExpectedImprovement(model, jitter=0.8)
    # use GA to optimize acquisition function
    # optimizer = GrammarGeneticProgrammingOptimizer(space,
    #                                                dynamic=True,
    #                                                population_size=100,
    #                                                tournament_prob=0.5,
    #                                                p_crossover=0.8,
    #                                                p_mutation=0.1)
    optimizer = GrammarGeneticProgrammingOptimizer(space,
                                                   dynamic=True,
                                                   population_size=100,
                                                   tournament_prob=0.25,
                                                   p_crossover=0.8,
                                                   p_mutation=0.2)
    # define BO loop
    bayesopt_loop_SSK = BayesianOptimizationLoop(model=model,
                                                 space=space,
                                                 acquisition=expected_improvement,
                                                 acquisition_optimizer=optimizer)

    # add loop summary
    bayesopt_loop_SSK.iteration_end_event.append(summary)
    # run BO loop
    stop_crit = FixedIterationsStoppingCondition(i_max=online_query_size)
    bayesopt_loop_SSK.run_loop(objective, stop_crit)
    print("Done")
    validity_optimizer = GrammarGeneticValidityOptimizer(space,
                                                         dynamic=True,
                                                         population_size=100,
                                                         tournament_prob=0.5,
                                                         p_crossover=0.8,
                                                         p_mutation=0.1)
    best_x, best_fit = validity_optimizer.get_best_valid(expected_improvement, auto)
    print(best_x)


def summary(loop, loop_state):
    print("Performing BO step {}".format(loop.loop_state.iteration))


def reformat(x):
    x = x.replace("lb", "(")
    x = x.replace("rb", ")")
    return x


def deformat(x):
    x = x.replace("(", "lb")
    x = x.replace(")", "rb")
    return x


def query(x):
    valid_value = False
    while not valid_value:
        print("How interesting is this formula [0-100]: ")
        print(reformat(x))
        y_out = float(input())
        if 0 <= y_out <= 100:
            valid_value = True
        else:
            print("Invalid Value: please input a number between 0 and 100.")
    # return 1-interest so we're minimizing
    return 1 - y_out / 100


def objective(X):
    # must take an n by 1 ndarray
    Y = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        x = X[i][0]
        Y[i] = query(x)
    return Y


def init_query(init_size, parser):
    xs = []
    ys = []
    # initialize dataset
    print("Please provide " + str(init_size) + " formulas and your interest in them...")
    for i in range(init_size):
        valid_ctl = False
        while not valid_ctl:
            print("Formula " + str(i + 1) + ": ")
            x_out = input()
            x_out = deformat(x_out)
            try:
                parse_trees = parser.parse(x_out.split())
            except ValueError as err:
                print("Grammatical Error: {0}".format(err))
                print("Please try again,\n")
                continue
            # check to see if the given formula is a valid CTL formula
            for _ in parse_trees:
                valid_ctl = True
                break
            if not valid_ctl:
                print("Grammatical Error: invalid formula.")
                print("Please try again,\n")

        xs.append(x_out)

        ys.append(query(x_out))
        print("\n")
        print("###################################################")
    return xs, ys


if __name__ == "__main__":
    # test_str = """
    #         Q -> PROB COMP PCT OB S CB
    #         COMP -> LEQ | LT | GEQ | GT
    #         PCT -> ZERO | ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT | NINE | TEN
    #         S -> S AND R | S OR R | S IMPLIES R | NOT S | LB S RB | LB P RB | T
    #         P -> GLB S | NXT S | FTR S | R UNTIL S | P
    #         R -> S
    #         NOT -> "!"
    #         AND -> "&"
    #         OR -> "|"
    #         LEQ -> "<="
    #         LT -> "<"
    #         GEQ -> ">="
    #         GT -> ">"
    #         PROB -> "P"
    #         ZERO -> "0"
    #         ONE -> "0.1"
    #         TWO -> "0.2"
    #         THREE -> "0.3"
    #         FOUR -> "0.4"
    #         FIVE -> "0.5"
    #         SIX -> "0.6"
    #         SEVEN -> "0.7"
    #         EIGHT -> "0.8"
    #         NINE -> "0.9"
    #         TEN -> "1.0"
    #         NOT -> "!"
    #         AND -> "&"
    #         OR -> "|"
    #         IMPLIES -> "=>"
    #         UNTIL -> "U"
    #         GLB -> "G"
    #         NXT  -> "X"
    #         FTR -> "F"
    #         LB -> "lb"
    #         RB -> "rb"
    #         OB -> "["
    #         CB -> "]"
    #         T -> "state=7" | "state=11" | "state=3" | "state=6" | "state=10"
    #         """
    test_str = """
                N -> PROB COMP PCT OB Q CB
                Q -> N | NOT LB Q RB | Q AND R | Q OR R | Q IMPLIES R | LB S RB | T
                R -> Q
                COMP -> LEQ | LT | GEQ | GT
                PCT -> ZERO | ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT | NINE | TEN
                S -> GLB O | NXT O | FTR O | O UNTIL P
                O -> S | Q
                P -> S | Q
                LEQ -> "<="
                LT -> "<"
                GEQ -> ">="
                GT -> ">"
                PROB -> "P"
                ZERO -> "0"
                ONE -> "0.1"
                TWO -> "0.2"
                THREE -> "0.3"
                FOUR -> "0.4"
                FIVE -> "0.5"
                SIX -> "0.6"
                SEVEN -> "0.7"
                EIGHT -> "0.8"
                NINE -> "0.9"
                TEN -> "1.0"
                NOT -> "!"
                AND -> "&"
                OR -> "|"
                IMPLIES -> "=>"
                UNTIL -> "U"
                GLB -> "G"
                NXT  -> "X"
                FTR -> "F"
                LB -> "lb"
                RB -> "rb"
                OB -> "["
                CB -> "]"
                T -> "state=7" | "state=11" | "state=3" | "state=6" | "state=10"
                """
    test_grammar = Grammar.fromstring(test_str)
    test_space = ParameterSpace([CFGParameter("test_grammar", test_grammar, max_length=15, min_length=0)])
    test_design = RandomDesign(test_space)
    print("Testing...")
    print(timeit.timeit('test_design.get_samples(10)', globals=globals(), number=10))
    samples = test_design.get_samples(10)
    for sample in samples:
        print(reformat(unparse(sample)[0]))
