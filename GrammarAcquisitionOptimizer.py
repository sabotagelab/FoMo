import logging

from boss.code.optimizers.GrammarGeneticAlgorithmAcquisitionOptimizer import GrammarGeneticProgrammingOptimizer, unparse

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.initial_designs import RandomDesign

import numpy as np

from model_check import Automaton, Obligation, checkObligation

_log = logging.getLogger(__name__)


class GrammarGeneticValidityOptimizer(GrammarGeneticProgrammingOptimizer):
    """
    Optimizes the acquisition function using Genetic programming over CFG parameters
    """

    def __init__(self, space: ParameterSpace, dynamic: bool = False, num_evolutions: int = 10,
                 population_size: int = 5, tournament_prob: float = 0.5,
                 p_crossover: float = 0.8, p_mutation: float = 0.05
                 ) -> None:
        """
        :param space: The parameter space spanning the search problem (has to consist of a single CFGParameter).
        :param num_steps: Maximum number of evolutions.
        :param dynamic: allow early stopping to choose number of steps (chooses between 10 and 100 evolutions)
        :param num_init_points: Population size.
        :param tournament_prob: proportion of population randomly chosen from which to choose a tree to evolve
                                (larger gives faster convergence but smaller gives better diversity in the population)
        :p_crossover: probability of crossover evolution (if not corssover then just keep the same (reproducton))
        :p_mutation: probability of randomly mutatiaon

        """
        super().__init__(space, dynamic, num_evolutions, population_size, tournament_prob, p_crossover, p_mutation)

    def get_best_valid(self, acquisition: Acquisition, automaton: Automaton, n: int = 10, sample_attempts: int = 3):
        """
        Optimize an acquisition function subject to validity in the automaton using a genetic algorithm

        :param acquisition: acquisition function to be maximized in pair with validity.
        :param automaton: automaton with respect to which formulas must be valid.
        :param n: the number of valid, maximizing formulas to return.
        :param sample_attempts: the number of times to resample the population if none of the samples are valid
        :return: a list of strings that maximize the acquisition function s.t. being valid in the automaton
        """

        # initialize population of tree
        random_design = RandomDesign(self.space)
        # in case the starting population is entirely invalid, resample
        for i in range(sample_attempts):
            population = random_design.get_samples(self.population_size)
            # calc fitness for current population
            fitness_pop = fitness_st_validity(population, acquisition, automaton)
            if sum(fitness_pop) == 0 and (i + 1) >= sample_attempts:
                raise ValueError('No valid samples could be found; try enumerative search instead.')
            elif sum(fitness_pop) != 0:
                break
        standardized_fitness_pop = fitness_pop / sum(fitness_pop)
        # initialize best location and score so far
        best_fit, best_x = get_top_n(population, fitness_pop, n)
        X_max = np.zeros((1, 1), dtype=object)
        X_max[0] = unparse(population[np.argmax(fitness_pop)])
        acq_max = np.max(fitness_pop).reshape(-1, 1)
        iteration_bests = []
        _log.info("Starting local optimization of acquisition function {}".format(type(acquisition)))
        for step in range(self.num_evolutions):
            _log.info("Performing evolution step {}".format(step))
            # recalc fitness
            population, fitness_pop, standardized_fitness_pop = self._recalc_fitness(sample_attempts, population,
                                                                                     standardized_fitness_pop,
                                                                                     acquisition, automaton)
            # update best location and score (if found better solution)
            acq_pop_max = np.max(fitness_pop)
            iteration_bests.append(acq_pop_max)
            _log.info("best acqusition score in the new population".format(acq_pop_max))
            if acq_pop_max > acq_max[0][0]:
                acq_max[0][0] = acq_pop_max
                X_max[0] = unparse(population[np.argmax(fitness_pop)])
            best_x, best_fit = compare_best(best_x, best_fit, population, fitness_pop)
        # if dynamic then keep running (stop when no improvement over most recent 10 populations)
        if self.dynamic:
            stop = False
        else:
            stop = True
        i = 10
        while not stop:
            population, fitness_pop, standardized_fitness_pop = self._recalc_fitness(sample_attempts, population,
                                                                                     standardized_fitness_pop,
                                                                                     acquisition, automaton)
            # update best location and score (if found better solution)
            acq_pop_max = np.max(fitness_pop)
            iteration_bests.append(acq_pop_max)
            _log.info("best acqusition score in the new population".format(acq_pop_max))
            if acq_pop_max > acq_max[0][0]:
                acq_max[0][0] = acq_pop_max
                X_max[0] = unparse(population[np.argmax(fitness_pop)])
            best_x, best_fit = compare_best(best_x, best_fit, population, fitness_pop)
            if acq_max[0][0] == max(iteration_bests[:-10]):
                stop = True
            # also stop if ran for 100 evolutions in total
            if i == 100:
                stop = True
            i += 1

        # return best n solutions from the whole optimization
        return best_x, best_fit

    def _recalc_fitness(self, sample_attempts, init_pop, std_fit_pop, acq, auto):
        # recalc fitness
        for i in range(sample_attempts):
            population = self._evolve(init_pop, std_fit_pop)
            # calc fitness for current population
            fitness_pop = fitness_st_validity(population, acq, auto)
            if sum(fitness_pop) == 0 and (i + 1) >= sample_attempts:
                raise ValueError('No valid samples could be found; try enumerative search instead.')
            elif sum(fitness_pop) != 0:
                standardized_fitness_pop = fitness_pop / sum(fitness_pop)
                return population, fitness_pop, standardized_fitness_pop


def fitness_st_validity(population, acquisition, automaton):
    formulas = unparse(population)
    fitness_pop = acquisition.evaluate(formulas)
    obligations = [Obligation.fromPCTL(reformat(phi[0])) for phi in formulas]
    validity_pop = np.array([checkObligation(automaton, obligation) for obligation in obligations]).reshape(-1, 1)
    return fitness_pop * validity_pop


def get_top_n(population, fitness_pop, n):
    temp_fit = fitness_pop
    maximizers = []
    maxes = []
    for i in range(n):
        max_index = np.argmax(temp_fit)
        x_max = unparse(population[max_index])
        maximizers.append(x_max)
        maxes.append(temp_fit[max_index])
        temp_fit[max_index] = -np.inf
    return maxes, maximizers


def compare_best(pop1, fit1, pop2, fit2):
    while max(fit2) > min(fit1):
        min_ind1 = np.argmin(fit1)
        max_ind2 = np.argmax(fit2)
        if pop2[max_ind2] not in pop1:
            pop1[min_ind1] = pop2[max_ind2]
            fit1[min_ind1] = fit2[max_ind2]
        else:
            fit2[max_ind2] = -np.inf
    return pop1, fit1


def reformat(x):
    x = x.replace("lb", "(")
    x = x.replace("rb", ")")
    return x
