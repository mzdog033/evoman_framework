import numpy as np


def find_individual(list_of_fitnesses, fitness, population):
    # find individual with this fitness
    individual_id = np.where(list_of_fitnesses ==
                             fitness)

    individual = population[individual_id[0][0]]
    return individual
