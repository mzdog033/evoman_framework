from algorithm import initialize_population, initialize_environment
from fitness import simulation, fittest_solution, one_max
from crossover import crossover

import numpy as np


def runner():
    # initlaize pop
    # run loops
    # inside loop, do fitness, selection, crossover, mutation, and create a new generation
    # add generation to list? keep track of generation number.
    # keep track of fitnesses. save the best ones each generation

    init_pop = initialize_population(5, 265)
    init_env = initialize_environment()
    print('pop', init_pop)
    # print('pop', init_pop.reshape())
    # print('env', init_env)

    for en in range(1, 4):
        # fitness function with population and environment

        # fitness array
        fitness = np.array([])
        for individual in init_pop:
            # print('individual', individual)
            individual_fitness = simulation(init_env, individual)
            # print('individual fitness', individual_fitness)

        fitness.append(individual_fitness)
        # parents = selection(fitness, self.population)
        offspring = crossover(parents)

        # print('fitness array', fitness)


runner()
