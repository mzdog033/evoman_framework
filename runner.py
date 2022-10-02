from init_environment import initialize_environment
from init_population import initialize_population
from fitness import simulation, fittest_solution
from crossover import crossover
from mutation import bit_flipping, mutation_operator
# from fitness import evaluate
from selection import tournament_selection
# from selection_probabilities import selection_probabilities

import numpy as np


def runner():
    generation = 1  # counter
    init_env = initialize_environment()

    # variables
    bit_length = 265
    n_population = 2  # increase ofc
    mutation_rate = 1/bit_length
    crossover_prob = 0.6  # to be tuned? TODO
    n_iterations = 1
    n_parents = 2  # what is this
    k_tournament_size = 5

    # initialize population
    init_pop = initialize_population(n_population, bit_length)
    # print('initial population', init_pop)

    best_fitness = fittest_solution(init_pop, init_env)
    print(
        'The current best solution in the initial generation is {0}'.format(best_fitness))

    for i in range(1, n_iterations + 1):  # generation loop
        new_generation = []

        # select parents - population / 2 (2 parents)
        parents_range = int(n_population/n_parents)
        mating_pool = []
        for j in range(parents_range):  # individiual loop

            # loop through list of population (halfed) twice (n_parents)
            for parent in range(n_parents):
                selected_mate = tournament_selection(
                    init_pop, simulation, k_tournament_size, init_env)
                mating_pool.append(selected_mate)

            # now mating time - crossover
            parent_1 = init_pop[mating_pool[0]]
            parent_2 = init_pop[mating_pool[1]]
            child_1, child_2 = crossover(
                parent_1, parent_2, crossover_prob, uniform=True)

            # Mutate the children
            child_1 = mutation_operator(bit_flipping, mutation_rate, child_1)
            child_2 = mutation_operator(bit_flipping, mutation_rate, child_2)

            # generational survival selection where all parents are replaced
            new_generation.append(child_1.tolist())
            new_generation.append(child_2.tolist())

        new_population = np.asarray(new_generation)

        best_fitness, list_of_fitnesses = fittest_solution(
            new_population, init_env)

        if i % 10 == 0:
            # print(
            #     'The current best population in generation {0} is {1}'.format(i, best_fitness))
            # prints stats
            print(
                f'Generation {generation} - Best: {best_fitness} Mean: {np.mean(list_of_fitnesses)} Std: {np.std(list_of_fitnesses)}')

        # increment generation
        generation += 1

        # Include a condition that stops when the optimal solution is found
        if best_fitness == bit_length:
            print('---'*20)
            print(
                'Done! The algorithm has found the optimal solution! Best fitness == bit_length 265')
            print(
                'The current best population in generation {0} is {1}'.format(i, best_fitness))
            break


runner()
