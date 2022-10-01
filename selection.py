import numpy as np


def tournament_selection(population, list_of_fitnesses, k_rounds) -> list:
    population_size = population.shape[0]
    genome_size = population.shape[1]
    half_population_size = round(population_size/k_rounds)
    selected_parents = np.array([])

    for i in range(half_population_size):
        individual1 = np.random.randint(population_size)
        individual2 = np.random.randint(population_size)

        fitness1 = list_of_fitnesses[individual1]
        fitness2 = list_of_fitnesses[individual2]

        # time to compete!
        if fitness1 > fitness2:
            selected_parents = np.concatenate(
                (selected_parents, population[individual1]))
        else:
            selected_parents = np.concatenate(
                (selected_parents, population[individual2]))

    selected_parents = selected_parents.reshape(
        half_population_size, genome_size)

    return selected_parents
