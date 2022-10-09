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


def round_robin_tournament_selection(individual_id, list_of_fitnesses):
    # dont need the opponent itself, only their fitness
    # FIND 10 OPPONENTS
    list_of_random_opponents = np.array([])
    for i in range(10):
        opponent_fitness = np.random.choice(list_of_fitnesses)
        list_of_random_opponents = np.append(
            list_of_random_opponents, opponent_fitness)

    # individuals index in population list = index in fitness list too
    individuals_fitness = list_of_fitnesses[individual_id]

    # COMPETE WITH 10 OPPONENTS
    individual_score = 0
    for opponent_fitness in list_of_random_opponents:
        if(individuals_fitness > opponent_fitness):
            individual_score = individual_score + 1

    return int(individual_score)
