import numpy as np

from fitness import fittest_solution

# parent selection


def tournament_selection(population, list_of_fitnesses, k_rounds, global_population_size, global_genome_size) -> list:
    print('Commencing parents selection: Tournament selection...')
    parents_size = round(global_population_size/k_rounds)
    selected_parents = np.array([])

    for i in range(parents_size):
        individual1 = np.random.randint(global_population_size)
        individual2 = np.random.randint(global_population_size)

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
        parents_size, global_genome_size)

    return selected_parents

# survival selection


def round_robin_tournament_selection(individual_id, list_of_fitnesses):

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


def probabilistic_survival_selection(population, list_of_fitnesses, mutated_children, env, global_population_size, global_genome_size):
    # add children to population
    population = np.append(population, mutated_children, axis=0)
    population_size = population.shape[0]

    # now find the fitness of all the children
    list_of_offspring_fitnesses = fittest_solution(
        mutated_children, env)

    # add children fitnesses to list of fitnesses
    list_of_fitnesses = np.append(
        list_of_fitnesses, list_of_offspring_fitnesses)

    # ROUND ROBIN TOURNAMENT SELECTION
    print('Commencing survival selection: Round-robin tournament...')

    round_robin_scores = np.array([])  # tournament scores
    individual_ids = np.arange(population_size)  # invdivdual ids

    # playing the tournaments
    for individual_id in range(population_size):
        individual_score = round_robin_tournament_selection(
            individual_id, list_of_fitnesses)
        round_robin_scores = np.append(
            round_robin_scores, individual_score)

    # dict of indivdual_id - number of wins
    id_score_dict = dict(
        zip(individual_ids, round_robin_scores))

    # sort dict by value - reverse sorted for some reason
    dict_sort_by_scores_ascending = sorted(
        id_score_dict.items(), key=lambda item: item[1])

    # sort dict the correct way by scores
    dtype = [('id', int), ('score', int)]
    dict_sort_by_scores_descending = np.array(
        dict_sort_by_scores_ascending, dtype=dtype)[::-1]

    # collect the ids of the winners from the competition (keys)
    round_robin_winners_ids = np.array([])

    for i in range(population_size):
        round_robin_winners_ids = np.append(
            round_robin_winners_ids, int(dict_sort_by_scores_descending[i][0]))

    # get top n winner ids, where n is original population size
    top_n_winners = round_robin_winners_ids[:global_population_size]

    # get top n winners from population, and create new population.
    new_population = np.array([])
    for i in range(len(top_n_winners)):
        new_population = np.append(
            new_population, population[int(top_n_winners[i])])

    # reshape
    new_population = new_population.reshape(
        global_population_size, global_genome_size)

    # return new population of tournament winners
    return new_population
