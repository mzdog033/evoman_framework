import numpy as np

# roulette wheel


# def cumulative_probability_distribution(selection_probability: list) -> list:
#     '''Calculates the cumulative probability distribution based on individual selection probabilities.'''
#     cum_prob_distribution = []
#     current_cum_prob_dis = 0
#     for i in range(len(selection_probability)):
#         current_cum_prob_dis += selection_probability[i]
#         cum_prob_distribution.append(current_cum_prob_dis)
#     return cum_prob_distribution


# def roulette_wheel_algorithm(cum_prob_distribution, number_of_parents=2) -> list:
#     '''
#     Implements the roulette wheel algorithm as discussed in the
#     accompanying text book by Eiben and Smith (2015).
#     '''
#     current_member = 1
#     mating_pool = []
#     while current_member <= number_of_parents:
#         i = 0  # Index
#         r = np.random.uniform()  # Random number between 0 and 1
#         while cum_prob_distribution[i] < r:
#             i += 1

#         mating_pool.append(i)
#         current_member += 1

#     return mating_pool


def tournament_selection(generation: list, fitness_function: callable, k: int, env) -> list:
    '''
    This implements the tournament selection. K random individual (with replacement) are 
    chosen and compete with each other. The index of the best individual is returned.
    '''

    # First step: Choose a random individual and score it
    number_individuals = generation.shape[0]
    current_winner = np.random.randint(0, number_individuals)
    # Get the score which is the one to beat!
    fitness = fitness_function(env, current_winner)

    # We already have one candidate, so we are left with k-1 to choose
    for candidates in range(k-1):
        contender_number = np.random.randint(0, number_individuals)
        if fitness_function(env, generation[contender_number]) > fitness:
            current_winner = contender_number
            fitness = fitness_function(env, generation[contender_number])

    return current_winner
