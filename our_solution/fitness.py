import numpy as np

from evoman.environment import Environment


def one_max(x: list) -> float:
    '''Takes a list of length bit_length and returns the sum of its elements.'''
    return np.sum(x)


def fittest_solution(fitness_function: callable, population, env) -> float:
    '''This returns the highest fitness value of the whole generation.'''
    list_of_fitnesses = np.array([])

    for i in range(population.shape[0]):
        individual = population[i]
        fitness = simulation(env, individual)

        # not using one_max now
        # list_of_sums = np.append(
        #     list_of_sums, [fitness_function(individual)])
        list_of_fitnesses = np.append(list_of_fitnesses, fitness)

    return np.max(list_of_fitnesses)


def simulation(env: Environment, x):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness


# evaluation - map fitness from simulation (y) to population x
def evaluate(x, env):
    return np.array(list(map(lambda y: simulation(env, y), x)))
