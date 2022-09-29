import numpy as np

from evoman.environment import Environment


def fittest_solution(population, env):
    '''This returns the highest fitness value of the whole generation.'''
    list_of_fitnesses = np.array([])

    for i in range(population.shape[0]):
        individual = population[i]
        fitness = simulation(env, individual)

        list_of_fitnesses = np.append(list_of_fitnesses, fitness)

    # best_fitness = np.max(list_of_fitnesses)  # one-max method: using np.max

    # return list of all fitnesses in the population, and the best fitness in that list
    return list_of_fitnesses


def simulation(env: Environment, x):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return float(fitness)
