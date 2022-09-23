import numpy as np


def one_max(x: list) -> float:
    '''Takes a list of length bit_length and returns the sum of its elements.'''
    return np.sum(x)


def fittest_solution(fitness_function: callable, generation) -> float:
    '''This returns the highest fitness value of the whole generation.'''
    return np.max([fitness_function(generation[i]) for i in range(generation.shape[0])])


def simulation(env, x):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness
