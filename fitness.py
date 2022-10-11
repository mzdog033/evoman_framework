import numpy as np

from evoman.environment import Environment


def fittest_solution(population, env):
    list_of_fitnesses = np.array([])

    for i in range(population.shape[0]):
        individual = population[i]
        fitness = simulation(env, individual)

        list_of_fitnesses = np.append(list_of_fitnesses, fitness)

    return list_of_fitnesses


def simulation(env: Environment, x):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return float(fitness)
