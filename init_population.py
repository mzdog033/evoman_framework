from numpy.random import uniform

from mutation import add_sigma_to_individual


def initialize_population(population_size, genome_size):
    population = uniform(-1, 1, size=(population_size, genome_size))

    return population
