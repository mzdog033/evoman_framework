from numpy.random import uniform

from mutation import add_sigma_to_individual


def initialize_population(population_size, genome_size, sigma):
    pop = uniform(-1, 1, size=(population_size-1, genome_size))

    # genome size becomes n + 1, as we add sigma as the end of the genome
    population = add_sigma_to_individual(sigma, pop)

    return population
