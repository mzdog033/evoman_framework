from numpy.random import uniform


def initialize_population(n_population: int, genome_lenght: int):
    pop = uniform(-1, 1, size=(n_population, genome_lenght))
    return pop
