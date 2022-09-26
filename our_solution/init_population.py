from numpy.random import uniform


def initialize_population(n_population: int, bit_length: int):
    # CODE FROM SGA-SOLUTION
    pop = uniform(-1, 1, size=(n_population, bit_length))
    return pop
