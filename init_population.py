from numpy.random import uniform

# initialize population with random values between 0 and 1 (uniform distribution) 
def initialize_population(n_population: int, genome_lenght: int): # -> np.array
    pop = uniform(-1, 1, size=(n_population, genome_lenght)) # np.array
    return pop

# x = initialize_population(10, 256)
# print(x)
