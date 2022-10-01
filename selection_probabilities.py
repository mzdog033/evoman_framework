import numpy as np


def selection_probabilities(generation, fitness_function: callable, sigma_scaling=False) -> list:
    '''
    Calculates the individual selection probabilities based on the fitness function. 
    Applies sigma-scaling if desired.
    '''

    number_individuals = generation.shape[0]
    total_fitness = np.sum([fitness_function(generation[i])
                           for i in range(number_individuals)])

    if sigma_scaling == True:

        mean_fitness = total_fitness/number_individuals
        std_fitness = np.std([fitness_function(generation[i])
                             for i in range(number_individuals)])
        c = 2  # Constant

        fitness_sigma = [np.max(fitness_function(generation[i])-(mean_fitness-(c*std_fitness)), 0) for i
                         in range(number_individuals)]

        # Now we need to sum up the sigma-scaled fitnesses
        total_fitness_sigma = np.sum(fitness_sigma)
        selection_prob = [fitness_sigma[i] /
                          total_fitness_sigma for i in range(number_individuals)]
    else:
        # Apply normal inverse scaling
        selection_prob = [(fitness_function(generation[i])/total_fitness)
                          for i in range(number_individuals)]
    return selection_prob
