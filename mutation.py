import numpy as np
from deap import tools


def probabilistic_gaussian_mutation(mutation_ratio, children, global_population_size, global_genome_size, step_size):
    print('Commencing mutation: Probabilistic Gaussian mutation...')

    mutated_children = np.array([])

    for child in children:
        # random number to compare with mutation_ratio
        random_no = np.random.uniform(0, 1)

        if(random_no < mutation_ratio):
            mutated_child = tools.mutGaussian(
                child, mu=0, sigma=step_size, indpb=0.1)
            # add mutated child to mutated_children list
            mutated_children = np.append(
                mutated_children, mutated_child)
        else:
            # add non-mutated child to mutated_children list
            mutated_children = np.append(
                mutated_children, child)

    mutated_children = mutated_children.reshape(
        global_population_size, global_genome_size)

    return mutated_children


def deterministic_gaussian_mutation(population, step_size, global_population_size, global_genome_size):
    print('Commencing mutation: Deterministic Gaussian Mutation...')

    mutated_children = np.array([])

    for individual in population:
        mutated_child = tools.mutGaussian(
            individual, mu=0, sigma=step_size, indpb=0.1)

        # add mutated child to mutated_children list
        mutated_children = np.append(
            mutated_children, mutated_child)

    mutated_children = mutated_children.reshape(
        global_population_size, global_genome_size)

    return mutated_children


def get_sigma(curr_generation, no_of_generations):
    # sigma(generation) = 1 - 0.9 * generation/total_generations
    sigma = 1 - 0.9 * (curr_generation/no_of_generations)
    return sigma


def add_sigma_to_individual(step_size, population, population_size, genome_size):
    # replaces the last element in the genome with sigma
    new_population = np.array([])

    for i in range(len(population)):
        individual = population[i]
        individual[genome_size-1] = step_size
        new_population = np.append(new_population, individual)

    new_population = new_population.reshape(population_size, genome_size)

    return new_population
