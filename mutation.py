import numpy as np
from deap import tools


def probabilistic_gaussian_mutation(mutation_ratio, children, toolbox, global_population_size, global_genome_size):
    print('Commencing mutation: Probabilistic Gaussian mutation...')

    mutated_children = np.array([])

    for child in children:
        # random number to compare with mutation_ratio
        random_no = np.random.uniform(0, 1)

        if(random_no < mutation_ratio):
            mutated_child = toolbox.mutate(child)
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


def deterministic_gaussian_mutation(population, curr_generation, total_generations, global_population_size, global_genome_size):
    print('Commencing mutation: Deterministic Gaussian Mutation...')

    mutated_children = np.array([])

    for individual in population:
        # sigma which gets smaller over the generations
        step_size = get_sigma(curr_generation, total_generations)
        print(f'step size {step_size} in generation {curr_generation}')

        # mutated_child = toolbox.mutate(individual)
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
