import numpy as np
from deap import base, tools


def gaussian_mutation(selected_individuals, mutation_operator: callable):
    # mutation_ratio = 1.4  # change this, this is a random number

    # sigma = 0.3
    # genome_length = selected_group.shape[1]
    # mutated_genes_count = round(mutation_ratio * genome_length)

    # selected_group_count = selected_group.shape[0]
    # mutants = np.array([])

    # for i in range(selected_group_count):
    #     mutant = selected_group[np.random.randint(selected_group_count)]

    #     for j in range(mutated_genes_count):
    #         gene_index = np.random.randint(genome_length)
    #         mutant[gene_index] += min(max(np.random.normal(0, sigma), -1), 1)

    #     mutants = np.concatenate((mutants, mutant), axis=None)

    return mutants.reshape(selected_group_count, genome_length)
