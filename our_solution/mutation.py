import numpy as np

# bit flipping and applying the bit flip

# NEEDS TO BE RANDOM RESEETING.. IDK... YEAH BC WE ARE NOT WORKING WITH BINARY NUMBERS


def bit_flipping(x: list) -> list:
    '''This function flips the bits in case mutation is applied.'''
    return 1 if x == 0 else 0

# uniform mutation


def mutation_operator(mutation_function: callable, p_mutation: float, x: list) -> np.ndarray:
    '''This function takes the mutation function and applies it 
    element-wise to the genes according to the mutation rate.'''

    return np.asarray([mutation_function(gene) if (np.random.uniform() <= p_mutation) else gene for gene in x])


# code from adrian
def gaussian_mutation(selected_group):  # TRY THIS!!!!!
    mutation_ratio = 1.4  # change this, this is a random number

    sigma = 0.3
    genome_length = selected_group.shape[1]
    mutated_genes_count = round(mutation_ratio * genome_length)

    selected_group_count = selected_group.shape[0]
    mutants = np.array([])

    for i in range(selected_group_count):
        mutant = selected_group[np.random.randint(selected_group_count)]

        for j in range(mutated_genes_count):
            gene_index = np.random.randint(genome_length)
            mutant[gene_index] += min(max(np.random.normal(0, sigma), -1), 1)

        mutants = np.concatenate((mutants, mutant), axis=None)

    return mutants.reshape(selected_group_count, genome_length)
