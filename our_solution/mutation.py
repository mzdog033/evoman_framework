import numpy as np

# bit flipping and applying the bit flip


def bit_flipping(x: list) -> list:
    '''This function flips the bits in case mutation is applied.'''
    return 1 if x == 0 else 0


def mutation_operator(mutation_function: callable, p_mutation: float, x: list) -> np.ndarray:
    '''This function takes the mutation function and applies it 
    element-wise to the genes according to the mutation rate.'''

    return np.asarray([mutation_function(gene) if (np.random.uniform() <= p_mutation) else gene for gene in x])
