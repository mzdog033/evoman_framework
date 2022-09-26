import numpy as np

from mutation import bit_flipping

# uniform always true!


def crossover(parent_1: list, parent_2: list, p_crossover: float, p_uni: float = 0.5, uniform: bool = False) -> tuple:
    '''This function applies crossover for the case of two parents.'''

    # Check if cross-over is applied
    if p_crossover > np.random.uniform():
        # Random uniform crossover
        if uniform:
            child_1 = []
            for gene in range(len(parent_1)):
                if p_uni > np.random.uniform():
                    # Choose first parent
                    child_1.append(parent_1[gene])
                else:
                    child_1.append(parent_2[gene])

            # The second child is used by using an inverse mapping,
            # We use the bit-flipping function defined above.
            child_2 = [bit_flipping(gene) for gene in child_1]

            return child_1, child_2

        # If no uniform crossover is selected, i.e. 1-point crossover is applied
        else:
            # We exclude the splitpoints in the beginning and the end
            split_point = np.random.randint(1, len(parent_1)-1)

            # Now return perform the one-point crossover
            child_1 = np.array([parent_1[gene] if gene <= split_point else parent_2[gene]
                                for gene in range(len(parent_1))])
            child_2 = np.array([parent_2[gene] if gene <= split_point else parent_1[gene]
                                for gene in range(len(parent_1))])

            return child_1, child_2
    else:
        # Just returns the original parents
        return parent_1, parent_2
