import numpy as np


def gaussian_mutation(mutation_ratio, children, toolbox, global_population_size, global_genome_size):
    print('Commencing mutation...')

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
