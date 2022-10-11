import numpy as np

# get parents from selected parents shape
# get selected parents from tournament selection


def crossover(selected_parents, parents_size, global_population_size, global_genome_size, toolbox):
    print('Commencing crossover: Adaptive Crossover...')

    children = np.array([])

    for parents in range(parents_size):
        parent_ind_1 = selected_parents[np.random.randint(
            len(selected_parents))]
        parent_ind_2 = selected_parents[np.random.randint(
            len(selected_parents))]

        ind1, ind2 = toolbox.crossover(
            parent_ind_1, parent_ind_2)
        children = np.append(children, ind1)
        children = np.append(children, ind2)

    children = children.reshape(
        global_population_size, global_genome_size)

    return children
