import numpy as np
from deap import tools
import random
# get parents from selected parents shape
# get selected parents from tournament selection


def twoPointCrossover(selected_parents, parents_size, global_population_size, global_genome_size, toolbox):
    print('Commencing crossover: Two-point Crossover...')

    children = np.array([])

    for parents in range(parents_size):
        parent_ind_1 = selected_parents[np.random.randint(
            len(selected_parents))]
        parent_ind_2 = selected_parents[np.random.randint(
            len(selected_parents))]

        toolbox.register("crossover", tools.cxTwoPoint)

        ind1, ind2 = toolbox.crossover(parent_ind_1, parent_ind_2)
        children = np.append(children, ind1)
        children = np.append(children, ind2)

    children = children.reshape(
        global_population_size, global_genome_size)

    return children

def uniformCrossover(selected_parents, parents_size, global_population_size, global_genome_size, toolbox, indpb = 0.1):
    print('Commencing crossover: Uniform Crossover...')

    children = np.array([])

    for parents in range(parents_size):
        parent_ind_1 = selected_parents[np.random.randint(
            len(selected_parents))]
        parent_ind_2 = selected_parents[np.random.randint(
            len(selected_parents))]

        toolbox.register("crossover", tools.cxUniform)

        ind1, ind2 = toolbox.crossover(
            parent_ind_1, parent_ind_2, indpb)
        children = np.append(children, ind1)
        children = np.append(children, ind2)

    children = children.reshape(
        global_population_size, global_genome_size)

    return children

def adaptiveCrossover(selected_parents, parents_size, global_population_size, global_genome_size, toolbox, indpb = 0.1):
    print('Commencing crossover: Adaptive Crossover...')

    children = np.array([])

    for parents in range(parents_size):
        parent_ind_1 = selected_parents[np.random.randint(
            len(selected_parents))]
        parent_ind_2 = selected_parents[np.random.randint(
            len(selected_parents))]

        parent1 = np.unpackbits(parent_ind_1, axis=1)
        parent2 = np.unpackbits(parent_ind_2, axis=1)
        parent1 = parent1.flatten()
        parent2 = parent2.flatten()
        if (parent1[len(parent1) - 1] == parent2[len(parent2) - 1] == 1):
            toolbox.register("crossover", tools.cxTwoPoint)
            ind1, ind2 = toolbox.crossover(parent_ind_1, parent_ind_2)
        elif (parent1[len(parent1) - 1] == parent2[len(parent2) - 1] == 0):
            toolbox.register("crossover", tools.cxUniform)
            ind1, ind2 = toolbox.crossover(parent_ind_1, parent_ind_2, indpb)
        elif (random.random() < 0.5): 
            toolbox.register("crossover", tools.cxTwoPoint)
            ind1, ind2 = toolbox.crossover(parent_ind_1, parent_ind_2)
        else:
            toolbox.register("crossover", tools.cxUniform)
            ind1, ind2 = toolbox.crossover(parent_ind_1, parent_ind_2, indpb)

        children = np.append(children, ind1)
        children = np.append(children, ind2)

    children = children.reshape(
        global_population_size, global_genome_size)

    return children
