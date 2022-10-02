import numpy as np
import numpy.random as random


def TwoPointCrossover(ind1, ind2):
    size = len(ind1)
    crossover_point1 = random.randint(1, size)
    crossover_point2 = random.randint(1, size - 1)
    if crossover_point2 >= crossover_point1:
        crossover_point2 += 1
    else:  # Swap the two cx points
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1

    ind1[crossover_point1:crossover_point2], ind2[crossover_point1:crossover_point2] \
        = ind2[crossover_point1:crossover_point2].copy(), ind1[crossover_point1:crossover_point2].copy()

    return ind1, ind2
