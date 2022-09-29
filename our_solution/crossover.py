import numpy as np
import numpy.random as random


def TwoPointCrossover(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
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
