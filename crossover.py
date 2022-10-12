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


### first uniform crossover (official DEAP) ###
def cxUniform(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probability for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2)) # size of the smaller individual
    for i in range(size): # iterate over the smaller individual 
        if random.random() < indpb: # if the random number is less than the independent probability 
            ind1[i], ind2[i] = ind2[i], ind1[i] # swap the attributes of the individuals
        else:
            pass # do nothing if the random number is greater than the independent probability 

    return ind1, ind2


### second uniform crossover ###
def UniformCrossover_1(parent1, parent2, prob):
    """Execute a uniform crossover with two parents"""
    """Copy because numpy returns a view of the data""" 
    
    child1 = parent1.copy() # copy the parent1 to child1 
    child2 = parent2.copy() # copy the parent2 to child2 
    if random.random() < prob: # if the random number is less than the probability 
        for i in range(len(parent1)): # iterate over the length of the parent1 

            if random.random() < 0.5: # if the random number is less than 0.5 
                child1[i] = parent1[i] # child1 is equal to parent1 
                child2[i] = parent2[i] # child2 is equal to parent2 
                
            else: # if the random number is greater than 0.5
                child1[i] = parent2[i] # child1 is equal to parent2 
                child2[i] = parent1[i] # child2 is equal to parent1 


### third uniform crossover ###
def UniformCrossover_2(parent1,parent2,point):
    point = random.randint(1,len(parent1))  #Crossover point
    parent1,parent2 = list(parent1),list(parent2) #convert str to list
    for i in range(point,len(parent1)):      #iterate over the length of the parent1
        parent1[i], parent2[i] = parent2[i], parent1[i]       #swap the genetic information
    parent1, parent2 = ''.join(parent1),''.join(parent2)     #Convert list to str
    return parent1,parent2


## testing the functions ##
parent1 = '1011011'                     #parents' Chromosomes
parent2 = '0100100'
print('Parent1:',parent1)
print('Parent2:',parent2)

point = random.randint(1,len(parent1))  #Crossover point
print('Crossover Point:',point)

offspring1,offspring2 = UniformCrossover_2(parent1, parent2, point)
print('Offspring1:',offspring1)         #Offspring Chromosomes
print('Offspring2:',offspring2)
