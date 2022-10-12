# runs the experiments
from algorithm import algorithm
import numpy as np
import pandas as pd
from find_individual import find_individual
from testing_overall_best_ind import play_top

ea1 = 1
ea2 = 2
ea3 = 3

enemies = np.array([[6, 4], [3, 7]])
overall_best_individuals = np.array([])
overall_best_fitness = np.array([])

# print('Commencing EA 1 - uniform crossover')
# algorithm(ea1, enemies)
# # overall_best_individuals = np.append(overall_best_individuals, EA1_best_ind)
# # overall_best_fitness = np.append(overall_best_fitness, EA1_best_fitness)
# # print('best of overall fitnesses after running EA - should be 2 - one for each enemygroup',
# #       len(overall_best_fitness))


# print('Commencing EA 2 - two point crossover')
# EA2_best_ind, EA2_best_fitness = algorithm(ea2, enemies)
# overall_best_individuals = np.append(overall_best_individuals, EA2_best_ind)
# overall_best_fitness = np.append(overall_best_fitness, EA2_best_fitness)

# print('Commencing EA 3 - adaptive crossover')
# EA3_best_ind, EA3_best_fitness = algorithm(ea3, enemies)
# overall_best_individuals = np.append(overall_best_individuals, EA3_best_ind)
# overall_best_fitness = np.append(overall_best_fitness, EA3_best_fitness)

# run 3 EAs
for EA_no in range(1, 4):
    algorithm(EA_no, enemies)

# test the overall best individual against all enemies 5 times

# read from file
path_individuals = f"./logs/best_individuals_overall.csv"
path_fitness = f"./logs/best_fitness_overall.csv"

best_inds_csv = pd.read_csv(
    path_individuals, delimiter=',', header=None)
best_fitness_csv = pd.read_csv(
    path_fitness, delimiter=',', header=None)

best_inds_arr = best_inds_csv.to_numpy()
best_fitness_arr = best_fitness_csv.to_numpy()

# test that individual
play_top(best_fitness_arr, best_inds_arr)
