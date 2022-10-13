# runs the experiments
from algorithm import algorithm
import numpy as np
import pandas as pd
from testing_overall_best_ind import play_top

enemies = np.array([[6, 4], [3, 7]])

# run 3 EAs
for EA_no in range(1, 4):
    print(f'Commencing EA {EA_no}...')
    algorithm(EA_no, enemies)

# test the overall best individual against all enemies 5 times
# read from file
path_individuals = f"./logs/best_individuals_overall.csv"
path_fitness = f"./logs/best_fitness_overall.csv"

best_inds_csv = pd.read_csv(
    path_individuals, delimiter=',', header=None)
best_fitness_csv = pd.read_csv(
    path_fitness, delimiter=',', header=None)

# find best individual
best_inds_arr = best_inds_csv.to_numpy()
best_fitness_arr = best_fitness_csv.to_numpy()

# test that individual
play_top(best_fitness_arr, best_inds_arr)
