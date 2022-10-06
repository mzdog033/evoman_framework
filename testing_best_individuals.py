from msilib.schema import Environment
import numpy as np
import pandas as pd


def play_top_ten(enemies, no_of_runs, enemy, env):

    print('\n------- Testing top individuals against enemy group of', enemies)
    best_solution_per_Enemy = np.array([])

    for run in range(no_of_runs):
        print('------- Testing with top individual no. ', run+1)

        path = './logs/µcommaλbest_individuals_pr_run' + \
            str(run+1)+'_enemy_group_'+str(enemy+1)+'.csv'
        best_inds_csv = pd.read_csv(
            path, delimiter=',', header=None)
        best_inds_arr = best_inds_csv.to_numpy()

        # test 10 best solutions against several enemies innit? The same group of enemies
        individual = best_inds_arr[run]

        fitness_from_best_ind_runs = np.array([])
        # run five times
        for j in range(5):
            # test each of the top 10 individuals against each set of enemies
            f, pl, el, t = env.play(pcont=individual)

            fitness_from_best_ind_runs = np.append(
                fitness_from_best_ind_runs, f)

        best_solution_per_Enemy = np.append(
            best_solution_per_Enemy, np.mean(fitness_from_best_ind_runs))
        print(
            f'Average solution for enemy group {enemies[enemy]}, run {run+1}: {best_solution_per_Enemy[run]}')

    return best_solution_per_Enemy
