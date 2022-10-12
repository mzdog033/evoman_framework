import numpy as np
import pandas as pd

from find_individual import find_individual


def play_top_ten(no_of_runs, env, enemygroup, EA_no):

    print('\n------- Testing top individuals against all enemies')
    average_fitnesses = np.array([])

    # initialize environment again
    enemies_list = [1, 2, 3, 4, 5, 6, 7, 8]

    for run in range(no_of_runs):
        print('------- Testing with top individual no. ', run+1)

        # reading best inds from list
        path = f"./logs/EP_{EA_no}_best_individuals_run{run+1}_enemygr{enemygroup}.csv"
        best_inds_csv = pd.read_csv(
            path, delimiter=',', header=None)
        best_inds_arr = best_inds_csv.to_numpy()

        # individual from top ten list
        individual = best_inds_arr[run]

        fitnesses = np.array([])
        # run five times
        for j in range(5):
            # test each of the top individuals against all the enemies

            env.update_parameter('enemies', enemies_list)
            f, pl, el, t = env.play(pcont=individual)

            fitnesses = np.append(
                fitnesses, f)

        print(
            f'\nFitnesses of best individual no. {run+1}: {fitnesses}')

        print(
            f'\nAverage fitness of best individual no. {run+1}: {np.mean(fitnesses)}\n')

        average_fitnesses = np.append(
            average_fitnesses, np.mean(fitnesses))

    # best fitness
    best_fitness = np.max(average_fitnesses)

    # find individual with that fitness
    best_individual = find_individual(
        average_fitnesses, best_fitness, best_inds_arr)
    best_individual = np.asarray(best_individual).flatten()

    # saves best individuals + fitness against each enemygroup, in each EA, into one file
    f = open(
        f"./logs/best_individuals_overall.csv", "a")
    g = open(
        f"./logs/best_fitness_overall.csv", "a")
    np.savetxt(f, best_individual, delimiter=',')
    np.savetxt(g, [best_fitness], delimiter=',')
    f.close()
    g.close()

    return best_fitness
