import numpy as np
from find_individual import find_individual
from init_environment import initialize_environment


def play_top(overall_best_fitness, overall_best_individuals):
    print('\n------- Testing overall best individual against all enemies')

    best_fitness = np.max(overall_best_fitness)
    best_individual = find_individual(
        overall_best_fitness, best_fitness, overall_best_individuals)

    enemies_list = [1, 2, 3, 4, 5, 6, 7, 8]

    env = initialize_environment(enemies_list)

    fitnesses = np.array([])
    # run five times
    for j in range(5):
        print('Run no.', j+1)
        # test each of the top individuals against all the enemies

        # env.update_parameter('enemies', enemies_list)
        f, pl, el, t = env.play(pcont=best_individual)

        fitnesses = np.append(
            fitnesses, f)

    # Print stats for current generation
    print(
        f'-------\nBest solution stats - Best: {np.max(fitnesses)} Mean: {np.mean(fitnesses)} Std: {np.std(fitnesses)}')
