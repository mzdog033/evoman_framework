
import numpy as np
from init_environment import initialize_environment
from init_population import initialize_population


def main_function():
    for enemy in range(3):  # /// 3-Enemies-loop start
        #  Replace range with enemies_id list

        # initialize environment with enemy
        env = initialize_environment(enemy)

        for EA in range(2):  # /// 2-EAs-loop start
            #  Replace range with EA list

            for run in range(10):  # /// 10-runs-loop start
                # initialize population (and environment??)

                population = initialize_population(5, 265)  # 150, 265
                average_fitness_pr_gen = np.array([])
                best_fitness_pr_gen = np.array([])

                for generation in range(1, 21):  # /// 20-generational-loop start
                    # track fitness
                    # track best individuals from population
                    # select for mating
                    # mate
                    # select for mutation (mutation ratio)
                    # mutation
                    # select for survival
                    # go to next generation

                    # save best fitness each generation in an array
                    # save average fitness each generation in an array

                    print('generation no: ', generation)
                #  /// 20-generational-loop finished

                #  save the 20 best fitnesses in each loop (you will have a list of 20 best fitnesses 10 times (10 * 20))
                # aka 10 lists of 20 fitnesses

            # /// 10-runs-loop finished

            # /// 10-best-indivudals-test-loop start
            for best_indivdual in range(10):

                # replace range with best_individuals list maybe
                # Play the best 10 individual again the enemy (env.play())
                # return the best fitness out of these 10 runs

                # /// 10-best-indivudals-test-loop finished

                # do the same thing for the next experiment: with the other selection mechanism

                # /// 2-EAs-loop finished

                #  /// 3-Enemies-loop finished
