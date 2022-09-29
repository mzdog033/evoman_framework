from deap import base, tools
import numpy as np
from init_environment import initialize_environment
from init_population import initialize_population
from fitness import fittest_solution
from selection import tournament_selection

# Initialize DEAP
toolbox = base.Toolbox()

toolbox = base.Toolbox()
toolbox.register("crossover", tools.cxTwoPoint)
# Gaussian Mutation - WHY THESE PARAMETERS?
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("parent_selection", tools.selTournament,
                 tournsize=5)  # WHY 5?? might need to be 2
# toolbox.register("evaluate", toolbox.evaluate)  # what does this do

k_tournament_size = 2


def main_function():
    for enemy in range(1, 4):  # /// 3-Enemies-loop start
        #  Replace range with enemies_id list

        # initialize environment with enemy
        env = initialize_environment(enemy)

        for EA in range(2):  # /// 2-EAs-loop start
            #  Replace range with EA list

            for run in range(10):  # /// 10-runs-loop start
                # initialize population (and environment??)
                print(f' -------- RUN {run+1} -------- ')

                population = initialize_population(5, 265)  # 150, 265
                average_fitness_pr_gen = np.array([])
                best_fitness_pr_gen = np.array([])

                for generation in range(1, 21):  # /// 20-generational-loop start
                    print(f'Generation no. {generation} running...')

                    #  fitness stuff
                    list_of_fitnesses = fittest_solution(population, env)

                    best_fitness_curr_gen = np.max(list_of_fitnesses)
                    best_fitness_pr_gen = np.append(
                        best_fitness_pr_gen, best_fitness_curr_gen)

                    avg_fitness_curr_gen = np.mean(list_of_fitnesses)
                    average_fitness_pr_gen = np.append(
                        average_fitness_pr_gen, avg_fitness_curr_gen)

                    print(
                        f'Generation {generation} - Best: {best_fitness_curr_gen} Mean: {avg_fitness_curr_gen} Std: {np.std(list_of_fitnesses)}')

                    # select parents
                    selected_parents = tournament_selection(
                        population, list_of_fitnesses, k_tournament_size)

                    print('init population shape', population.shape[0])
                    print('selected parents shape 1',
                          selected_parents.shape[0])
                    parents_size = selected_parents.shape[0]
                    print('selected parents shape 2',
                          selected_parents.shape[1])
                    print('parent size', parents_size)

                    # what is this? only 2 individuals
                    parent1 = selected_parents[0]
                    parent2 = selected_parents[1]

                    children = np.array([])

                    # mate
                    for parents in range(parents_size):
                        parent_ind_1 = parent1[np.random.randint(parents_size)]
                        parent_ind_2 = parent2[np.random.randint(parents_size)]

                        ind1, ind2 = tools.cxTwoPoint(
                            parent_ind_1, parent_ind_2)
                        children = np.append(children, ind1)
                        children = np.append(children, ind2)

                    # children_list = np.array([])
                    # for individual in
                    # children = tools.cxTwoPoint()

                    # select for mutation (mutation ratio)
                    # mutation
                    # select for survival
                    # go to next generation

                    # save best fitness each generation in an array
                    # save average fitness each generation in an array

                #  /// 20-generational-loop finished

                #  save the 20 best fitnesses in each loop (you will have a list of 20 best fitnesses 10 times (10 * 20))
                # aka 10 lists of 20 fitnesses

            # /// 10-runs-loop finished

            # /// 10-best-indivudals-test-loop start
            for best_indivdual in range(10):
                print('hi')

                # replace range with best_individuals list maybe
                # Play the best 10 individual again the enemy (env.play())
                # return the best fitness out of these 10 runs

                # /// 10-best-indivudals-test-loop finished

                # do the same thing for the next experiment: with the other selection mechanism

                # /// 2-EAs-loop finished

                #  /// 3-Enemies-loop finished


main_function()
