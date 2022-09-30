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
genome_size = 256
mutation_ratio = 0.1
toolbox.register("mutate", tools.mutGaussian, mu=0,
                 sigma=1, indpb=mutation_ratio)
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

                population = initialize_population(10, 265)  # 150, 265
                genome_size = population.shape[1]
                average_fitness_pr_gen = np.array([])
                best_fitness_pr_gen = np.array([])

                for generation in range(1, 21):  # /// 20-generational-loop start
                    #  fitness stuff
                    list_of_fitnesses = fittest_solution(population, env)

                    best_fitness_curr_gen = np.max(list_of_fitnesses)
                    best_fitness_pr_gen = np.append(
                        best_fitness_pr_gen, best_fitness_curr_gen)

                    avg_fitness_curr_gen = np.mean(list_of_fitnesses)
                    average_fitness_pr_gen = np.append(
                        average_fitness_pr_gen, avg_fitness_curr_gen)

                    #  printing..
                    print(f'Generation no. {generation} running...')
                    print(
                        f'Generation {generation} - Best: {best_fitness_curr_gen} Mean: {avg_fitness_curr_gen} Std: {np.std(list_of_fitnesses)}')

                    # select parents
                    selected_parents = tournament_selection(
                        population, list_of_fitnesses, k_tournament_size)  # selected_parents size (2 * 256)

                    parents_size = selected_parents.shape[0]

                    # what is this? only 2 individuals, because thats the half of 5. and population is currently only 5!

                    # mate
                    children = np.array([])

                    for parents in range(parents_size):
                        parent_ind_1 = selected_parents[np.random.randint(
                            len(selected_parents))]
                        parent_ind_2 = selected_parents[np.random.randint(
                            len(selected_parents))]

                        ind1, ind2 = toolbox.crossover(
                            parent_ind_1, parent_ind_2)
                        children = np.append(children, ind1)

                    children = children.reshape(parents_size, genome_size)

                    # mutation
                    # choose a few children based on some probability
                    mutated_children = np.array([])
                    for child in children:
                        random_no = np.random.uniform(0, 1)
                        print('random number', random_no)

                        if(random_no < mutation_ratio):
                            print('mutation time')
                            mutated_child = toolbox.mutate(child)
                            # add mutated child to mutated_children list
                            mutated_children = np.append(
                                mutated_children, mutated_child)
                        else:
                            print('no mutation time')
                            # add non-mutated child to mutated_children list
                            mutated_children = np.append(
                                mutated_children, child)

                    print('mutated childre no.', len(mutated_children))
                    mutated_children = mutated_children.reshape(
                        len(children), genome_size)

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
