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
        best_solution_per_Enemy = np.array([])

        # INITIALIZE ENVIRONMENT with enemy
        env = initialize_environment(enemy)

        for EA in range(1):  # /// 2-EAs-loop start
            #  Replace range with EA list

            average_fitnesses_pr_run = np.array([])
            best_fitnesses_pr_run = np.array([])
            best_inds_pr_run = np.array([])

            for run in range(10):  # /// 10-runs-loop start
                print(f' -------- RUN {run+1} -------- ')
                print('Initial stats: ')

                # INITIALIZE POPULATION
                population = initialize_population(10, 265)  # 150, 265
                genome_size = population.shape[1]
                population_size = population.shape[0]
                average_fitness_pr_gen = np.array([])
                best_fitness_pr_gen = np.array([])
                best_inds_pr_gen = np.array([])

                for generation in range(1, 2):  # /// 20-generational-loop start
                    #  fitness stuff
                    list_of_fitnesses = fittest_solution(population, env)
                    # TRYING TO MAP FITNESS TO POPULATION
                    # print('fitness ',
                    #       list_of_fitnesses[0])
                    # print('pop ',
                    #       population[0])

                    # pop_fitness_map = dict(zip(population, list_of_fitnesses))
                    # print('fitness-population mapping', pop_fitness_map)

                    # best fitness curr generation
                    best_fitness_curr_gen = np.max(list_of_fitnesses)
                    best_ind_idx = np.where(list_of_fitnesses ==
                                            best_fitness_curr_gen)
                    # best individual curr generation
                    best_ind_curr_gen = population[best_ind_idx[0][0]]
                    # avg fitness curr generation
                    avg_fitness_curr_gen = np.mean(list_of_fitnesses)

                    # add to lists
                    best_fitness_pr_gen = np.append(
                        best_fitness_pr_gen, best_fitness_curr_gen)
                    best_inds_pr_gen = np.append(
                        best_inds_pr_gen, best_ind_curr_gen)
                    average_fitness_pr_gen = np.append(
                        average_fitness_pr_gen, avg_fitness_curr_gen)

                    #  printing..
                    print(f'Generation no. {generation} running...')

                    # print('population size.:', population_size)

                    # PARENT SELECTION
                    selected_parents = tournament_selection(
                        population, list_of_fitnesses, k_tournament_size)  # selected_parents size (5 * 256)
                    parents_size = selected_parents.shape[0]
                    # print('parents no.:', parents_size)

                    # CROSSOVER
                    children = np.array([])

                    for parents in range(parents_size):
                        parent_ind_1 = selected_parents[np.random.randint(
                            len(selected_parents))]
                        parent_ind_2 = selected_parents[np.random.randint(
                            len(selected_parents))]

                        ind1, ind2 = toolbox.crossover(
                            parent_ind_1, parent_ind_2)
                        children = np.append(children, ind1)
                        children = np.append(children, ind2)

                    children = children.reshape(population_size, genome_size)
                    # print('children no.:', children.shape[0])

                    # MUTATION
                    # choose a few children based on some probability "mutation_ratio"
                    mutated_children = np.array([])
                    for child in children:
                        # random number to compare with mutation_ratio
                        random_no = np.random.uniform(0, 1)

                        if(random_no < mutation_ratio):
                            mutated_child = toolbox.mutate(child)
                            # add mutated child to mutated_children list
                            mutated_children = np.append(
                                mutated_children, mutated_child)
                        else:
                            # add non-mutated child to mutated_children list
                            mutated_children = np.append(
                                mutated_children, child)

                    mutated_children = mutated_children.reshape(
                        population_size, genome_size)
                    # print('mutated no.:', mutated_children.shape[0])

                    # select for survival

                    # go to next generation
                    # For now, lets just replace the whole population.
                    population = mutated_children

                    # Print stats for current generation
                    print(
                        f'Generation {generation} stats - Best: {best_fitness_curr_gen} Mean: {avg_fitness_curr_gen} Std: {np.std(list_of_fitnesses)}')

                #  /// 20-generational-loop finished

                # saving lists of best fitnesses, average fitnesses and best individuals from the generational-loop
                average_fitnesses_pr_run = np.append(
                    average_fitnesses_pr_run, average_fitness_pr_gen)
                best_fitnesses_pr_run = np.append(
                    best_fitnesses_pr_run, best_fitness_pr_gen)
                best_inds_pr_run = np.append(
                    best_inds_pr_run, best_inds_pr_gen)
            # /// 10-runs-loop finished

            # /// 10-best-indivudals-test-loop start
            for i in range(10):
                print('Testing top individuals against enemy %...', enemy)
                individual = best_inds_pr_run[i]
                fitness_from_best_ind_runs = np.array([])

                # run five times
                for j in range(5):
                    f, pl, el, t = env.play(pcont=individual)

                    fitness_from_best_ind_runs = np.append(
                        fitness_from_best_ind_runs, f)

                best_solution_per_Enemy = np.append(
                    best_solution_per_Enemy, np.mean(fitness_from_best_ind_runs))
                print(
                    f'Best solution for enemy {enemy}: {best_solution_per_Enemy[enemy-1]}')

                # /// 10-best-indivudals-test-loop finished

                # do the same thing for the next experiment: with the other selection mechanism

        # /// 2-EAs-loop finished

    #  /// 3-Enemies-loop finished


main_function()
