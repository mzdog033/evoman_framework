from deap import base, tools
import numpy as np
from init_environment import initialize_environment
from init_population import initialize_population
from fitness import fittest_solution
from selection import tournament_selection
import os
import pandas as pd

# Initialize DEAP
toolbox = base.Toolbox()

toolbox = base.Toolbox()
toolbox.register("crossover", tools.cxTwoPoint)
# Gaussian Mutation - WHY THESE PARAMETERS?
global_genome_size = 265
mutation_ratio = 0.2
toolbox.register("mutate", tools.mutGaussian, mu=0,
                 sigma=1, indpb=0.1)
# toolbox.register("evaluate", toolbox.evaluate)  # what does this do
global_population_size = 150
no_of_runs = 10
no_of_generations = 20
k_tournament_size = 2


def main_function():

    # FIX ISSUE IN ENVIRONMENT WITH PADDED ZEROS

    #

    best_solution_per_Enemy = np.array([])
    for enemy in range(1, 4):  # /// 3-Enemies-loop start
        # check if files exist MAKE FOR EACH ENEMEY!!!!!!!!!!!!
        # if os.path.exists('./logs/average_fitnesses_pr_run.csv'):
        #     os.remove('./logs/average_fitnesses_pr_run.csv')
        # if os.path.exists('./logs/best_fitnesses_pr_run.csv'):
        #     os.remove('./logs/best_fitnesses_pr_run.csv')
        # if os.path.exists('./logs/best_individuals_pr_run.csv'):
        #     os.remove('./logs/best_individuals_pr_run.csv')

        # INITIALIZE ENVIRONMENT with enemy
        env = initialize_environment(enemy)

        for EA in range(1):  # /// 2-EAs-loop start
            #  Replace range with EA list

            average_fitness_pr_gen = np.array([])
            best_fitness_pr_gen = np.array([])
            best_inds_pr_gen = np.zeros((no_of_runs, global_genome_size))
            best_fitness = -1
            for run in range(no_of_runs):  # /// 10-runs-loop start
                print(f' -------- RUN {run+1} -------- ')
                print('Initial stats: ')

                # INITIALIZE POPULATION
                population = initialize_population(
                    global_population_size, global_genome_size)  # 150, 256
                genome_size = population.shape[1]
                population_size = population.shape[0]

                # /// 20-generational-loop start
                for generation in range(1, no_of_generations+1):
                    #  fitness stuff
                    list_of_fitnesses = fittest_solution(population, env)

                    # best fitness curr generation
                    best_fitness_curr_gen = np.max(list_of_fitnesses)
                    # avg fitness curr generation
                    avg_fitness_curr_gen = np.mean(list_of_fitnesses)

                    # check if best fitness this gen is better than the current best fitness
                    if best_fitness_curr_gen > best_fitness:
                        best_fitness = best_fitness_curr_gen

                        # find individual with this fitness
                        best_ind_idx = np.where(list_of_fitnesses ==
                                                best_fitness)
                        best_individual = population[best_ind_idx[0][0]]
                        # save individual to list
                        best_inds_pr_gen[run] = best_individual

                    # add to lists
                    best_fitness_pr_gen = np.append(
                        best_fitness_pr_gen, best_fitness_curr_gen)
                    average_fitness_pr_gen = np.append(
                        average_fitness_pr_gen, avg_fitness_curr_gen)
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
                    # population = mutated_children # µ,λ
                    population = np.concatenate((population, mutated_children)) # µ+λ
                    population_size = population.shape[0]
                    population = population.reshape(population_size, genome_size)

                    # Print stats for current generation
                    print(
                        f'-------\nGeneration {generation} stats - Best: {best_fitness_curr_gen} Mean: {avg_fitness_curr_gen} Std: {np.std(list_of_fitnesses)}')

                #  /// 20-generational-loop finished

                # save to file

                if run == no_of_runs-1:
                    average_fitness_pr_gen = average_fitness_pr_gen.reshape(
                        no_of_runs, no_of_generations)
                    best_fitness_pr_gen = best_fitness_pr_gen.reshape(
                        no_of_runs, no_of_generations)
                    best_inds_pr_gen = best_inds_pr_gen.reshape(
                        no_of_runs, genome_size)

                f = open("./logs/µplusλaverage_fitnesses_pr_run"+str(run)+".csv", "a")
                g = open("./logs/µplusλbest_fitnesses_pr_run"+str(run)+".csv", "a")
                h = open("./logs/µplusλbest_individuals_pr_run"+str(run)+".csv", "a")
                np.savetxt(f, average_fitness_pr_gen, delimiter=',')
                np.savetxt(g, best_fitness_pr_gen, delimiter=',')
                np.savetxt(h, best_inds_pr_gen, delimiter=',')
                f.close()
                g.close()
                h.close()

            # /// 10-runs-loop finished

            # /// 10-best-indivudals-test-loop start

            print('\n------- Testing top individuals against enemy ', enemy)
            for i in range(no_of_runs):
                print('------- Testing with top individual no. ', i+1)

                path = './logs/µplusλbest_individuals_pr_run'+str(i)+'.csv'
                best_inds_csv = pd.read_csv(
                    path, delimiter=',', header=None)
                best_inds_arr = best_inds_csv.to_numpy()

                individual = best_inds_arr[i]
                print(individual)

                fitness_from_best_ind_runs = np.array([])
                # run five times
                for j in range(5):
                    f, pl, el, t = env.play(pcont=individual)

                    fitness_from_best_ind_runs = np.append(
                        fitness_from_best_ind_runs, f)

                best_solution_per_Enemy = np.append(
                    best_solution_per_Enemy, np.mean(fitness_from_best_ind_runs))
                print(
                    f'Best solution for enemy {enemy}:d {best_solution_per_Enemy[enemy-1]}')

                # /// 10-best-indivudals-test-loop finished

                # do the same thing for the next experiment: with the other selection mechanism

        # /// 2-EAs-loop finished

    #  /// 3-Enemies-loop finished


main_function()
