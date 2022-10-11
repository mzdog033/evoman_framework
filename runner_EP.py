from deap import base, tools
import numpy as np
from crossover import crossover
from init_environment import initialize_environment
from init_population import initialize_population
from fitness import fittest_solution
from mutation import gaussian_mutation
from selection import probabilistic_survival_selection, round_robin_tournament_selection, tournament_selection
import pandas as pd

from testing_best_individuals import play_top_ten

# Runner 2 copy


# Initialize DEAP
toolbox = base.Toolbox()
toolbox.register("crossover", tools.cxTwoPoint)

mutation_ratio = 0.2
toolbox.register("mutate", tools.mutGaussian, mu=0,
                 sigma=1, indpb=0.1)

global_population_size = 10
global_genome_size = 265
no_of_runs = 1
no_of_generations = 2

k_tournament_size = 2

np.random.seed(420)  # why 420? copied form optimization_generalist_demo.py


def main_function():
    average_solution_per_enemyset = np.array([])
    for enemy in range(1):  # /// 3-Enemies-loop start

        # INITIALIZE ENVIRONMENT with sets of enemies
        enemies = np.array([[6, 4]])
        env = initialize_environment(enemies[enemy])

        for EA in range(1):  # /// 2-EAs-loop start

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
                # population_size = population.shape[0]

                # /// 20-generational-loop start
                for generation in range(1, no_of_generations+1):
                    #  get list of all fitnesses in the population
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

                    # PARENT SELECTION
                    selected_parents = tournament_selection(
                        population, list_of_fitnesses, k_tournament_size)  # selected_parents size (5 * 256)
                    parents_size = selected_parents.shape[0]

                    # CROSSOVER - EACH INDIVIDUAL PRODUCES ONE CHILD, but also there is no crossover, only mutation...of what
                    children = crossover(
                        selected_parents, parents_size, global_population_size, global_genome_size, toolbox)

                    # MUTATION - Produce one child via mutation - SELF-ADAPTIVE THROUGH MUTATION STEP SIZE (IN META-EP)?
                    mutated_children = gaussian_mutation(
                        mutation_ratio, children, toolbox, global_population_size, global_genome_size)

                    # SURVIVAL SELECTION - PROBABILISTIC mu + mu
                    new_population = probabilistic_survival_selection(population, list_of_fitnesses,
                                                                      mutated_children, env, global_population_size, global_genome_size)
                    population = new_population
                    population_size = population.shape[0]

                    # Print stats for current generation
                    print(
                        f'-------\nGeneration {generation} stats - Best: {best_fitness_curr_gen} Mean: {avg_fitness_curr_gen} Std: {np.std(list_of_fitnesses)}')

                #  /// 20-generational-loop finished

                # save to file
                # if run == no_of_runs-1:
                #     average_fitness_pr_gen = average_fitness_pr_gen.reshape(
                #         no_of_runs, no_of_generations)
                #     best_fitness_pr_gen = best_fitness_pr_gen.reshape(
                #         no_of_runs, no_of_generations)
                #     best_inds_pr_gen = best_inds_pr_gen.reshape(
                #         no_of_runs, genome_size)

                # f = open("./logs/µcommaλaverage_fitnesses_pr_run" +
                #          str(run+1)+"_enemy_group_"+str(enemy+1)+".csv", "a")
                # g = open("./logs/µcommaλbest_fitnesses_pr_run" +
                #          str(run+1)+"_enemy_group_"+str(enemy+1)+".csv", "a")
                # h = open("./logs/µcommaλbest_individuals_pr_run" +
                #          str(run+1)+"_enemy_group_"+str(enemy+1)+".csv", "a")
                # np.savetxt(f, average_fitness_pr_gen, delimiter=',')
                # np.savetxt(g, best_fitness_pr_gen, delimiter=',')
                # np.savetxt(h, best_inds_pr_gen, delimiter=',')
                # f.close()
                # g.close()
                # h.close()

            # /// 10-runs-loop finished

            # /// 10-best-indivudals-test-loop start

            average_solution_per_enemyset = play_top_ten(
                enemies, no_of_runs, enemy, env)
            print('best sol pr enemy-set array', average_solution_per_enemyset)

            # /// 10-best-indivudals-test-loop finished

            # do the same thing for the next experiment

        # /// 2-EAs-loop finished

    #  /// 3-Enemies-loop finished


main_function()
