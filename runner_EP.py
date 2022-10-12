from deap import base, tools
import numpy as np
from crossover import uniformCrossover, twoPointCrossover, adaptiveCrossover
from init_environment import initialize_environment
from init_population import initialize_population
from fitness import fittest_solution
from mutation import add_sigma_to_individual, deterministic_gaussian_mutation, get_sigma, probabilistic_gaussian_mutation
from selection import probabilistic_survival_selection, round_robin_tournament_selection, tournament_selection
import pandas as pd

from testing_best_individuals import play_top_ten

# Runner 2 copy


# Initialize DEAP
toolbox = base.Toolbox()
toolbox.register("mutate", tools.mutGaussian, mu=0,
                 sigma=1, indpb=0.1)

global_population_size = 10
global_genome_size = 265  # 265 + 1
no_of_runs = 1
no_of_generations = 2
sigma = 0.5
EA_no = 1  # EA 1 or 2

np.random.seed(420)  # why 420? copied form optimization_generalist_demo.py


def main_function():
    # array to save solutions per set of enemies / per experiment
    average_solution_per_enemyset = np.array([])
    enemies = np.array([[6, 4], [3, 7]])  # our enemy sets

    for enemygroup in range(1, 3):  # /// 3-Enemies-loop start
        print(
            f'\n------- Training against enemy group no {enemygroup}: enemies {enemies[enemygroup-1]}')

        # INITIALIZE ENVIRONMENT with sets of enemies
        env = initialize_environment(enemies[enemygroup-1])

        # arrays to save stats throughout the runs
        average_fitness_pr_gen = np.array([])
        best_fitness_pr_gen = np.array([])
        best_inds_pr_gen = np.zeros((no_of_runs, global_genome_size))

        best_fitness = -1  # initial best fitness

        for run in range(no_of_runs):  # /// 10-runs-loop start
            print(f' -------- RUN {run+1} -------- ')

            # INITIALIZE POPULATION
            population = initialize_population(
                global_population_size, global_genome_size)  # 150, 265

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

                # PARENT SELECTION - none, all individuals are parents

                # CROSSOVER - no crossover, only mutation of individuals

                # sigma which gets smaller over the generations
                step_size = get_sigma(generation, no_of_generations)

                # we add sigma at the end of the genome (replacing the original last value in the genome)
                population = add_sigma_to_individual(
                    step_size, population, global_population_size, global_genome_size)

                # MUTATION - each individual creates one child through mutation
                mutated_children = deterministic_gaussian_mutation(
                    population, step_size, global_population_size, global_genome_size)

                # SURVIVAL SELECTION - round-robin tournament
                new_population = probabilistic_survival_selection(population, list_of_fitnesses,
                                                                  mutated_children, env, global_population_size, global_genome_size)
                population = new_population

                # Print stats for current generation
                print(
                    f'-------\nGeneration {generation} stats - Best: {best_fitness_curr_gen} Mean: {avg_fitness_curr_gen} Std: {np.std(list_of_fitnesses)}')

            #  /// 20-generational-loop finished

            # save stats to files
            if run == no_of_runs-1:
                average_fitness_pr_gen = average_fitness_pr_gen.reshape(
                    no_of_runs, no_of_generations)
                best_fitness_pr_gen = best_fitness_pr_gen.reshape(
                    no_of_runs, no_of_generations)
                best_inds_pr_gen = best_inds_pr_gen.reshape(
                    no_of_runs, global_genome_size)

            f = open(
                f"./logs/EP_{EA_no}_average_fitnesses_pr_run{run+1}_enemy_group_{enemygroup}.csv", "a")
            g = open(
                f"./logs/EP_{EA_no}_best_fitnesses_pr_run{run+1}_enemy_group_{enemygroup}.csv", "a")
            h = open(
                f"./logs/EP_{EA_no}_best_individuals_pr_run{run+1}_enemy_group_{enemygroup}.csv", "a")
            np.savetxt(f, average_fitness_pr_gen, delimiter=',')
            np.savetxt(g, best_fitness_pr_gen, delimiter=',')
            np.savetxt(h, best_inds_pr_gen, delimiter=',')
            f.close()
            g.close()
            h.close()

        # /// 10-runs-loop finished

        # /// 10-best-indivudals-test-loop start
        average_solution_per_enemyset = play_top_ten(
            enemies, no_of_runs, env, enemygroup, EA_no)
        print('best sol pr enemy-set array', average_solution_per_enemyset)

        # /// 10-best-indivudals-test-loop finished

        # do the same thing for the next experiment

    #  /// 3-Enemies-loop finished


main_function()
