from deap import base
import numpy as np
from crossover import uniformCrossover, twoPointCrossover, adaptiveCrossover
from find_individual import find_individual
from init_environment import initialize_environment
from init_population import initialize_population
from fitness import fittest_solution
from mutation import add_sigma_to_individual, get_sigma, probabilistic_gaussian_mutation
from selection import probabilistic_survival_selection, tournament_selection
from testing_best_individuals import play_top_ten

# Initialize DEAP
toolbox = base.Toolbox()

global_population_size = 150
global_genome_size = 265
no_of_runs = 10
no_of_generations = 20
mutation_ratio = 0.5
k_tournament_size = 2

np.random.seed(420)


def algorithm(EA_no: int, enemies):
    for enemygroup in range(1, 3):
        print(
            f'\n------- Training against enemy group no {enemygroup}: enemies {enemies[enemygroup-1]}')

        # INITIALIZE ENVIRONMENT with sets of enemies
        env = initialize_environment(enemies[enemygroup-1])

        # arrays to save stats throughout the runs
        average_fitness_pr_gen = np.array([])
        best_fitness_pr_gen = np.array([])
        best_inds_pr_gen = np.zeros((no_of_runs, global_genome_size))

        best_fitness = float('-inf')  # initial best fitness

        for run in range(no_of_runs):
            print(f' -------- RUN {run+1} -------- ')

            # INITIALIZE POPULATION
            population = initialize_population(
                global_population_size, global_genome_size)  # 150, 265

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

                    # get best individual and to list
                    best_inds_pr_gen[run] = find_individual(
                        list_of_fitnesses, best_fitness, population)

                # add stats to lists
                best_fitness_pr_gen = np.append(
                    best_fitness_pr_gen, best_fitness_curr_gen)
                average_fitness_pr_gen = np.append(
                    average_fitness_pr_gen, avg_fitness_curr_gen)

                # PARENT SELECTION
                selected_parents = tournament_selection(
                    population, list_of_fitnesses, k_tournament_size, global_population_size, global_genome_size)
                parents_size = selected_parents.shape[0]

                # CROSSOVER - uniform, two-point or adaptive
                if EA_no == 1:
                    children = uniformCrossover(
                        population, parents_size, global_population_size, global_genome_size, toolbox)
                if EA_no == 2:
                    children = twoPointCrossover(
                        population, parents_size, global_population_size, global_genome_size, toolbox)
                if EA_no == 3:
                    children = adaptiveCrossover(
                        population, parents_size, global_population_size, global_genome_size, toolbox)

                # sigma which gets smaller over the generations
                step_size = get_sigma(generation, no_of_generations)

                # MUTATION - probabilistic mutation: some offspring get mutated
                mutated_children = probabilistic_gaussian_mutation(
                    mutation_ratio, children, global_population_size, global_genome_size, step_size)

                # we add sigma at the end of the genome (replacing the original last value in the genome)
                mutated_children = add_sigma_to_individual(
                    step_size, mutated_children, global_population_size, global_genome_size)

                # SURVIVAL SELECTION - round-robin tournament
                new_population = probabilistic_survival_selection(population, list_of_fitnesses,
                                                                  mutated_children, env, global_population_size, global_genome_size)
                population = new_population

                # Print stats for current generation
                print(
                    f'-------\nGeneration {generation} stats - Best: {best_fitness_curr_gen} Mean: {avg_fitness_curr_gen} Std: {np.std(list_of_fitnesses)}')

            # save stats to files
            if run == no_of_runs-1:
                average_fitness_pr_gen = average_fitness_pr_gen.reshape(
                    no_of_runs, no_of_generations)
                best_fitness_pr_gen = best_fitness_pr_gen.reshape(
                    no_of_runs, no_of_generations)
                best_inds_pr_gen = best_inds_pr_gen.reshape(
                    no_of_runs, global_genome_size)

            f = open(
                f"./logs/EA_{EA_no}_avg_fitnesses_run{run+1}_enemygr{enemygroup}.csv", "a")
            g = open(
                f"./logs/EA_{EA_no}_best_fitnesses_run{run+1}_enemygr{enemygroup}.csv", "a")
            h = open(
                f"./logs/EA_{EA_no}_best_individuals_run{run+1}_enemygr{enemygroup}.csv", "a")
            np.savetxt(f, average_fitness_pr_gen, delimiter=',')
            np.savetxt(g, best_fitness_pr_gen, delimiter=',')
            np.savetxt(h, best_inds_pr_gen, delimiter=',')
            f.close()
            g.close()
            h.close()

        # test against all enemies
        best_fitness = play_top_ten(
            no_of_runs, env, enemygroup, EA_no)

        print(
            f'best fitness trained on enemy set {enemies[enemygroup-1]}, EA {EA_no}: {best_fitness}')
