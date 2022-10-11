from deap import base, tools
import numpy as np
from init_environment import initialize_environment
from init_population import initialize_population
from fitness import fittest_solution
from selection import round_robin_tournament_selection, tournament_selection
import pandas as pd

from testing_best_individuals import play_top_ten

# Runner 2 copy


# Initialize DEAP
toolbox = base.Toolbox()
toolbox.register("crossover", tools.cxTwoPoint)
global_genome_size = 265
mutation_ratio = 0.2
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
global_population_size = 10
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

        for EA in range(1):  # /// 2-EAs-loop start # 0 = GA, 1 = EA 
            average_fitness_pr_gen = np.array([]) 
            best_fitness_pr_gen = np.array([]) 
            best_inds_pr_gen = np.zeros((no_of_runs, global_genome_size)) # np.zeros because of the append function later on
            best_fitness = -1 # best fitness of all generations 
            for run in range(no_of_runs):  # /// 10-runs-loop start
                print(f' -------- RUN {run+1} -------- ')
                print('Initial stats: ')

                # INITIALIZE POPULATION
                # [0] = rows, [1] = columns 
                population = initialize_population(global_population_size, global_genome_size)  # 150, 256

                # population.shape[0] is the number of rows in the population array 
                population_size = population.shape[0] # 150 

                # population.shape[1] is the number of columns in the population array
                genome_size = population.shape[1]     # 256

                print("population size: ", population_size) # population.shape[0] = 150
                print("genome size: ", genome_size)         # population.shape[1] = 256

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
                        best_ind_idx = np.where(list_of_fitnesses == best_fitness)  # returns a tuple with the index of the best individual 
                        best_individual = population[best_ind_idx[0][0]] # get the best individual from the population array 
                        # save individual to list
                        best_inds_pr_gen[run] = best_individual # save the best individual of this run to the list of best individuals 

                    # add to lists of best and average fitnesses 
                    best_fitness_pr_gen = np.append(best_fitness_pr_gen, best_fitness_curr_gen)
                    average_fitness_pr_gen = np.append(average_fitness_pr_gen, avg_fitness_curr_gen)

                    # PARENT SELECTION - DETERMINISTIC SELECTION, WHERE EACH INDIVIDUAL PRUDCES ONE CHILD VIA MUTATION
                    
                    selected_parents = tournament_selection(population, list_of_fitnesses, k_tournament_size)  # selected_parents size (5 * 256)
                    parents_size = selected_parents.shape[0]

                    # CROSSOVER - Each INDIVIDUAL PRODUCES ONE CHILD, but also there is no crossover, only mutation...of what
                    children = np.array([])

                    for parents in range(parents_size): # /// 5-parents-loop start
                        parent_ind_1 = selected_parents[np.random.randint(len(selected_parents))]
                        print(f"parent_ind_1, {parent_ind_1}")
                        parent_ind_2 = selected_parents[np.random.randint(len(selected_parents))]
                        print(f"parent_ind_2, {parent_ind_2}")

                        ind1, ind2 = toolbox.crossover(parent_ind_1, parent_ind_2)
                        children = np.append(children, ind1) # append the child to the children array  
                        children = np.append(children, ind2) # append the child to the children array 
                        # /// 5-parents-loop end 

                    children = children.reshape(population_size, genome_size)

                    # MUTATION - Produce one child via mutation - SELF-ADAPTIVE THROUGH MUTATION STEP SIZE (IN META-EP)
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

                    # SURVIVAL SELECTION - PROBABILISTIC mu + mu - this is the round robin

                    print('population shape',
                          population.shape[0], population.shape[1])

                    population = np.append(
                        population, mutated_children, axis=1)
                    print('population + children',
                          population.shape[0], population.shape[1])

                    # ROUND ROBIN TOURNAMENT SELECTION
                    print('round robin time')
                    round_robin_scores = np.array([])
                    individual_ids = np.arange(population_size)
                    for individual_id in range(population_size):
                        individual_score = round_robin_tournament_selection(
                            individual_id, list_of_fitnesses)
                        round_robin_scores = np.append(
                            round_robin_scores, individual_score)

                    # dict of indivdual_id - number of wins
                    id_score_dict = dict(
                        zip(individual_ids, round_robin_scores))

                    # sort dict by value - reverse sorted for some reason
                    dict_sort_by_scores_ascending = sorted(
                        id_score_dict.items(), key=lambda item: item[1])

                    # sort dict the correct way by scores
                    dtype = [('id', int), ('score', int)]
                    dict_sort_by_scores_descending = np.array(
                        dict_sort_by_scores_ascending, dtype=dtype)[::-1]

                    # collect the ids of the winners from the competition (keys)
                    round_robin_winners_ids = np.array([])
                    for i in range(len(population)):  # population includes children
                        round_robin_winners_ids = np.append(
                            round_robin_winners_ids, int(dict_sort_by_scores_descending[i][0]))

                    # get top n winner ids, where n is original population size
                    top_n_winners = round_robin_winners_ids[:population_size]

                    # get top n winners from population, and create new population.
                    # new population size should be same as oriignal population size
                    new_population = np.array([])
                    print('length top n winners', len(top_n_winners))
                    for i in range(len(top_n_winners)):
                        new_population = np.append(
                            new_population, population[int(top_n_winners[i])])

                    population = new_population  # population size cannot be reshaped currently
                    population = population.reshape(population_size, genome_size)
                    print('POP SHAPE', population.shape)
                    # print('new popoulation',population.shape[0], population.shape[1])

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

                f = open("./logs/µcommaλaverage_fitnesses_pr_run" +
                         str(run+1)+"_enemy_group_"+str(enemy+1)+".csv", "a")
                g = open("./logs/µcommaλbest_fitnesses_pr_run" +
                         str(run+1)+"_enemy_group_"+str(enemy+1)+".csv", "a")
                h = open("./logs/µcommaλbest_individuals_pr_run" +
                         str(run+1)+"_enemy_group_"+str(enemy+1)+".csv", "a")
                np.savetxt(f, average_fitness_pr_gen, delimiter=',')
                np.savetxt(g, best_fitness_pr_gen, delimiter=',')
                np.savetxt(h, best_inds_pr_gen, delimiter=',')
                f.close()
                g.close()
                h.close()

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
