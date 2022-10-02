import numpy as np
import matplotlib.pyplot as plt

def line_plot(average_fitness_generation, best_solutions_fitness, num_gens, enemy_id, alg_nr):
    data_points_avg = np.array([])  
    data_points_max = np.array([])  
    standard_deviations_average = np.array([])
    standard_deviations_max = np.array([])

    for i in range(num_gens):
        avg_of_a_gen = np.array([])
        max_of_a_gen = np.array([])
        
        for j in range(average_fitness_generation.size - 1):
            if(j % num_gens == i):
                avg_of_a_gen = np.append(avg_of_a_gen, average_fitness_generation[j])
                max_of_a_gen = np.append(max_of_a_gen, best_solutions_fitness[j])

        data_points_avg = np.append(data_points_avg, np.average(avg_of_a_gen))
        data_points_max = np.append(data_points_max, np.average(max_of_a_gen))  

        standard_deviations_average = np.append(standard_deviations_average, np.std(avg_of_a_gen))  
        standard_deviations_max = np.append(standard_deviations_max, np.std(max_of_a_gen))  


    array_gens = np.array([i + 1 for i in range(num_gens)])  
    figure, plot = plt.subplots()
    plot.errorbar(array_gens, data_points_avg, yerr=standard_deviations_average,marker="o", color="blue", label="Means of means")
    plot.errorbar(array_gens, data_points_max, yerr=standard_deviations_max,marker="o", color="red", label="Means of maxes")
    plot.legend(loc='upper left')
    
    xint = range(num_gens + 1)
    plt.xticks(xint)  
    plt.xlabel('generation')
    plt.ylabel('average fitness')
    plt.title(f"{alg_nr}-Enemy{enemy_id}")

    plt.savefig(f"LinePlotsAlg{alg_nr}Enemy{enemy_id}.png")


def box_plot(performance1, performance2, enemy_id):
    fig, ax = plt.subplots()
    ax.boxplot([performance1, performance2], labels=["µ,λ", "µ+λ"])
    plt.ylabel("Fitness")
    ax.set_title(f"Best individuals against enemy {enemy_id}")

    plt.savefig(f"BestIndividualsBoxEnemy{enemy_id}.png")

enemies = [1,2,3]
generations = 20
filepath = './logs/'

for enemy in enemies:
    avg_comma = np.genfromtxt(filepath+'µcommaλaverage_fitnesses_pr_run9of enemy'+str(enemy)+'.csv', delimiter=",")
    best_comma = np.genfromtxt(filepath+'µcommaλbest_fitnesses_pr_run9of enemy'+str(enemy)+'.csv', delimiter=",")

    avg_plus = np.genfromtxt(filepath+'µplusλaverage_fitnesses_pr_run9of enemy'+str(enemy)+'.csv', delimiter=",")
    best_plus = np.genfromtxt(filepath+'µplusλbest_fitnesses_pr_run9of enemy'+str(enemy)+'.csv', delimiter=",")

    line_plot(avg_comma.flatten(), best_comma.flatten(), generations, enemy, 'µcommaλ')
    line_plot(avg_plus.flatten(), best_plus.flatten(), generations, enemy, 'µcommaλ')
    box_plot(best_comma.max(axis=1), best_plus.max(axis=1), enemy)

