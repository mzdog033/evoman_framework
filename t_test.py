import scipy.stats as stats
import numpy as np
 
# Creating data groups
filepath = './logs_final/Sam/'
data_group1 = np.genfromtxt(filepath+'comma/logs/µcommaλaverage_fitnesses_pr_run9.csv', delimiter=",")
data_group2 = np.genfromtxt(filepath+'plus/logs/µplusλaverage_fitnesses_pr_run9.csv', delimiter=",")

# Uncomment the below lines to take the t test of the best fitness per generation of two survivor selection methods.
# data_group1 = np.genfromtxt(filepath+'comma/logs/µcommaλbest_fitnesses_pr_run9.csv', delimiter=",")
# data_group2 = np.genfromtxt(filepath+'plus/logs/µplusλbest_fitnesses_pr_run9.csv', delimiter=",")
# data_group1 = data_group1.max(axis=1)
# data_group2 = data_group2.max(axis=1)

data_group1 = data_group1.flatten()
data_group2 = data_group2.flatten()

print(stats.ttest_ind(a=data_group1, b=data_group2, equal_var=True))