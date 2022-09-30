from unittest import BaseTestSuite
import pandas as pd
best_individuals_csv = pd.read_csv('logs/best_individuals_pr_run.csv')
# best_individuals = best_individuals_csv['']

print(best_individuals_csv.head())

arr = best_individuals_csv.to_numpy()
print(arr)

arr = arr.reshape(round(len(arr)/3), round(len(arr)/2))
print(arr)
