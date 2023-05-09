import pandas as pd
import numpy as np
from CCA import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, initializing, restricted_CCA
from CCA import pareto_frontier
from sklearn.model_selection import train_test_split

mode = 1

dataset1, dataset2, ks = initializing(6, False)

training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=0)
training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=0)

k1, k2 = ks[0]

kernel_1 = [0, 1, 2]
kernel_2 = [3, 4]
bucket1 = []
bucket2 = []
objval = 0
time_limit = 100

# dataset = pd.read_csv('datasets/winequalityred_scaled.csv', sep=",")
#
# dataset1 = dataset.iloc[:, 0:6]
# dataset2 = dataset.iloc[:, 6:11]
#
# training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=0)
# training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=0)
# resultados = restricted_CCA(kernel_1, bucket1, kernel_2, bucket2, objval, training_dataset1, training_dataset2, k1, k2, time_limit,
#                    best_response=False, bigM_estimation=True)

# print(resultados)
# objective, w1, w2, time_elapsed = greedy_CCA(dataset1, dataset2, k1, k2)

objval_train, w1, w2, time_elapsed, n_iter = CCA(training_dataset1, training_dataset2, k1, k2, time_limit = time_limit, best_response=False)

norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

print(objval_test)
# objective, w1, w2, time_elapsed = kernelsearch_CCA(dataset1, dataset2, k1, k2, NB = 10, time_limit=100)

# continuous_CCA(dataset1, dataset2, k1, k2, NB=k1 * k2, time_limit=1000, best_response=False)


# multistart_CCA(dataset1=dataset1, dataset2=dataset2, k1=k1, k2=k2, max_iter=20, time_limit=300, best_reponse=True)
# pareto_frontier(dataset1, dataset2, k1_min=1, k1_max=34, k2_min=1, k2_max=34)

