"""
Exporting the datasets to check in R
"""

from CCA import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, \
    pareto_frontier, initializing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

start = False

n_rows = -1

if start:
    n_rows = results.shape[0]

total_iterations = 0
time_limit = 7200


for mode in range(6, 10):
    dataset1, dataset2, ks = initializing(mode)

    # for k1, k2 in ks:

    # Greedy_CCA:
    print('****************************')
    print('Solving using the Greedy_CCA')
    print('****************************')

    for i in range(5):
        if total_iterations > n_rows:
            training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
            training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

            training_dataset1.to_csv('./datasets/r_datasets/training_dataset1_mode_' + str(mode) + '_iteration_' + str(i) + '.csv')
            training_dataset2.to_csv('./datasets/r_datasets/training_dataset2_mode_' + str(mode) + '_iteration_' + str(i) + '.csv')

            test_dataset1.to_csv('./datasets/r_datasets/test_dataset1_mode_' + str(mode) + '_iteration_' + str(i) + '.csv')
            test_dataset2.to_csv('./datasets/r_datasets/test_dataset2_mode_' + str(mode) + '_iteration_' + str(i) + '.csv')
            # objval_train, w1, w2, time_elapsed = greedy_CCA(training_dataset1, training_dataset2, k1, k2)
            #
            # norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
            # norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)
            #
            # objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)
            #
            # serie = pd.Series([mode, "greedy_CCA", k1, k2, objval_train, objval_test, time_elapsed, -1],
            #                   index=results.columns)
            #
            # results = results.append(serie, ignore_index=True)
            # results.to_csv('./results/results.csv')

        total_iterations += 1
