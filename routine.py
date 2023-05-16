from CCA import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, pareto_frontier, initializing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Cota superior de tiempo:
# 6 datasets, 14/6 ks, 9 methods, 5 training sets, 1000 sec per combination

results = pd.DataFrame(columns=['dataset', 'scaling', 'method', 'k1', 'k2', 'ObjVal_Train', 'ObjVal_Test', 'Time_Elapsed', 'N_Iter'])

# it = 9

start = True
n_rows = -1

if start:
    results = pd.read_csv('results/results_new2.csv').iloc[:, 1:]
    n_rows = results.shape[0]

total_iterations = 0
time_limit = 1000

for mode in range(5):

    for scaling in [False, True]:
        dataset1, dataset2, ks = initializing(mode, scaling)

        for k1, k2 in ks:

            # Greedy_CCA:
            print('****************************')
            print('Solving using the Greedy_CCA')
            print('****************************')

            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed = greedy_CCA(training_dataset1, training_dataset2, k1, k2)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "greedy_CCA", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            # CCA:
            print('*******************************************')
            print('Solving using the CCA without Best Response')
            print('*******************************************')


            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed, n_iter = CCA(training_dataset1, training_dataset2, k1, k2, time_limit = time_limit, best_response=False)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "CCA_without", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            # CCA:
            print('****************************************')
            print('Solving using the CCA with Best Response')
            print('****************************************')



            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed, n_iter = CCA(training_dataset1, training_dataset2, k1, k2, time_limit = time_limit, best_response=True)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "CCA_with", k1, k2, objval_train, objval_test, time_elapsed, n_iter], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            # multistart_CCA:
            print('***************************************************')
            print('Solving using the multistart_CCA with Best Response')
            print('***************************************************')


            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed = multistart_CCA(training_dataset1, training_dataset2, k1, k2, time_limit = time_limit, best_response=True)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "multistart_CCA", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            # benders_CCA:
            print('******************************************************************')
            print('Solving using the benders_CCA without Best Response Initialization')
            print('******************************************************************')



            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed, n_cut = benders_CCA(training_dataset1, training_dataset2, k1, k2, time_limit = time_limit, init = False)


                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "benders_CCA_without", k1, k2, objval_train, objval_test, time_elapsed, n_cut], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            print('***************************************************************')
            print('Solving using the benders_CCA with Best Response Initialization')
            print('***************************************************************')



            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed, n_cut = benders_CCA(training_dataset1, training_dataset2, k1, k2, time_limit = time_limit, init = True)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "benders_CCA_with", k1, k2, objval_train, objval_test, time_elapsed, n_cut], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            print('********************************************************')
            print('Solving using the kernelsearch_CCA without Best Response')
            print('********************************************************')


            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed = kernelsearch_CCA(training_dataset1, training_dataset2, k1, k2, NB = k1*k2, best_response = False)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "kernelsearch_CCA_without", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            print('*****************************************************')
            print('Solving using the kernelsearch_CCA with Best Response')
            print('*****************************************************')


            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed = kernelsearch_CCA(training_dataset1, training_dataset2, k1, k2, NB = k1*k2, time_limit = time_limit, best_response = True)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "kernelsearch_CCA_with", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1

            print('**********************************************************')
            print('Solving using the combined kernelsearch with Best Response')
            print('**********************************************************')


            for i in range(5):
                if total_iterations > n_rows:

                    training_dataset1, test_dataset1 = train_test_split(dataset1, test_size=0.3, random_state=i)
                    training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.3, random_state=i)

                    objval_train, w1, w2, time_elapsed = combined_CCA(training_dataset1, training_dataset2, k1, k2, NB = k1*k2, time_limit = time_limit)

                    norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                    norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                    objval_test = (w1.T @ test_dataset1.T @ test_dataset2 @ w2) / (norm1 * norm2)

                    serie = pd.Series([mode, scaling, "combined_CCA", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)

                    results = results.append(serie, ignore_index=True)
                    results.to_csv('./results/results_new2.csv')

                total_iterations += 1


            print(total_iterations)