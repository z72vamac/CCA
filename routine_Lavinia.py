from CCA import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, pareto_frontier, initializing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Cota superior de tiempo:
# 6 datasets, 14/6 ks, 9 methods, 5 training sets, 1000 sec per combination

results = pd.DataFrame(columns=['dataset', 'method', 'k1', 'k2', 'ObjVal_Train', 'ObjVal_Test', 'Time_Elapsed', 'N_Iter'])

# it = 9

# results = pd.read_csv('results/results.csv').iloc[:, 1:]
# it = results.shape[0]
dna = pd.read_csv('datasets/winequalityred_scaled.csv').iloc[:, 1:]
# rna = pd.read_csv('datasets/rna_scaled.csv').iloc[:, 1:]
# rna1 = pd.read_csv('datasets/rna_scaled1.csv').iloc[:, 1:]

genechr = np.array(pd.read_csv('datasets/genechr.csv'))

# print(genechr[[18590, 18629]])

results = pd.DataFrame(columns=['dataset', 'method', 'k1', 'k2', 'ObjVal_Train', 'ObjVal_Test', 'Time_Elapsed', 'N_Iter'])

dataset1, dataset2, ks = initializing(1)

i = 0
tmax = 7200
mode = 7
# dataset1, test_rna1 = train_test_split(results, test_size=0.25, random_state=i)
ks = [(1, 1)]



for k1, k2 in ks:

    # Greedy_CCA:
    print('****************************')
    print('Solving using the Greedy_CCA')
    print('****************************')


    objval_train, w1, w2, time_elapsed = greedy_CCA(dataset1, dataset2, k1, k2)

    objvals.append(objval_train)
    times.append(time_elapsed)

    ws_1.append(w1)
    ws_2.append(w2)

    np.savetxt("w1s.csv", ws_1, delimiter=",")
    np.savetxt("w2s.csv", ws_2, delimiter=",")

    serie = pd.Series([mode, "greedy_CCA", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), -1], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')

    # CCA:
    print('*******************************************')
    print('Solving using the CCA without Best Response')
    print('*******************************************')



    
        
        

        objval_train, w1, w2, time_elapsed, n_iter = CCA(dataset1, dataset2, k1, k2, time_limit=tmax, best_response=False, bigM_estimation = False)

        objvals.append(objval_train)
        times.append(time_elapsed)

        norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
        norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)

        objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)

        objvals_test.append(objval_test)

    serie = pd.Series([mode, "CCA_without", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), -1], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')

    # CCA:
    print('****************************************')
    print('Solving using the CCA with Best Response')
    print('****************************************')


    n_iters = []

    
        
        

        objval_train, w1, w2, time_elapsed, n_iter = CCA(dataset1, dataset2, k1, k2, time_limit=tmax, best_response=True, bigM_estimation = False)

        objvals.append(objval_train)
        times.append(time_elapsed)
        n_iters.append(n_iter)

        norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
        norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)

        objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)

        objvals_test.append(objval_test)

    serie = pd.Series([mode, "CCA_with", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), np.mean(n_iters)], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')

    # multistart_CCA:
    print('***************************************************')
    print('Solving using the multistart_CCA with Best Response')
    print('***************************************************')



    
        
        

        objval_train, w1, w2, time_elapsed = multistart_CCA(dataset1, dataset2, k1, k2, time_limit=tmax, best_response=True, bigM_estimation = False)

        objvals.append(objval_train)
        times.append(time_elapsed)

        norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
        norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)

        objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)

        objvals_test.append(objval_test)

    serie = pd.Series([mode, "multistart_CCA", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), -1], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')


    # benders_CCA:
    print('******************************************************************')
    print('Solving using the benders_CCA without Best Response Initialization')
    print('******************************************************************')


    n_cuts = []

    
        
        

        objval_train, w1, w2, time_elapsed, n_cut = benders_CCA(dataset1, dataset2, k1, k2, time_limit=tmax, init = False, bigM_estimation = False)



    serie = pd.Series([mode, "benders_CCA_without", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), np.mean(n_cuts)], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')


    print('***************************************************************')
    print('Solving using the benders_CCA with Best Response Initialization')
    print('***************************************************************')


    n_cuts = []

    
        
        

        objval_train, w1, w2, time_elapsed, n_cut = benders_CCA(dataset1, dataset2, k1, k2, time_limit=tmax, init = True, bigM_estimation = False)



    serie = pd.Series([mode, "benders_CCA_with", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), np.mean(n_cuts)], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')

    print('********************************************************')
    print('Solving using the kernelsearch_CCA without Best Response')
    print('********************************************************')



    
        
        

        objval_train, w1, w2, time_elapsed = kernelsearch_CCA(dataset1, dataset2, k1, k2, time_limit=tmax, NB = k1*k2, best_response = False, bigM_estimation = False)

        objvals.append(objval_train)
        times.append(time_elapsed)

        norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
        norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)

        objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)

        objvals_test.append(objval_test)

    serie = pd.Series([mode, "kernelsearch_CCA_without", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), -1], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')

    print('*****************************************************')
    print('Solving using the kernelsearch_CCA with Best Response')
    print('*****************************************************')



    
        
        

        objval_train, w1, w2, time_elapsed = kernelsearch_CCA(dataset1, dataset2, k1, k2, time_limit=tmax, NB = k1*k2, best_response = True, bigM_estimation = False)

        objvals.append(objval_train)
        times.append(time_elapsed)

        norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
        norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)

        objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)

        objvals_test.append(objval_test)

    serie = pd.Series([mode, "kernelsearch_CCA_with", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), -1], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')

    print('**********************************************************')
    print('Solving using the combined kernelsearch with Best Response')
    print('**********************************************************')



    

        objval_train, w1, w2, time_elapsed = combined_CCA(dataset1, dataset2, k1, k2, time_limit=tmax, NB = k1*k2, bigM_estimation = False)

        objvals.append(objval_train)
        times.append(time_elapsed)

        norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
        norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)

        objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)

        objvals_test.append(objval_test)

    serie = pd.Series([mode, "combined_CCA", k1, k2, np.mean(objvals), np.mean(objvals_test), np.mean(times), -1], index=results.columns)

    results = results.append(serie, ignore_index=True)
    results.to_csv('./results/results_chrom1.csv')