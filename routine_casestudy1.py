from CCA_JP import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, pareto_frontier, initializing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Cota superior de tiempo:
# 6 datasets, 14/6 ks, 9 methods, 5 training sets, 1000 sec per combination

results = pd.DataFrame(columns=['dataset', 'method', 'k1', 'k2', 'ObjVal_Train', 'ObjVal_Test', 'Time_Elapsed', 'N_Iter'])

# it = 9

# results = pd.read_csv('results/results.csv').iloc[:, 1:]
# it = results.shape[0]
dna1 = pd.read_csv('datasets/dna_scaled1.csv').iloc[:, 1:]
# rna = pd.read_csv('datasets/rna_scaled.csv').iloc[:, 1:]
rna1 = pd.read_csv('datasets/rna_scaled1.csv').iloc[:, 1:]

genechr = np.array(pd.read_csv('datasets/genechr.csv'))

# print(genechr[[18590, 18629]])

results = pd.read_csv('results/results_chrom1.csv').iloc[:, 1:]

start = True

n_rows = -1

if start:
    n_rows = results.shape[0]

# results = pd.DataFrame(columns=['dataset', 'method', 'k1', 'k2', 'ObjVal_Train', 'ObjVal_Test', 'Time_Elapsed', 'N_Iter'])

tmax = 7200
mode = 7
# training_rna1, test_rna1 = train_test_split(rna, test_size=0.25, random_state=i)
ks = [(5, 5), (10, 10), (20, 20)]

total_iterations = 0

for k1, k2 in ks:


    # Greedy_CCA:
    print('****************************')
    print('Solving using the Greedy_CCA')
    print('****************************')

    for i in range(5):
        
        if total_iterations > n_rows:
            
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.25, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.25, random_state=i)
    
            objval_train, w1, w2, time_elapsed = greedy_CCA(training_rna1, training_dna1, k1, k2)
        
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
    
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)
        
            serie = pd.Series([mode, "greedy_CCA", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)
    
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')
        
        total_iterations += 1
    
    
            
    # CCA:
    print('*******************************************')
    print('Solving using the CCA without Best Response')
    print('*******************************************')


    for i in range(5):
        
        if total_iterations > n_rows:
            
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
    
            objval_train, w1, w2, time_elapsed, n_iter = CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, best_response=False, bigM_estimation = True)
    
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
    
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)
        
            serie = pd.Series([mode, "CCA_without", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)
        
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')
            
        total_iterations += 1
            
        
    # CCA:
    print('****************************************')
    print('Solving using the CCA with Best Response')
    print('****************************************')


    for i in range(5):
        
        if total_iterations > n_rows:
            
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
    
            objval_train, w1, w2, time_elapsed, n_iter = CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, init = 0, best_response=True, bigM_estimation = True)
    
            
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
    
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)
    
    
            serie = pd.Series([mode, "CCA_with", k1, k2, objval_train, objval_test, time_elapsed, n_iter], index=results.columns)
        
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')
        
        total_iterations += 1
            
    
    
    # multistart_CCA:
    print('***************************************************')
    print('Solving using the multistart_CCA with Best Response')
    print('***************************************************')

    for i in range(5):
        
        if total_iterations > n_rows:
            
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
    
            objval_train, w1, w2, time_elapsed = multistart_CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, best_response=True, bigM_estimation = True)
    
            
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
    
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)
    
    
            serie = pd.Series([mode, "multistart_CCA", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)
        
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')
        
        total_iterations += 1


    # benders_CCA:
    print('******************************************************************')
    print('Solving using the benders_CCA without Best Response Initialization')
    print('******************************************************************')


    for i in range(5):
        
        if total_iterations > n_rows:
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
    
            objval_train, w1, w2, time_elapsed, n_cut = benders_CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, init = False, bigM_estimation = True)
    
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
    
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)
        
            serie = pd.Series([mode, "benders_CCA_without", k1, k2, objval_train, objval_test, time_elapsed, n_cut], index=results.columns)
    
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')
            
        total_iterations += 1



    print('***************************************************************')
    print('Solving using the benders_CCA with Best Response Initialization')
    print('***************************************************************')

    

    for i in range(5):
        
        if total_iterations > n_rows:
            
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
    
            objval_train, w1, w2, time_elapsed, n_cut = benders_CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, init = True, bigM_estimation = True)
    
    
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
    
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)

            serie = pd.Series([mode, "benders_CCA_with", k1, k2, objval_train, objval_test, time_elapsed, n_cut], index=results.columns)
        
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')

        total_iterations += 1

    print('********************************************************')
    print('Solving using the kernelsearch_CCA without Best Response')
    print('********************************************************')

    for i in range(5):
        
        if total_iterations > n_rows:
            
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
            
            objval_train, w1, w2, time_elapsed = kernelsearch_CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, NB = k1*k2, best_response = False, bigM_estimation = True)
            
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
            
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)     
            
            serie = pd.Series([mode, "kernelsearch_CCA_without", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)
            
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')

        total_iterations += 1

    print('*****************************************************')
    print('Solving using the kernelsearch_CCA with Best Response')
    print('*****************************************************')

    
    for i in range(5):
        
        if total_iterations > n_rows:
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
            
            objval_train, w1, w2, time_elapsed = kernelsearch_CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, NB = k1*k2, best_response = True, bigM_estimation = True)
                
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
            
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)
            
            serie = pd.Series([mode, "kernelsearch_CCA_with", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)
            
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')
        
        total_iterations += 1
        
    print('**********************************************************')
    print('Solving using the combined kernelsearch with Best Response')
    print('**********************************************************')

    for i in range(5):
        
        if total_iterations > n_rows:
            
            training_rna1, test_rna1 = train_test_split(rna1, test_size=0.3, random_state=i)
            training_dna1, test_dna1 = train_test_split(dna1, test_size=0.3, random_state=i)
            
            objval_train, w1, w2, time_elapsed = combined_CCA(training_rna1, training_dna1, k1, k2, time_limit=tmax, NB = k1*k2, bigM_estimation = True)
            
            norm1 = np.sqrt(w1.T @ test_rna1.T @ test_rna1 @ w1)
            norm2 = np.sqrt(w2.T @ test_dna1.T @ test_dna1 @ w2)
            
            objval_test = (w1.T @ test_rna1.T @ test_dna1 @ w2) / (norm1 * norm2)
            
            serie = pd.Series([mode, "combined_CCA", k1, k2, objval_train, objval_test, time_elapsed, -1], index=results.columns)
            
            results = results.append(serie, ignore_index=True)
            results.to_csv('./results/results_chrom1.csv')
            
        total_iterations += 1