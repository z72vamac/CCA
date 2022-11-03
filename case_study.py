import pandas as pd
import numpy as np
from CCA_JP import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, initializing
from sklearn.model_selection import train_test_split


dna1 = pd.read_csv('datasets/dna_scaled1.csv').iloc[:, 1:]
# rna = pd.read_csv('datasets/rna_scaled.csv').iloc[:, 1:]
rna1 = pd.read_csv('datasets/rna_scaled1.csv').iloc[:, 1:]

genechr = np.array(pd.read_csv('datasets/genechr.csv'))

# print(genechr[[18590, 18629]])



i = 0

# training_dataset1, test_dataset1 = train_test_split(rna, test_size=0.25, random_state=i)
training_dataset1, test_dataset1 = train_test_split(rna1, test_size=0.25, random_state=i)

training_dataset2, test_dataset2 = train_test_split(dna1, test_size=0.25, random_state=i)

# objective, w1_sol, w2_sol, time_elapsed = greedy_CCA(training_dataset1, training_dataset2, k1 = 5, k2 = 5)
# objective, w1_sol, w2_sol, time_elapsed = kernelsearch_CCA(training_dataset1, training_dataset2, k1 = 5, k2 = 5, NB = 10,time_limit = 7200, bigM_estimation = True)
# objective, w1_sol, w2_sol, time_elapsed = combined_CCA(training_dataset1, training_dataset2, k1 = 5, k2 = 5, NB = 40,time_limit = 1000)
#objective, w1_sol, w2_sol, time_elapsed = multistart_CCA(training_dataset1, training_dataset2, k1 = 5, k2 = 5, n_iter = 10, time_limit = 7200, best_response=True, bigM_estimation = True)
objective, w1_sol, w2_sol, time_elapsed = benders_CCA(training_dataset1, training_dataset2, k1 = 5, k2 = 5, init = True, time_limit = 1000, bigM_estimation = True)
# objective, w1_sol, w2_sol, time_elapsed = CCA(training_dataset1, training_dataset2,k1=5,k2=5,time_limit=3600, bigM_estimation = True)

n, p1 = training_dataset1.shape

I = [i for i in range(p1) if w1_sol[i] != 0]

print(I)

print("To be patient: " + str(sum([genechr[i][0] == 1 for i in I])))

