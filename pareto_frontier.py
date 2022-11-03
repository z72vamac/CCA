"""
Checking how the nonzeros change when we vary the k.
"""

from CCA import CCA
import numpy as np


def pareto_frontier(dataset1, dataset2, k_min, k_max, time_limit = 1000):

    ws_1 = []
    ws_2 = []
    objvals = []

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    # training_data = df.sample(frac=0.7, random_state=1)

    objvals = []

    for k in range(k_min, k_max):
        corrs = []

        w1, w2, objval = CCA(dataset1, dataset2, k, k, time_limit = time_limit, best_reponse=True)

        ws_1.append(w1)
        ws_2.append(w2)

        objvals.append(objval)

        nonzero_w1 = np.array(ws_1) != 0
        nonzero_w2 = np.array(ws_2) != 0

        np.savetxt("nonzero_w1.csv", nonzero_w1, fmt='%d', delimiter=",")
        np.savetxt("nonzero_w2.csv", nonzero_w2, fmt='%d', delimiter=",")
        np.savetxt("objvals.csv", objvals, delimiter=",")

    return nonzero_w1, nonzero_w2, objvals

# p1 = 34
# k_min = 7
# k_max = 34
#
# df = pd.read_csv('music_scaled.csv', sep=",")
#
# nonzero_w1, nonzero_w2, objvals = pareto_frontier(df, p1, k_min, k_max)



# print(objvals)