import pandas as pd
import numpy as np
from CCA import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, initializing
from CCA import pareto_frontier

mode = 1

dataset1, dataset2, ks = initializing(3)

k1, k2 = ks[0]

objective, w1, w2, time_elapsed = greedy_CCA(dataset1, dataset2, k1, k2)

# CCA(dataset1, dataset2, k1, k2, time_limit=7200)

# objective, w1, w2, time_elapsed = kernelsearch_CCA(dataset1, dataset2, k1, k2, NB = 10, time_limit=100)

# continuous_CCA(dataset1, dataset2, k1, k2, NB=k1 * k2, time_limit=1000, best_response=False)


# multistart_CCA(dataset1=dataset1, dataset2=dataset2, k1=k1, k2=k2, max_iter=20, time_limit=300, best_reponse=True)
# pareto_frontier(dataset1, dataset2, k1_min=1, k1_max=34, k2_min=1, k2_max=34)

