import pandas as pd
import numpy as np
from CCA import continuous_CCA, greedy_CCA, CCA, multistart_CCA, benders_CCA, kernelsearch_CCA, combined_CCA, initializing
from CCA import pareto_frontier

mode = 1

dataset1, dataset2, ks = initializing(1)

pareto_frontier(dataset1, dataset2, k1_min=1, k1_max=6, k2_min=1, k2_max=5, name_train="paretowine_train", name_test="paretowine_test")
