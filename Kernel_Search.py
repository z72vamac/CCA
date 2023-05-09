#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from sklearn.cross_decomposition import CCA

from sklearn.preprocessing import StandardScaler

# In[6]:




# def CCA(dataset1, dataset2, k1, k2, continuous = False):
#
#     M = max([1/np.sqrt(min(LA.eig(dataset1.T @ dataset1)[0])), 1/np.sqrt(min(LA.eig(dataset2.T @ dataset2)[0]))])
#
#     n, p1 = dataset1.shape
#     n, p2 = dataset2.shape
#
#     X1 = np.array(dataset1)
#     X2 = np.array(dataset2)
#
#     product12 = X1.T @ X2
#     product11 = X1.T @ X1
#     product22 = X2.T @ X2
#
#     def fixing_w1(w1, continuous):
#         print("\nSolving the model fixing w1")
#
#         MODEL = gp.Model("Fixing w1")
#
#         w2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w2")
#
#         if continuous:
#             z2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, name="z2")
#         else:
#             z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")
#
#         MODEL.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in range(p2) for j in range(p2)) <= 1)
#
#         MODEL.addConstr(z2.sum('*') <= k2)
#
#         MODEL.addConstrs(w2[j] <= M * z2[j] for j in range(p2))
#         MODEL.addConstrs(w2[j] >= -M * z2[j] for j in range(p2))
#
#         MODEL.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)), GRB.MAXIMIZE)
#
#         MODEL.Params.OutputFlag = 0
#         MODEL.Params.MIPGap = 5e-2
#
#         MODEL.optimize()
#
#
#         w2_sol = np.zeros(p2)
#         z2_sol = np.zeros(p2)
#
#         if continuous:
#             for j in range(p2):
#                 w2_sol[j] = w2[j].X
#                 z2_sol[j] = z2[j].X
#         else:
#             for j in range(p2):
#                 if z2[j].X > 0.5:
#                     w2_sol[j] = w2[j].X
#                     z2_sol[j] = 1
#                 else:
#                     w2_sol[j] = 0
#                     z2_sol[j] = 0
#
#         print("Objective Value: " + str(MODEL.ObjVal))
#
#         return MODEL.ObjVal, w2_sol, z2_sol
#
#     def fixing_w2(w2, continuous):
#         print("\nSolving the model fixing w2")
#
#         MODEL = gp.Model("Fixing w2")
#
#         w1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w1")
#
#         if continuous:
#             z1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, name="z1")
#         else:
#             z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")
#
#         MODEL.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in range(p1) for j in range(p1)) <= 1)
#
#         MODEL.addConstr(z1.sum('*') <= k1)
#
#         MODEL.addConstrs(w1[i] <= M * z1[i] for i in range(p1))
#         MODEL.addConstrs(w1[i] >= -M * z1[i] for i in range(p1))
#
#         MODEL.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)), GRB.MAXIMIZE)
#
#         MODEL.Params.OutputFlag = 0
#         MODEL.Params.MIPGap = 5e-2
#
#         MODEL.optimize()
#
#         w1_sol = np.zeros(p1)
#         z1_sol = np.zeros(p1)
#
#         if continuous:
#             for i in range(p1):
#                 w1_sol[i] = w1[i].X
#                 z1_sol[i] = z1[i].X
#         else:
#             for i in range(p1):
#                 if z1[i].X > 0.5:
#                     w1_sol[i] = w1[i].X
#                     z1_sol[i] = 1
#                 else:
#                     w1_sol[i] = 0
#                     z1_sol[i] = 0
#
#         print("Objective Value: " + str(MODEL.ObjVal))
#
#         return MODEL.ObjVal, w1_sol, z1_sol
#
#     MODEL = gp.Model("CCA_Continuous")
#
#     w1 = MODEL.addMVar(p1, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w1")
#
#     if continuous:
#         z1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, name="z1")
#     else:
#         z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")
#
#
#     w2 = MODEL.addMVar(p2, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w2")
#
#     if continuous:
#         z2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, name="z2")
#     else:
#         z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")
#
#
#     MODEL.addConstr(w1 @ product11 @ w1 <= 1.0)
#     MODEL.addConstr(w2 @ product22 @ w2 <= 1.0)
#
#     MODEL.addConstr(z1.sum('*') <= k1)
#     MODEL.addConstr(z2.sum('*') <= k2)
#
#     MODEL.addConstrs(w1[i] <= M*z1[i] for i in range(p1))
#     MODEL.addConstrs(w1[i] >= -M*z1[i] for i in range(p1))
#
#     MODEL.addConstrs(w2[j] <= M*z2[j] for j in range(p2))
#     MODEL.addConstrs(w2[j] >= -M*z2[j] for j in range(p2))
#
#     MODEL.setObjective(w1 @ product12 @ w2, GRB.MAXIMIZE)
#
#     MODEL.Params.TimeLimit = 30
#     MODEL.Params.NonConvex = 2
#     MODEL.optimize()
#
#     obj_old = 0
#     obj_new = MODEL.ObjVal
#
#     objectives = []
#
#     w1 = np.array(w1.X)
#
#     it = 0
#
#     w1s = []
#     w2s = []
#
#     while abs(obj_old - obj_new) >= 1e-3:
#
#         obj_old = obj_new
#         obj_new, w2, z2 = fixing_w1(w1, continuous)
#         obj_new, w1, z1 = fixing_w2(w2, continuous)
#         objectives.append(obj_new)
#         w1s.append(w1)
#         w2s.append(w2)
#         it += 1
#
#     w1 = w1s[np.argmax(objectives)]
#     w2 = w2s[np.argmax(objectives)]
#
#     return w1, w2, max(objectives)

def CCA_restricted(kernel_1, bucket1, kernel_2, bucket2, dataset1, dataset2, k1, k2, objval, time_limit, best_response=False):

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1
    product22 = X2.T @ X2

    def fixing_w1(w1):
        print("\nSolving the model fixing w1")

        MODEL = gp.Model("Fixing w1")

        w2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w2")

        z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")

        MODEL.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in range(p2) for j in range(p2)) <= 1)

        MODEL.addConstr(z2.sum('*') <= k2)

        MODEL.addConstrs(w2[j] <= M * z2[j] for j in range(p2))
        MODEL.addConstrs(w2[j] >= -M * z2[j] for j in range(p2))

        MODEL.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)),
                           GRB.MAXIMIZE)

        MODEL.Params.OutputFlag = 0
        MODEL.Params.MIPGap = 5e-2

        MODEL.optimize()

        w2_sol = np.zeros(p2)
        z2_sol = np.zeros(p2)

        for j in range(p2):
            if z2[j].X > 0.5:
                w2_sol[j] = w2[j].X
                z2_sol[j] = 1
            else:
                w2_sol[j] = 0
                z2_sol[j] = 0

        print("Objective Value: " + str(MODEL.ObjVal))

        return MODEL.ObjVal, w2_sol, z2_sol

    def fixing_w2(w2, continuous):
        print("\nSolving the model fixing w2")

        MODEL = gp.Model("Fixing w2")

        w1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w1")

        z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")

        MODEL.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in range(p1) for j in range(p1)) <= 1)

        MODEL.addConstr(z1.sum('*') <= k1)

        MODEL.addConstrs(w1[i] <= M * z1[i] for i in range(p1))
        MODEL.addConstrs(w1[i] >= -M * z1[i] for i in range(p1))

        MODEL.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)),
                           GRB.MAXIMIZE)

        MODEL.Params.OutputFlag = 0
        MODEL.Params.MIPGap = 5e-2

        MODEL.optimize()

        w1_sol = np.zeros(p1)
        z1_sol = np.zeros(p1)

        if continuous:
            for i in range(p1):
                w1_sol[i] = w1[i].X
                z1_sol[i] = z1[i].X
        else:
            for i in range(p1):
                if z1[i].X > 0.5:
                    w1_sol[i] = w1[i].X
                    z1_sol[i] = 1
                else:
                    w1_sol[i] = 0
                    z1_sol[i] = 0

        print("Objective Value: " + str(MODEL.ObjVal))

        return MODEL.ObjVal, w1_sol, z1_sol

    w1_indices = kernel_1 + bucket1
    z1_indices = w1_indices

    w2_indices = kernel_2 + bucket2
    z2_indices = w2_indices

    dataset1M = dataset1.iloc[:, w1_indices]
    dataset2M = dataset2.iloc[:, w2_indices]

    M = max([1/np.sqrt(min(LA.eig(dataset1M.T @ dataset1M)[0])), 1/np.sqrt(min(LA.eig(dataset2M.T @ dataset2M)[0]))])

    MODEL = gp.Model("CCA_Continuous")

    w1 = MODEL.addVars(w1_indices, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w1")
    z1 = MODEL.addVars(z1_indices, vtype=GRB.BINARY, name="z1")

    w2 = MODEL.addVars(w2_indices, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w2")
    z2 = MODEL.addVars(z2_indices, vtype=GRB.BINARY, name="z2")

    MODEL.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in w1_indices for j in w1_indices) <= 1.0)
    MODEL.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in w2_indices for j in w2_indices) <= 1.0)

    MODEL.addConstr(z1.sum('*') <= k1)
    MODEL.addConstr(z2.sum('*') <= k2)

    if len(bucket1) > 0:
        MODEL.addConstr(gp.quicksum(z1[i] for i in bucket1) >= 1)

    if len(bucket2) > 0:
        MODEL.addConstr(gp.quicksum(z2[j] for j in bucket2) >= 1)

    MODEL.addConstrs(w1[i] <= M * z1[i] for i in w1_indices)
    MODEL.addConstrs(w1[i] >= -M * z1[i] for i in w1_indices)

    MODEL.addConstrs(w2[j] <= M * z2[j] for j in w2_indices)
    MODEL.addConstrs(w2[j] >= -M * z2[j] for j in w2_indices)

    objective = gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in w1_indices for j in w2_indices)

    MODEL.addConstr(objective >= objval)

    MODEL.setObjective(objective, GRB.MAXIMIZE)

    MODEL.Params.TimeLimit = time_limit
    MODEL.Params.NonConvex = 2
    MODEL.optimize()

    if best_response:
        first_time = time.time()
        final_time = time_limit - 10

        obj_old = 0
        obj_new = MODEL.ObjVal

        objectives = []

        w1 = np.array(w1.X)

        it = 0

        w1s = []
        w2s = []

        z1s = []
        z2s = []

        while abs(obj_old - obj_new) >= 1e-3:

            obj_old = obj_new
            obj_new, w2, z2 = fixing_w1(w1)
            obj_new, w1, z1 = fixing_w2(w2)
            objectives.append(obj_new)
            w1s.append(w1)
            w2s.append(w2)
            z1s.append(z1)
            z2s.append(z2)
            it += 1

        w1 = w1s[np.argmax(objectives)]
        w2 = w2s[np.argmax(objectives)]
        z1 = z1s[np.argmax(objectives)]
        z2 = z2s[np.argmax(objectives)]


    if MODEL.Status == 3 or MODEL.SolCount == 0:
        pass
    else:
        for i in bucket1:
            if z1[i].X > 0.5:
                kernel_1.append(i)

        for j in bucket2:
            if z2[j].X > 0.5:
                kernel_2.append(j)

        objval = MODEL.ObjVal

    return kernel_1, kernel_2, objval



def KS_CCA(dataset1, dataset2, k1, k2, tmax=1200, NB=-1, show=True):

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    w1, w2, objective = CCA_continuous(dataset1, dataset2)

    product12 = np.array(dataset1.T) @ np.array(dataset2)

    values_ws = {}

    for i in range(p1):
        for j in range(p2):
            values_ws[(i, j)] = w1[i]*product12[i, j]*w2[j]

    sorted_keys = sorted(values_ws, key=values_ws.get, reverse=True)

    sorted_w1 = []
    sorted_w2 = []

    for i, j in sorted_keys:
        sorted_w1.append(i)
        sorted_w2.append(j)

    sorted_w1 = list(dict.fromkeys(sorted_w1))
    sorted_w2 = list(dict.fromkeys(sorted_w2))

    if NB < 0:
        k = k1*k2

        splits_1 = np.ceil(np.linspace(0, p1, k + 1))
        splits_2 = np.ceil(np.linspace(0, p2, k + 1))

        kernel_1 = sorted_w1[int(splits_1[0]):int(splits_1[k2])]
        kernel_2 = sorted_w2[int(splits_2[0]):int(splits_2[k1])]

        buckets_1 = []
        buckets_2 = []

        for i in range(k2, k+1-k2, k2):
            buckets_1.append(sorted_w1[int(splits_1[i]):int(splits_1[i + k2])])

        for j in range(k1, k+1-k1, k1):
            buckets_2.append(sorted_w2[int(splits_2[j]):int(splits_2[j + k1])])

    else:

        splits_1 = np.ceil(np.linspace(0, p1, NB + 1))
        splits_2 = np.ceil(np.linspace(0, p2, NB + 1))

        kernel_1 = sorted_w1[int(splits_1[0]):int(splits_1[1])]
        kernel_2 = sorted_w2[int(splits_2[0]):int(splits_2[1])]

        buckets_1 = []
        buckets_2 = []

        for i in range(1, NB):
            buckets_1.append(sorted_w1[int(splits_1[i]):int(splits_1[i + 1])])

        for j in range(1, NB):
            buckets_2.append(sorted_w2[int(splits_2[j]):int(splits_2[j + 1])])

    objectives = []

    t = tmax/(1 + len(buckets_1) + len(buckets_2))

    kernel_1, kernel_2, objective = CCA_restricted(kernel_1, [], kernel_2, [], dataset1, dataset2, k1, k2, 0, t)

    objectives.append(objective)

    for bucket1 in buckets_1:
        print("Kernel 1: " + str(kernel_1))
        print("Bucket 1: " + str(bucket1))

        print("Kernel 2: " + str(kernel_2))

        kernel_1, kernel_2, objective = CCA_restricted(kernel_1, bucket1, kernel_2, [], dataset1, dataset2, k1, k2, objective, t)

        print("Objective Value: " + str(objective))

        objectives.append(objective)

    for bucket2 in buckets_2:
        print("Kernel 1: " + str(kernel_1))

        print("Kernel 2: " + str(kernel_2))
        print("Bucket 2: " + str(bucket2))

        kernel_1, kernel_2, objective = CCA_restricted(kernel_1, [], kernel_2, bucket2, dataset1, dataset2, k1, k2, objective, t)

        print("Objective Value: " + str(objective))

        objectives.append(objective)

    if show:
        plt.plot(objectives)
        plt.show()

    return max(objective)