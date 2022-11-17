
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import time
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.cross_decomposition as sk
from sklearn.model_selection import train_test_split


# Para ver sincronizacion en onedrive.

def continuous_CCA(dataset1, dataset2):

    start_time = time.time()

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product12 = X1.T @ X2
    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    cca = sk.CCA(n_components=1)
    cca.fit(dataset1, dataset2)

    w1 = np.array(cca.x_weights_)
    w2 = np.array(cca.y_weights_)

    numerator = w1.T @ product12 @ w2
    denominator = np.sqrt(w1.T @ product11 @ w1)*np.sqrt(w2.T @ product22 @ w2)

    objective = float(numerator/denominator)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Time Elapsed: " + str(elapsed_time))
    print("Objective Value: " + str(objective))

    return objective, w1, w2, elapsed_time

def greedy_CCA(dataset1, dataset2, k1, k2, time_limit=1000):

    start_time = time.time()

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    I = []
    J = []

    rho_max = 0
    i_max = 0
    j_max = 0

    for i in range(p1):
        for j in range(p2):
            rhoij = product12[i, j]/(np.sqrt(product11[i, i])*np.sqrt(product22[j, j]))

            if rhoij > rho_max:
                rho_max = rhoij
                i_max = i
                j_max = j

    I.append(i_max)
    J.append(j_max)

    k1 -= 1
    k2 -= 1

    while (k1 > 0) or (k2 > 0):

        if k1 > 0:
            rhoi_max = 0

            for i in range(p1):
                if i not in I:
                    # print(I + [i])
                    rhoi, w1, w2, elapsed_time = continuous_CCA(dataset1.iloc[:, I + [i]], dataset2.iloc[:, J])

                    if rhoi > rhoi_max:
                        rhoi_max = rhoi
                        i_max = i
        else:
            rhoi_max = 0

        if k2 > 0:
            rhoj_max = 0

            for j in range(p2):
                if j not in J:
                    rhoj, w1, w2, elapsed_time = continuous_CCA(dataset1.iloc[:, I], dataset2.iloc[:, J + [j]])

                    if rhoj > rhoj_max:
                        rhoj_max = rhoj
                        j_max = j

        else:
            rhoj_max = 0

        if rhoi_max > rhoj_max:
            k1 -= 1
            I = I + [i_max]
        else:
            k2 -= 1
            J = J + [j_max]

        print("I= " + str(I))
        print("J= " + str(J))

    objective, w1, w2, elapsed_time = continuous_CCA(dataset1.iloc[:, I], dataset2.iloc[:, J])

    # print(w1)
    # print(w2)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Time Elapsed: " + str(elapsed_time))
    print("Objective Value: " + str(objective))

    w1_sol = np.zeros(p1)
    for i, n in zip(I, range(len(I))):
        w1_sol[i] = w1[n]

    w2_sol = np.zeros(p2)
    for j, n in zip(J, range(len(J))):
        w2_sol[j] = w2[n]

    return objective, w1_sol, w2_sol, elapsed_time

def fixing_w1(w1, k2, dataset1, dataset2, time_limit = 100, benders = False, bigM_estimation = True):
    print("Solving the model fixing w1\n")

    if bigM_estimation:
        eigs2 = LA.eig(dataset2.T @ dataset2)[0]
        eigs2_real = [eigval for eigval in eigs2 if abs(eigval) > 0.1 and not(np.iscomplex(eigval))]

        M = float(1/np.sqrt(min(eigs2_real)))
        
        # M = 1/np.sqrt(min(np.abs(LA.eig(dataset2.T @ dataset2)[0])))
    else:
        M = 1

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    MODEL = gp.Model("Fixing w1")

    w2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w2")
    z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")

    if benders:
        alpha1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha1")
        alpha2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha2")

        beta1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="beta1")
        beta2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="beta2")

        gamma1 = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma1")
        gamma2 = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma2")

        MODEL.addConstrs(beta1[i] == alpha1[i] + gp.quicksum(product12[i, j]*w2[j] for j in range(p2)) - 2*gp.quicksum(product11[i, j]*gamma1*w1[i] for j in range(p1)) for i in range(p1))
        MODEL.addConstrs(beta2[j] == alpha2[j] + gp.quicksum(product12[i, j]*w1[i] for i in range(p1)) - 2*gp.quicksum(product22[i, j]*gamma2*w2[j] for i in range(p2)) for j in range(p2))

    MODEL.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in range(p2) for j in range(p2)) <= 1)

    MODEL.addConstr(z2.sum('*') <= k2)

    MODEL.addConstrs(w2[j] <= M * z2[j] for j in range(p2))
    MODEL.addConstrs(w2[j] >= -M * z2[j] for j in range(p2))

    MODEL.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)), GRB.MAXIMIZE)

    MODEL.Params.TimeLimit = time_limit
    MODEL.Params.OutputFlag = 0
    MODEL.Params.MIPGap = 5e-2
    MODEL.Params.Threads = 12

    if benders:
        MODEL.Params.NonConvex = 2

    MODEL.optimize()

    if MODEL.Status == 3 or MODEL.SolCount == 0:

        if benders:
            return -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return -1, np.nan, np.nan

    print("Objective Value: " + str(MODEL.ObjVal))

    objective = MODEL.ObjVal

    w2_sol = np.array([w2[j].X for j in range(p2)])
    z2_sol = np.array([z2[j].X for j in range(p2)])

    if benders:
        w1_sol = w1
        z1_sol = w1 != 0
        alpha1_sol = np.zeros(p1)
        alpha2_sol = np.zeros(p2)
        beta1_sol = np.zeros(p1)
        beta2_sol = np.zeros(p2)

        for i in range(p1):
            if w1_sol[i] != 0:
                alpha1_sol[i] = 0
                beta1_sol[i] = 0
            else:
                alpha1_sol[i] = alpha1[i].X
                beta1_sol[i] = beta1[i].X

        for j in range(p2):
            if w2_sol[j] != 0:
                alpha2_sol[j] = 0
                beta2_sol[j] = 0
            else:
                alpha2_sol[j] = alpha2[j].X
                beta2_sol[j] = beta2[j].X

        return objective, w2_sol, z2_sol, w1_sol, z1_sol, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol

    return objective, w2_sol, z2_sol

def fixing_w2(w2, k1, dataset1, dataset2, time_limit = 100, benders = False, bigM_estimation = True):
    print("Solving the model fixing w2\n")

    if bigM_estimation:
        eigs2 = LA.eig(dataset1.T @ dataset1)[0]
        eigs2_real = [eigval for eigval in eigs2 if abs(eigval) > 0.1 and not(np.iscomplex(eigval))]

        M = float(1/np.sqrt(min(eigs2_real)))
        
        # M = 1/np.sqrt(min(np.abs(LA.eig(dataset1.T @ dataset1)[0])))
    else:
        M = 1



    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    MODEL = gp.Model("Fixing w2")

    w1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w1")
    z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")

    if benders:
        alpha1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha1")
        alpha2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha2")

        beta1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="beta1")
        beta2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="beta2")

        gamma1 = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma1")
        gamma2 = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma2")

        MODEL.addConstrs(beta1[i] == alpha1[i] + gp.quicksum(product12[i, j]*w2[j] for j in range(p2)) - 2*gp.quicksum(product11[i, j]*gamma1*w1[i] for j in range(p1)) for i in range(p1))
        MODEL.addConstrs(beta2[j] == alpha2[j] + gp.quicksum(product12[i, j]*w1[i] for i in range(p1)) - 2*gp.quicksum(product22[i, j]*gamma2*w2[j] for i in range(p2)) for j in range(p2))


    MODEL.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in range(p1) for j in range(p1)) <= 1)

    MODEL.addConstr(z1.sum('*') <= k1)

    MODEL.addConstrs(w1[i] <= M * z1[i] for i in range(p1))
    MODEL.addConstrs(w1[i] >= -M * z1[i] for i in range(p1))

    MODEL.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)), GRB.MAXIMIZE)

    MODEL.Params.OutputFlag = 0
    MODEL.Params.MIPGap = 5e-2
    MODEL.Params.TimeLimit = time_limit
    MODEL.Params.Threads = 12

    if benders:
        MODEL.Params.NonConvex = 2

    MODEL.optimize()

    if MODEL.Status == 3 or MODEL.SolCount == 0:

        if benders:
            return -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return -1, np.nan, np.nan

    print("Objective Value: " + str(MODEL.ObjVal))

    objective = MODEL.ObjVal

    w1_sol = np.array([w1[i].X for i in range(p1)])
    z1_sol = np.array([z1[i].X for i in range(p1)])

    if benders:
        w2_sol = w2
        z2_sol = w2 != 0
        alpha1_sol = np.zeros(p1)
        alpha2_sol = np.zeros(p2)
        beta1_sol = np.zeros(p1)
        beta2_sol = np.zeros(p2)

        for i in range(p1):
            if w1_sol[i] != 0:
                alpha1_sol[i] = 0
                beta1_sol[i] = 0
            else:
                alpha1_sol[i] = alpha1[i].X
                beta1_sol[i] = beta1[i].X

        for j in range(p2):
            if w2_sol[j] != 0:
                alpha2_sol[j] = 0
                beta2_sol[j] = 0
            else:
                alpha2_sol[j] = alpha2[j].X
                beta2_sol[j] = beta2[j].X

        return objective, w1_sol, z1_sol, w2_sol, z2_sol, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol

    return objective, w1_sol, z1_sol

def CCA(dataset1, dataset2, k1, k2, time_limit = 1000, init = -1, best_response=False, bigM_estimation = True):

    print("*******")
    print("Computing the CCA. k1 = " + str(k1) + "; k2 = " + str(k2))
    print("*******")

    if bigM_estimation:
        # print(LA.eig(dataset1.T @ dataset1)[0])
        eigs1 = LA.eig(dataset1.T @ dataset1)[0]
        eigs1_real = [eigval for eigval in eigs1 if abs(eigval) > 0.1 and not(np.iscomplex(eigval))]
        eigs2 = LA.eig(dataset2.T @ dataset2)[0]
        eigs2_real = [eigval for eigval in eigs2 if abs(eigval) > 0.1 and not(np.iscomplex(eigval))]

        M = float(max([1/np.sqrt(min(eigs1_real)), 1/np.sqrt(min(eigs2_real))]))

        if M > 1.0e+06:
            M = 1.0e+04
        print("Valor de M:"+str(M))
    else:
        M = 1

    start_time = time.time()

    MODEL = gp.Model("CCA")

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    w1 = MODEL.addVars(p1, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w1")
    z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")

    w2 = MODEL.addVars(p2, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w2")
    z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")

    MODEL.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in range(p1) for j in range(p1)) <= 1.0)
    MODEL.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in range(p2) for j in range(p2)) <= 1.0)

    MODEL.addConstr(z1.sum('*') <= k1)
    MODEL.addConstr(z2.sum('*') <= k2)

    MODEL.addConstrs(w1[i] <= M*z1[i] for i in range(p1))
    MODEL.addConstrs(w1[i] >= -M*z1[i] for i in range(p1))

    MODEL.addConstrs(w2[j] <= M*z2[j] for j in range(p2))
    MODEL.addConstrs(w2[j] >= -M*z2[j] for j in range(p2))

    if init > -1:
        np.random.seed(init)

        arr1 = list([1] * k1 + [0] * (p1 - k1))
        random.shuffle(arr1)

        for i in range(p1):
            MODEL.addConstr(z1[i] <= arr1.copy()[i])

        arr2 = np.array([1] * k2 + [0] * (p2 - k2))
        random.shuffle(arr2)

        for i in range(p2):
            MODEL.addConstr(z2[i] <= arr2.copy()[i])

    MODEL.addConstr(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)) <= 1)
    MODEL.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)), GRB.MAXIMIZE)

    MODEL.Params.NonConvex = 2
    MODEL.Params.Threads = 12

    if best_response:
        MODEL.Params.TimeLimit = time_limit/10
    else:
        MODEL.Params.TimeLimit = time_limit

    MODEL.optimize()

    if MODEL.SolCount == 0 or MODEL.ObjVal < 0.001:

        if best_response:

            if init < 0:
                init = 2000

            np.random.seed(init)

            arr1 = list([1] * k1 + [0] * (p1 - k1))
            random.shuffle(arr1)

            for i in range(p1):
                MODEL.addConstr(z1[i] <= arr1.copy()[i])

            arr2 = np.array([1] * k2 + [0] * (p2 - k2))
            random.shuffle(arr2)

            for i in range(p2):
                MODEL.addConstr(z2[i] <= arr2.copy()[i])
                
            MODEL.optimize()

            w1 = np.array([w1[i].X for i in range(p1)])
            w2 = np.array([w2[j].X for j in range(p2)])

        else:
            
            end_time = time.time()
            
            time_elapsed = end_time - start_time
            
            return -1, np.zeros(p1), np.zeros(p2), time_elapsed, 0

    else:
        w1 = np.array([w1[i].X for i in range(p1)])
        w2 = np.array([w2[j].X for j in range(p2)])

    obj_old = 0
    obj_new = MODEL.ObjVal

    objectives = []

    w1s = [w1]
    w2s = [w2]

    objectives = [MODEL.ObjVal]

    time_initialization = MODEL.Runtime

    n_iter = 0

    if best_response:

        time_remaining = time_limit - time_initialization
        end_time = time.time()

        while abs(obj_old - obj_new) >= 1e-3 and (end_time-start_time) <= time_remaining:

            obj_old = obj_new
            obj_new, w2, z2 = fixing_w1(w1, k2, dataset1, dataset2, time_limit = time_remaining/10, benders = False, bigM_estimation=bigM_estimation)

            objectives.append(obj_new)
            w1s.append(w1)
            w2s.append(w2)

            n_iter += 1
            end_time = time.time()

            if abs(obj_old - obj_new) <= 1e-3 or (end_time - start_time) >= time_remaining:
                break

            obj_new, w1, z1 = fixing_w2(w2, k1, dataset1, dataset2, time_limit = time_remaining/10, benders = False, bigM_estimation=bigM_estimation)

            objectives.append(obj_new)
            w1s.append(w1)
            w2s.append(w2)

            n_iter += 1
            end_time = time.time()

            if abs(obj_old - obj_new) <= 1e-3 or (end_time - start_time) >= time_remaining:
                break

    w1 = w1s[np.argmax(objectives)]
    w2 = w2s[np.argmax(objectives)]

    objective = max(objectives)

    end_time = time.time()

    time_elapsed = end_time - start_time

    print("Time Elapsed: " + str(time_elapsed))
    print("Objective Value: " + str(objective))

    if best_response:
        print("Number of iterations: " + str(n_iter))

    return objective, w1, w2, time_elapsed, n_iter

def multistart_CCA(dataset1, dataset2, k1, k2, n_iter=10, time_limit = 1000, best_response=False, bigM_estimation=True):

    start_time = time.time()

    objectives = []

    w1s = []
    w2s = []

    for i in range(n_iter):
        objective, w1, w2, time_elapsed, iters = CCA(dataset1, dataset2, k1, k2, time_limit = time_limit/n_iter, init = i, best_response=best_response, bigM_estimation=bigM_estimation)

        w1s.append(w1)
        w2s.append(w2)

        objectives.append(objective)

    w1_sol = w1s[np.argmax(objectives)]
    w2_sol = w2s[np.argmax(objectives)]

    objective = max(objectives)

    end_time = time.time()

    time_elapsed = end_time - start_time

    print("Time Elapsed: " + str(time_elapsed))
    print("Objective Value: " + str(objective))

    return objective, w1_sol, w2_sol, time_elapsed


def subproblem(z1, z2, dataset1, dataset2, time_limit = 100, bigM_estimation=True):

    print("Solving subproblem by fixing z1 and z2\n")

    start_time = time.time()

    MODEL2 = gp.Model("Subproblem")

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    z1_indices = [i for i in range(p1) if z1[i] > 0.5]
    z2_indices = [j for j in range(p2) if z2[j] > 0.5]

    if bigM_estimation:
        dataset1M = dataset1.iloc[:, z1_indices]
        dataset2M = dataset2.iloc[:, z2_indices]

        n, u1 = dataset1M.shape
        n, u2 = dataset2M.shape

        # print(dataset1M)
        # print(dataset2M)

        if u1 == 0:
            M = 1 / np.sqrt(min(np.abs(LA.eig(dataset2M.T @ dataset2M)[0])))
        elif u2 == 0:
            M = 1 / np.sqrt(min(np.abs(LA.eig(dataset1M.T @ dataset1M)[0])))
        else:
            M = max([1 / np.sqrt(min(np.abs(LA.eig(dataset1M.T @ dataset1M)[0]))),
                     1 / np.sqrt(min(np.abs(LA.eig(dataset2M.T @ dataset2M)[0])))])
    else:
        M = 1

    w1 = MODEL2.addVars(p1, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w1")
    w2 = MODEL2.addVars(p2, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w2")

    alpha1 = MODEL2.addVars(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha1")
    alpha2 = MODEL2.addVars(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha2")

    beta1 = MODEL2.addVars(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="beta1")
    beta2 = MODEL2.addVars(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="beta2")

    gamma1 = MODEL2.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma1")
    gamma2 = MODEL2.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma2")

    for i in range(p1):
        if z1[i] < 0.5:
            w1[i] = 0
        # else:
        #     alpha1[i] = 0
        #     beta1[i] = 0

    for j in range(p2):
        if z2[j] < 0.5:
            w2[j] = 0
        # else:
        #     alpha2[j] = 0
        #     beta2[j] = 0

    MODEL2.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in range(p1) for j in range(p1)) <= 1)
    MODEL2.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in range(p2) for j in range(p2)) <= 1)

    MODEL2.addConstrs(beta1[i] == alpha1[i] + gp.quicksum(product12[i, j] * w2[j] for j in range(p2)) - 2 * gp.quicksum(
        product11[i, j] * gamma1 * w1[i] for j in range(p1)) for i in range(p1))
    MODEL2.addConstrs(beta2[j] == alpha2[j] + gp.quicksum(product12[i, j] * w1[i] for i in range(p1)) - 2 * gp.quicksum(
        product22[i, j] * gamma2 * w2[j] for i in range(p2)) for j in range(p2))

    MODEL2.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)),
                        GRB.MAXIMIZE)

    MODEL2.Params.NonConvex = 2

    MODEL2.Params.MIPGap = 1e-2
    MODEL2.Params.TimeLimit = time_limit

    MODEL2.optimize()

    objective = MODEL2.ObjVal

    w1_sol = np.zeros(p1)
    z1_sol = z1
    w2_sol = np.zeros(p2)
    z2_sol = z2

    alpha1_sol = np.zeros(p1)
    alpha2_sol = np.zeros(p2)
    beta1_sol = np.zeros(p1)
    beta2_sol = np.zeros(p2)

    for i in range(p1):
        if z1[i] > 0.5:
            w1_sol[i] = w1[i].X
            alpha1_sol[i] = 0
            beta1_sol[i] = 0
        else:
            w1_sol[i] = 0
            alpha1_sol[i] = alpha1[i].X
            beta1_sol[i] = beta1[i].X

    for j in range(p2):
        if z2[j] > 0.5:
            w2_sol[j] = w2[j].X
            alpha2_sol[j] = 0
            beta2_sol[j] = 0
        else:
            w2_sol[j] = 0
            alpha2_sol[j] = alpha2[j].X
            beta2_sol[j] = beta2[j].X

    if MODEL2.Status == 3:
        MODEL2.computeIIS()
        MODEL2.write("infeasible_subproblem.ilp")

    end_time = time.time()

    time_elapsed = end_time - start_time

    print("Time Elapsed Subproblem: " + str(time_elapsed))
    print("Objective Value Subproblem: " + str(objective))

    return objective, w1_sol, z1_sol, w2_sol, z2_sol, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol

def benders_CCA(dataset1, dataset2, k1, k2, init = True, time_limit = 1000, bigM_estimation=True):

    print("*******")
    print("Computing the benders_CCA. k1 = " + str(k1) + "; k2 = " + str(k2))
    if init:
        print("Initializing with best_response")
    print("*******")

    if bigM_estimation:
         # print(LA.eig(dataset1.T @ dataset1)[0])
         eigs1 = LA.eig(dataset1.T @ dataset1)[0]
         eigs1_real = [eigval for eigval in eigs1 if abs(eigval) > 0.1 and not(np.iscomplex(eigval))]
         eigs2 = LA.eig(dataset2.T @ dataset2)[0]
         eigs2_real = [eigval for eigval in eigs2 if abs(eigval) > 0.1 and not(np.iscomplex(eigval))]

         M = float(max([1/np.sqrt(min(eigs1_real)), 1/np.sqrt(min(eigs2_real))]))
    else:
        M = 1



    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    start_time = time.time()

    MODEL = gp.Model("Initializing Model")

    z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")
    z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")
    ub = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="ub")
    lb = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="lb")

    MODEL.addConstr(z1.sum('*') <= k1)
    MODEL.addConstr(z2.sum('*') <= k2)

    MODEL.addConstr(lb <= ub)

    np.random.seed(1)

    MODEL._z1 = z1
    MODEL._z2 = z2
    MODEL._ub = ub
    MODEL._lb = lb

    w1s = []
    w2s = []
    objectives = []

    if init:
        initialization_time = time.time()

        objective, w1_sol, w2_sol, time_initialization, n_iter = CCA(dataset1, dataset2, k1, k2, time_limit=time_limit/10, best_response=False, bigM_estimation=bigM_estimation)

        if objective < 0:
            objective, w1_sol, w2_sol, time_initialization, n_iter = CCA(dataset1, dataset2, k1, k2, time_limit=time_limit/10, init = 0, best_response=False, bigM_estimation=bigM_estimation)

        obj_old = 0
        obj_new = objective

        w1s.append(w1_sol)
        w2s.append(w2_sol)
        objectives.append(objective)

        # time_remaining = (time_limit - time_initialization)/5
        end_time = time.time()

        while 1:
            # while it <= 10:

            obj_old = obj_new

            obj_new, w2_sol, z2_sol, w1_sol, z1_sol, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol = fixing_w1(w1_sol, k2, dataset1, dataset2, benders = True, bigM_estimation=bigM_estimation)

            if obj_new >= 0:
                MODEL.addConstr(MODEL._ub <= obj_new
                                + gp.quicksum(alpha1_sol[i] * M * MODEL._z1[i] for i in range(p1) if z1_sol[i] < 0.5)
                                + gp.quicksum(alpha2_sol[j] * M * MODEL._z2[j] for j in range(p2) if z2_sol[j] < 0.5)
                                + gp.quicksum(beta1_sol[i] * M * MODEL._z1[i] for i in range(p1) if z1_sol[i] < 0.5)
                                + gp.quicksum(beta2_sol[j] * M * MODEL._z2[j] for j in range(p2) if z2_sol[j] < 0.5))

                MODEL.addConstr(lb >= obj_new)

            end_time = time.time()

            if obj_new < 0:
                w1_sol = w1s[-1]
                w2_sol = w2s[-1]

            if abs(obj_old - obj_new) <= 1e-3 or end_time - start_time >= time_limit/2:
                break

            w1s.append(w1_sol)
            w2s.append(w2_sol)
            objectives.append(obj_new)

            obj_new, w1_sol, z1_sol, w2_sol, z2_sol, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol = fixing_w2(w2_sol, k1, dataset1, dataset2, benders = True, bigM_estimation=bigM_estimation)

            if obj_new >= 0:
                MODEL.addConstr(MODEL._ub <= obj_new
                                + gp.quicksum(alpha1_sol[i] * M * MODEL._z1[i] for i in range(p1) if z1_sol[i] < 0.5)
                                + gp.quicksum(alpha2_sol[j] * M * MODEL._z2[j] for j in range(p2) if z2_sol[j] < 0.5)
                                + gp.quicksum(beta1_sol[i] * M * MODEL._z1[i] for i in range(p1) if z1_sol[i] < 0.5)
                                + gp.quicksum(beta2_sol[j] * M * MODEL._z2[j] for j in range(p2) if z2_sol[j] < 0.5))

                MODEL.addConstr(lb >= obj_new)

            end_time = time.time()

            if obj_new < 0:
                w1_sol = w1s[-1]
                w2_sol = w2s[-1]

            if abs(obj_old - obj_new) <= 1e-3  or end_time - start_time >= time_limit/2:
                break

            w1s.append(w1_sol)
            w2s.append(w2_sol)
            objectives.append(obj_new)

        initialization_elapsed = end_time - initialization_time
        # time_remaining *= 10
        # time_remaining -= initialization_elapsed

        # print(time_remaining)

        z1_sol = w1_sol != 0
        z2_sol = w2_sol != 0

    else:
        z1_sol = np.zeros(p1)
        arr1 = np.array([1] * k1 + [0] * (p1 - k1))
        np.random.shuffle(arr1)

        for i in range(p1):
            z1_sol[i] = arr1[i]

        z2_sol = np.zeros(p2)
        arr2 = np.array([1] * k2 + [0] * (p2 - k2))
        np.random.shuffle(arr2)

        for i in range(p2):
            z2_sol[i] = arr2[i]

    MODEL.setObjective(ub, GRB.MAXIMIZE)
    MODEL.write('model.lp')

    eps = 0.001
    LB = 0
    UB = continuous_CCA(dataset1, dataset2)[0]

    lbs = [LB]
    ubs = [UB]
    n_cuts = 0

    end_time = time.time()

    while abs(UB - LB) > eps and (end_time-start_time) <= time_limit:
        objective, w1_sol, z1_sol, w2_sol, z2_sol, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol = subproblem(z1_sol, z2_sol, dataset1, dataset2, bigM_estimation=bigM_estimation)

        w1s.append(w1_sol)
        w2s.append(w2_sol)
        objectives.append(objective)

        MODEL.addConstr(MODEL._lb >= LB)
        MODEL.addConstr(MODEL._ub <= objective
                        + gp.quicksum(alpha1_sol[i] * M * MODEL._z1[i] for i in range(p1) if z1_sol[i] < 0.5)
                        + gp.quicksum(alpha2_sol[j] * M * MODEL._z2[j] for j in range(p2) if z2_sol[j] < 0.5)
                        + gp.quicksum(beta1_sol[i] * M * MODEL._z1[i] for i in range(p1) if z1_sol[i] < 0.5)
                        + gp.quicksum(beta2_sol[j] * M * MODEL._z2[j] for j in range(p2) if z2_sol[j] < 0.5))

        n_cuts += 1
        MODEL.optimize()


        z1_sol = [z1[i].X for i in range(p1)]
        z2_sol = [z2[j].X for j in range(p2)]

        LB = max(LB, objective)
        UB = min(UB, MODEL.ObjVal)

        print('LB = ' + str(LB))
        print('UB = ' + str(UB))

        end_time = time.time()
        print(end_time-start_time)

    objective, w1_sol, z1_sol, w2_sol, z2_sol, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol = subproblem(z1_sol, z2_sol, dataset1, dataset2, bigM_estimation=bigM_estimation)

    w1s.append(w1_sol)
    w2s.append(w2_sol)
    objectives.append(objective)

    w1_sol = w1s[np.argmax(objectives)]
    w2_sol = w2s[np.argmax(objectives)]
    objective = max(objectives)

    # objective = max(LB, objective)

    end_time = time.time()

    time_elapsed = end_time - start_time

    print("\nTime Elapsed: " + str(time_elapsed))
    print("Objective Value: " + str(objective))
    print("Number of cuts: " + str(n_cuts))

    return objective, w1_sol, w2_sol, time_elapsed, n_cuts

def restricted_fixing_w1(kernel_1, bucket1, kernel_2, bucket2, objval, w1_sol, k2, dataset1, dataset2, combined=False, bigM_estimation=True):

    print("\nSolving the model fixing w1")

    print("Kernel 2: " + str(kernel_2))
    print("Bucket 2: " + str(bucket2))

    M = 1/np.sqrt(min(np.abs(LA.eig(dataset2.T @ dataset2)[0])))

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    w1_indices = kernel_1
    z1_indices = w1_indices

    w2_indices = kernel_2 + bucket2
    z2_indices = w2_indices

    if bigM_estimation:
        dataset2M = dataset2.iloc[:, w2_indices]

        M = 1/np.sqrt(min(np.abs(LA.eig(dataset2M.T @ dataset2M)[0])))
    else:
        M = 1

    MODEL = gp.Model("Fixing w1")

    w2 = MODEL.addVars(w2_indices, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w2")
    z2 = MODEL.addVars(z2_indices, vtype=GRB.BINARY, name="z2")

    MODEL.update()

    MODEL.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in w2_indices for j in w2_indices) <= 1)

    MODEL.addConstr(z2.sum('*') <= k2)

    if len(bucket2) > 0:
        MODEL.addConstr(gp.quicksum(z2[j] for j in bucket2) >= 1)

    MODEL.addConstrs(w2[j] <= M * z2[j] for j in z2_indices)
    MODEL.addConstrs(w2[j] >= -M * z2[j] for j in z2_indices)

    objective = gp.quicksum(w1_sol[i] * product12[i, j] * w2[j] for i in w1_indices for j in w2_indices)

    MODEL.addConstr(objective >= objval)

    MODEL.setObjective(objective, GRB.MAXIMIZE)

    # MODEL.Params.OutputFlag = 0
    # MODEL.Params.MIPGap = 5e-2
    MODEL.Params.TimeLimit = 100
    MODEL.Params.Threads = 12

    MODEL.optimize()

    if MODEL.SolCount == 0 or MODEL.ObjVal < 0.01:

        w2_sol = np.zeros(1)
        z2_sol = np.zeros(1)

        if combined:
            return kernel_1, kernel_2, -1, w2_sol, z2_sol
        else:
            return objval, w2_sol, z2_sol

    print("Objective Value: " + str(MODEL.ObjVal))

    if combined:
        for j in bucket2:
            if w2[j].X != 0:
                kernel_2.append(j)
        w2_sol = {}

        for j in w2_indices:
            w2_sol[j] = w2[j].X

        return kernel_1, kernel_2, MODEL.ObjVal, w2_sol, z2

    else:
        return MODEL.ObjVal, w2, z2

def restricted_fixing_w2(kernel_1, bucket1, kernel_2, bucket2, objval, w2_sol, k1, dataset1, dataset2, combined = False, bigM_estimation=True):

    print("\nSolving the model fixing w2")

    print("Kernel 1: " + str(kernel_1))
    print("Bucket 1: " + str(bucket1))

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    w1_indices = kernel_1 + bucket1
    z1_indices = w1_indices

    w2_indices = kernel_2
    z2_indices = w2_indices

    if bigM_estimation:
        dataset1M = dataset1.iloc[:, w1_indices]

        M = 1/np.sqrt(min(np.abs(LA.eig(dataset1M.T @ dataset1M)[0])))
    else:
        M = 1

    MODEL = gp.Model("Fixing w2")

    w1 = MODEL.addVars(w1_indices, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w1")
    z1 = MODEL.addVars(z1_indices, vtype=GRB.BINARY, name="z1")

    MODEL.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in w1_indices for j in w1_indices) <= 1)

    MODEL.addConstr(z1.sum('*') <= k1)

    if len(bucket1) > 0:
        MODEL.addConstr(gp.quicksum(z1[i] for i in bucket1) >= 1)

    MODEL.addConstrs(w1[i] <= M * z1[i] for i in z1_indices)
    MODEL.addConstrs(w1[i] >= -M * z1[i] for i in z1_indices)

    objective = gp.quicksum(w1[i] * product12[i, j] * w2_sol[j] for i in w1_indices for j in w2_indices)

    MODEL.addConstr(objective >= objval)

    MODEL.setObjective(objective, GRB.MAXIMIZE)

    # MODEL.Params.OutputFlag = 0
    # MODEL.Params.MIPGap = 5e-2
    MODEL.Params.TimeLimit = 100
    MODEL.Params.Threads = 12

    MODEL.optimize()

    if MODEL.Status == 3 or MODEL.SolCount == 0:

        w1_sol = np.zeros(1)
        z1_sol = np.zeros(1)

        if combined:
            return kernel_1, kernel_2, -1, w1_sol, z1_sol

        else:
            return objval, w1_sol, z1_sol

    print("Objective Value: " + str(MODEL.ObjVal))

    if combined:
        for i in bucket1:
            if w1[i].X != 0:
                kernel_1.append(i)

        w1_sol = {}
        for i in w1.keys():
            w1_sol[i] = w1[i].X

        return kernel_1, kernel_2, MODEL.ObjVal, w1_sol, w2_sol

    else:
        return MODEL.ObjVal, w1, z1

def restricted_CCA(kernel_1, bucket1, kernel_2, bucket2, objval, dataset1, dataset2, k1, k2, time_limit, best_response=False, bigM_estimation=True):

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1 + 1e-7*np.identity(p1)


    product22 = X2.T @ X2 + 1e-7*np.identity(p2)



    w1_indices = kernel_1 + bucket1
    z1_indices = w1_indices

    w2_indices = kernel_2 + bucket2
    z2_indices = w2_indices

    dataset1M = dataset1.iloc[:, w1_indices]
    dataset2M = dataset2.iloc[:, w2_indices]

    if bigM_estimation:
        M = max([1/np.sqrt(min(np.abs(LA.eig(dataset1M.T @ dataset1M)[0]))), 1/np.sqrt(min(np.abs(LA.eig(dataset2M.T @ dataset2M)[0])))])
    else:
        M = 1

    start_time = time.time()

    MODEL = gp.Model("Restricted_CCA")

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

    if best_response:
        MODEL.Params.TimeLimit = time_limit/10
    else:
        MODEL.Params.TimeLimit = time_limit

    # MODEL.Params.MIPGap = 0.05
    MODEL.Params.NonConvex = 2
    MODEL.Params.Threads = 12
    MODEL.optimize()

    if MODEL.SolCount == 0 or MODEL.ObjVal <= 0.01:

        w1_sol = np.zeros(1)
        w2_sol = np.zeros(1)

        return kernel_1, kernel_2, -1, w1_sol, w2_sol

    else:
        obj_old = 0
        obj_new = MODEL.ObjVal

        objectives = []
        w1_sol = {}
        for i in w1.keys():
            w1_sol[i] = w1[i].X

        w2_sol = {}
        for j in w2.keys():
            w2_sol[j] = w2[j].X

        # w1 = np.array([w1[i].X for i in w1_indices])
        # w2 = np.array([w2[j].X for j in w2_indices])

        w1s = [w1_sol]
        w2s = [w2_sol]

        objectives = [MODEL.ObjVal]

        time_initialization = MODEL.Runtime

        if best_response:

            it = 0

            time_remaining = time_limit - time_initialization
            end_time = time.time()

            while 1:
                # while it <= 10:

                obj_old = obj_new

                obj_new, w2, z2 = restricted_fixing_w1(kernel_1, bucket1, kernel_2, bucket2, obj_old, w1_sol, k2, dataset1, dataset2, bigM_estimation=bigM_estimation)

                objectives.append(obj_new)
                w1s.append(w1)
                w2s.append(w2)

                it += 1
                end_time = time.time()

                if abs(obj_old - obj_new) <= 1e-3 or (end_time-start_time) >= time_remaining:
                    break

                obj_new, w1, z1 = restricted_fixing_w2(kernel_1, bucket1, kernel_2, bucket2, obj_old, w2, k1, dataset1, dataset2, bigM_estimation=bigM_estimation)

                objectives.append(obj_new)
                w1s.append(w1)
                w2s.append(w2)

                it += 1
                end_time = time.time()

                if abs(obj_old - obj_new) <= 1e-3 or (end_time-start_time) <= time_remaining:
                    break

        w1_sol = w1s[np.argmax(objectives)]
        w2_sol = w2s[np.argmax(objectives)]

        objective = max(objectives)

        end_time = time.time()

        time_elapsed = end_time - start_time

        for i in bucket1:
            if w1_sol[i] != 0:
                kernel_1.append(i)

        for j in bucket2:
            if w2_sol[j] != 0:
                kernel_2.append(j)

        w1 = {}
        for i in w1_sol.keys():
            w1[i] = w1_sol[i]

        w2 = {}
        for j in w2_sol.keys():
            w2[j] = w2_sol[j]

        return kernel_1, kernel_2, objective, w1, w2

def kernelsearch_CCA(dataset1, dataset2, k1, k2, NB, time_limit=1000, best_response = False, bigM_estimation=True):

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    start_time = time.time()

    objective, w1, w2, elapsed_time = continuous_CCA(dataset1, dataset2)

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

    # if NB < 0:
    #     k = k1*k2
    #
    #     splits_1 = np.ceil(np.linspace(0, p1, k + 1))
    #     splits_2 = np.ceil(np.linspace(0, p2, k + 1))
    #
    #     kernel_1 = sorted_w1[int(splits_1[0]):int(splits_1[k2])]
    #     kernel_2 = sorted_w2[int(splits_2[0]):int(splits_2[k1])]
    #
    #     buckets_1 = []
    #     buckets_2 = []
    #
    #     for i in range(k2, k+1-k2, k2):
    #         buckets_1.append(sorted_w1[int(splits_1[i]):int(splits_1[i + k2])])
    #
    #     for j in range(k1, k+1-k1, k1):
    #         buckets_2.append(sorted_w2[int(splits_2[j]):int(splits_2[j + k1])])
    #
    # else:
    if NB >= min([p1, p2]):
        print("Changing the NB to: " + str(min([p1, p2])))
        NB = min([p1, p2])

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

    print("Initial Kernel_1: " + str(kernel_1))
    print("Buckets_1: " + str(buckets_1))

    print("Initial Kernel_2: " + str(kernel_1))
    print("Buckets_2: " + str(buckets_2))

    # t = time_limit/(1 + len(buckets_1) + len(buckets_2))

    t = time_limit/(1 + len(buckets_1))

    kernel_1, kernel_2, objective, w1, w2 = restricted_CCA(kernel_1, [], kernel_2, [], 0, dataset1, dataset2, k1, k2, t, best_response=False, bigM_estimation=bigM_estimation)

    objectives = [objective]
    w1s = [w1]
    w2s = [w2]

    for bucket1, bucket2 in zip(buckets_1, buckets_2):
        print("Kernel 1: " + str(kernel_1))
        print("Kernel 2: " + str(kernel_2))

        print("\nBucket 1: " + str(bucket1))
        print("Bucket 2: " + str(bucket2))

        kernel_1, kernel_2, objective, w1, w2 = restricted_CCA(kernel_1, bucket1, kernel_2, bucket2, max(objectives), dataset1, dataset2, k1, k2, t, bigM_estimation=bigM_estimation)

        objectives.append(objective)
        w1s.append(w1)
        w2s.append(w2)

    w1 = w1s[np.argmax(objectives)]
    w2 = w2s[np.argmax(objectives)]

    objective = max(objectives)

    w1_sol = np.zeros(p1)
    w2_sol = np.zeros(p2)

    for i in range(p1):
        if i in w1.keys():
            w1_sol[i] = w1[i]

    for j in range(p2):
        if j in w2.keys():
            w2_sol[j] = w2[j]

    end_time = time.time()

    time_elapsed = end_time - start_time

    end_time = time.time()

    time_elapsed = end_time - start_time
    # for bucket2 in buckets_2:
    #     print("Kernel 1: " + str(kernel_1))
    #
    #     print("Kernel 2: " + str(kernel_2))
    #     print("Bucket 2: " + str(bucket2))
    #
    #     kernel_1, kernel_2, objective = CCA_restricted(kernel_1, [], kernel_2, bucket2, dataset1, dataset2, k1, k2, objective, t)
    #
    #     print("Objective Value: " + str(objective))
    #
    #     objectives.append(objective)

    # if show:
    #     plt.plot(objectives)
    #     plt.show()

    print("Time Elapsed: " + str(time_elapsed))
    print("Objective Value: " + str(objective))

    return objective, w1_sol, w2_sol, time_elapsed

def combined_CCA(dataset1, dataset2, k1, k2, NB, time_limit=1000, bigM_estimation=True):

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    start_time = time.time()

    objective, w1, w2, elapsed_time = continuous_CCA(dataset1, dataset2)

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

    # print(sorted_w1)
    # print(sorted_w2)
    # if NB < 0:
    #     k = k1*k2
    #
    #     splits_1 = np.ceil(np.linspace(0, p1, k + 1))
    #     splits_2 = np.ceil(np.linspace(0, p2, k + 1))
    #
    #     kernel_1 = sorted_w1[int(splits_1[0]):int(splits_1[k2])]
    #     kernel_2 = sorted_w2[int(splits_2[0]):int(splits_2[k1])]
    #
    #     buckets_1 = []
    #     buckets_2 = []
    #
    #     for i in range(k2, k+1-k2, k2):
    #         buckets_1.append(sorted_w1[int(splits_1[i]):int(splits_1[i + k2])])
    #
    #     for j in range(k1, k+1-k1, k1):
    #         buckets_2.append(sorted_w2[int(splits_2[j]):int(splits_2[j + k1])])
    #
    # else:
    if NB >= min([p1, p2]):
        print("Changing the NB to: " + str(min([p1, p2])))
        NB = min([p1, p2])

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

    print("Initial Kernel_1: " + str(kernel_1))
    print("Buckets_1: " + str(buckets_1))

    print("Initial Kernel_2: " + str(kernel_2))
    print("Buckets_2: " + str(buckets_2))

    # t = time_limit/(1 + len(buckets_1) + len(buckets_2))

    t = time_limit/(1 + len(buckets_1))

    kernel_1, kernel_2, objective, w1, w2 = restricted_CCA(kernel_1, [], kernel_2, [], 0, dataset1, dataset2, k1, k2, t, bigM_estimation=bigM_estimation)

    objectives = [objective]
    w1s = [w1]
    w2s = [w2]

    time_initialization = t

    it = 0

    time_remaining = time_limit - time_initialization
    end_time = time.time()

    while it < NB-1:
        # while it <= 10:

        obj_old = objective

        bucket2 = buckets_2[it]

        kernel_1, kernel_2, obj_new, w2, z2 = restricted_fixing_w1(kernel_1, [], kernel_2, bucket2, max(objectives), w1, k2, dataset1, dataset2, combined=True, bigM_estimation=bigM_estimation)

        end_time = time.time()

        if obj_new < 0:
            w1 = w1s[-1]
            w2 = w2s[-1]

        objectives.append(obj_new)
        w1s.append(w1)
        w2s.append(w2)
        # if abs(obj_old - obj_new) <= 1e-6 or (end_time - start_time) >= time_remaining:
        #     break

        obj_old = obj_new

        # print(w1)
        # print(w2)

        bucket1 = buckets_1[it]

        kernel_1, kernel_2, obj_new, w1, z1 = restricted_fixing_w2(kernel_1, bucket1, kernel_2, [], max(objectives), w2, k1, dataset1, dataset2, combined=True, bigM_estimation = bigM_estimation)

        if obj_new < 0:
            w1 = w1s[-1]
            w2 = w2s[-1]

        objectives.append(obj_new)
        w1s.append(w1)
        w2s.append(w2)

        end_time = time.time()

        # print(w1)
        # print(w2)

        # if abs(obj_old - obj_new) <= 1e-6 or (end_time - start_time) <= time_remaining:
        #     break

        it += 1

    w1 = w1s[np.argmax(objectives)]
    w2 = w2s[np.argmax(objectives)]

    objective = max(objectives)

    end_time = time.time()

    time_elapsed = end_time - start_time

    w1_sol = np.zeros(p1)
    for i in range(p1):
        if i in w1.keys():
            w1_sol[i] = w1[i]

    w2_sol = np.zeros(p2)
    for j in range(p2):
        if j in w2.keys():
            w2_sol[j] = w2[j]

    # print(w1_sol)
    # print(w2_sol)
    # for bucket2 in buckets_2:
    #     print("Kernel 1: " + str(kernel_1))
    #
    #     print("Kernel 2: " + str(kernel_2))
    #     print("Bucket 2: " + str(bucket2))
    #
    #     kernel_1, kernel_2, objective = CCA_restricted(kernel_1, [], kernel_2, bucket2, dataset1, dataset2, k1, k2, objective, t)
    #
    #     print("Objective Value: " + str(objective))
    #
    #     objectives.append(objective)

    # if show:
    #     plt.plot(objectives)
    #     plt.show()

    print("Time Elapsed: " + str(time_elapsed))
    print("Objective Value: " + str(objective))

    return objective, w1_sol, w2_sol, time_elapsed

def pareto_frontier(dataset1, dataset2, k1_min, k1_max, k2_min, k2_max, save_ws = True, time_limit = 1000, name_train="pareto_train", name_test="pareto_test"):

    pareto_train = np.zeros((k1_max, k2_max))
    pareto_test = np.zeros((k1_max, k2_max))

    ws_1 =  []
    ws_2 = []

    for k1 in range(k1_min, k1_max+1):
        for k2 in range(k2_min, k2_max+1):
            objvals = []
            corrs = []

            ws_1_prov = []
            ws_2_prov = []

            for i in range(5):

                training_dataset1, test_dataset1 = train_test_split(dataset1, test_size = 0.7, random_state = i)
                training_dataset2, test_dataset2 = train_test_split(dataset2, test_size=0.7, random_state=i)

                objval, w1, w2, time_elapsed = combined_CCA(training_dataset1, training_dataset2, k1, k2, NB = k1*k2, time_limit = time_limit)

                objvals.append(objval)

                ws_1_prov.append(w1)
                ws_2_prov.append(w2)

                norm1 = np.sqrt(w1.T @ test_dataset1.T @ test_dataset1 @ w1)
                norm2 = np.sqrt(w2.T @ test_dataset2.T @ test_dataset2 @ w2)

                cor = (w1.T @ test_dataset1.T @ test_dataset2 @ w2)/(norm1*norm2)

                corrs.append(cor)

            pareto_train[k1-1, k2-1] = np.mean(objvals)
            pareto_test[k1 - 1, k2 - 1] = np.mean(corrs)

            np.savetxt(name_train + ".csv", pareto_train, delimiter=",")
            np.savetxt(name_test + ".csv", pareto_test, delimiter=",")

            ws_1.append(np.mean(ws_1_prov, axis = 1))
            ws_2.append(np.mean(ws_2_prov, axis=1))

            np.savetxt("w1s.csv", ws_1, delimiter=",")
            np.savetxt("w2s.csv", ws_2, delimiter=",")


def initializing(mode):

    if mode == 0:
        df = pd.read_csv('datasets/music_scaled.csv', sep=",")

        dataset1 = df.iloc[:, 0:34]
        dataset2 = df.iloc[:, 34:68]

        ks = [(3, 3), (5, 5), (10, 10)]

    if mode == 1:
        df = pd.read_csv('datasets/winequalityred_scaled.csv', sep=",")

        dataset1 = df.iloc[:, 0:6]
        dataset2 = df.iloc[:, 6:11]

        ks = [(2, 2), (3, 3)]

    if mode == 2:
        df = pd.read_csv('datasets/winequalitywhite_scaled.csv', sep=",")

        dataset1 = df.iloc[:, 0:6]
        dataset2 = df.iloc[:, 6:11]

        ks = [(2, 2), (3, 3)]

    if mode == 3:
        df = pd.read_csv('yearprediction_scaled.csv', sep=",")

        dataset1 = df.iloc[:, 0:45]
        dataset2 = df.iloc[:, 45:90]

        ks = [(3, 3), (5, 5), (10, 10)]

    if mode == 4:
        df = pd.read_csv('datasets/studentmat_scaled.csv', sep=",")

        dataset1 = df.iloc[:, 0:13]
        dataset2 = df.iloc[:, 13:26]

        ks = [(3, 3), (5, 5)]

    if mode == 5:
        df = pd.read_csv('datasets/studentpor_scaled.csv', sep=",")

        dataset1 = df.iloc[:, 0:13]
        dataset2 = df.iloc[:, 13:26]

        ks = [(3, 3), (5, 5)]

    return dataset1, dataset2, ks