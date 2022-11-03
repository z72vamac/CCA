
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from CCA import CCA, multistart_CCA


def benders_CCA(dataset1, dataset2, k1, k2, init = True, max_iter = 2000, best_reponse=False, show = True):

    print("*******")
    print("Computing the benders_CCA. k1 = " + str(k1) + "; k2 = " + str(k2))
    print("*******")

    M = max([1/np.sqrt(min(LA.eig(dataset1.T @ dataset1)[0])), 1/np.sqrt(min(LA.eig(dataset2.T @ dataset2)[0]))])

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape

    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1
    product22 = X2.T @ X2

    def subproblem(MODEL, z1, z2):

        print("\n\nSolving subproblem by fixing z1 and z2\n")

        MODEL2 = gp.Model("Subproblem")

        n, p1 = dataset1.shape

        n, p2 = dataset2.shape

        z1_indices = [i for i in range(p1) if z1[i] > 0.5]
        z2_indices = [j for j in range(p2) if z2[j] > 0.5]

        print(z1)
        print(z2)

        dataset1M = dataset1.iloc[:, z1_indices]
        dataset2M = dataset2.iloc[:, z2_indices]

        n, u1 = dataset1M.shape
        n, u2 = dataset2M.shape

        if u1 == 0:
            M = 1 / np.sqrt(min(LA.eig(dataset2M.T @ dataset2M)[0]))
        elif u2 == 0:
            M = 1 / np.sqrt(min(LA.eig(dataset1M.T @ dataset1M)[0]))
        else:
            M = max([1 / np.sqrt(min(LA.eig(dataset1M.T @ dataset1M)[0])),
                     1 / np.sqrt(min(LA.eig(dataset2M.T @ dataset2M)[0]))])

        w1 = MODEL2.addVars(p1, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w1")
        # z1 = MODEL2.addMVar(p1, vtype=GRB.BINARY, name="z1")
        # w2 = MODEL.addMVar(p1, vtype=GRB.BINARY, name="z1")

        w2 = MODEL2.addVars(p2, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w2")
        # z2 = MODEL2.addMVar(p2, vtype=GRB.BINARY, name="z2")

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

        # MODEL2.addConstrs(w1[j] <=  M*z1[j] for j in range(p1))
        # MODEL2.addConstrs(w1[j] >= -M*z1[j] for j in range(p1))
        #
        # MODEL2.addConstrs(w2[j] <=  M*z2[j] for j in range(p2))
        # MODEL2.addConstrs(w2[j] >= -M*z2[j] for j in range(p2))

        MODEL2.addConstr(gp.quicksum(w1[i] * product11[i, j] * w1[j] for i in range(p1) for j in range(p1)) <= 1)
        MODEL2.addConstr(gp.quicksum(w2[i] * product22[i, j] * w2[j] for i in range(p2) for j in range(p2)) <= 1)

        # product12prod1 = [gp.quicksum(product12[i, j]*z2old[j]*w2[j] for j in range(p2)) for i in range(p1)]
        # product11w1 = product11 @ w1
        # product12prod2 = [gp.quicksum(product12[i, j] * z1old[i] * w1[i] for i in range(p1)) for j in range(p2)]
        # product22w2 = product22 @ w2

        # product12w2 = product12 @ w2
        # product12w1 = product12.T @ w1

        # MODEL.addConstr(np.ones(p2) @ z2 <= k2)
        MODEL2.addConstrs(beta1[i] == alpha1[i] + gp.quicksum(product12[i, j]*w2[j] for j in range(p2)) - 2*gp.quicksum(product11[i, j]*gamma1*w1[i] for j in range(p1)) for i in range(p1))
        MODEL2.addConstrs(beta2[j] == alpha2[j] + gp.quicksum(product12[i, j]*w1[i] for i in range(p1)) - 2*gp.quicksum(product22[i, j]*gamma2*w2[j] for i in range(p1)) for j in range(p2))



        # MODEL.addConstrs(w2[j] <= M * z2[j] for j in range(p2))
        # MODEL.addConstrs(w2[j] >= -M * z2[j] for j in range(p2))

        MODEL2.setObjective(gp.quicksum(w1[i] * product12[i, j] * w2[j] for i in range(p1) for j in range(p2)), GRB.MAXIMIZE)
        # MODEL.setObjective(np.ones(p1) @ w1, GRB.MAXIMIZE)

        MODEL2.Params.NonConvex = 2

        MODEL2.Params.MIPGap = 1e-2
        # MODEL2.Params.TimeLimit = 5
        # MODEL2.write('subproblem.lp')

        MODEL2.optimize()

        objval_subproblem = MODEL2.ObjVal
        print("\nObjval Subproblem: " + str(objval_subproblem))

        # MODEL.addConstr(MODEL._ub <= 1)
        alpha1_sol = np.zeros(p1)
        alpha2_sol = np.zeros(p2)
        beta1_sol = np.zeros(p1)
        beta2_sol = np.zeros(p2)

        for i in range(p1):
            if z1[i] > 0.5:
                alpha1_sol[i] = 0
                beta1_sol[i] = 0
            else:
                alpha1_sol[i] = alpha1[i].X
                beta1_sol[i] = beta1[i].X

        for j in range(p2):
            if z2[j] > 0.5:
                alpha2_sol[j] = 0
                beta2_sol[j] = 0
            else:
                alpha2_sol[j] = alpha2[j].X
                beta2_sol[j] = beta2[j].X

        MODEL.update()

        # print(np.array(z1.X))
        # print(np.array(z2.X))

        if MODEL.Status == 3:
            MODEL.computeIIS()
            MODEL.write("infeasible.ilp")

        return objval_subproblem, alpha1_sol, alpha2_sol, beta1_sol, beta2_sol


    MODEL = gp.Model("Initializing Model")

    n, p1 = dataset1.shape

    n, p2 = dataset2.shape

    z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")
    z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")
    ub = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="ub")
    lb = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="lb")

    z1_old = np.zeros(p1)
    z2_old = np.zeros(p2)

    MODEL.addConstr(z1.sum('*') <= k1)
    MODEL.addConstr(z2.sum('*') <= k2)

    MODEL.addConstr(lb <= ub)

    product12 = X1.T @ X2
    product11 = X1.T @ X1
    product22 = X2.T @ X2

    np.random.seed(1)

    if init:
        w1, w2, objective = CCA(X1, X2, k, k, best_reponse=True)

        for i in range(p1):
            if w1[i] != 0:
                z1_old[i] = 1
            else:
                z1_old[i] = 0

        for j in range(p2):
            if w2[j] != 0:
                z2_old[j] = 1
            else:
                z2_old[j] = 0

        MODEL.addConstr(lb >= objective)

    else:
        arr1 = np.array([1] * k1 + [0] * (p1 - k1))
        np.random.shuffle(arr1)

        for i in range(p1):
            z1_old[i] = arr1[i]

        arr2 = np.array([1] * k2 + [0] * (p2 - k2))
        np.random.shuffle(arr2)

        for i in range(p2):
            z2_old[i] = arr2[i]

        lb.start = 0
        ub.start = 1

    MODEL.setObjective(ub, GRB.MAXIMIZE)

    MODEL._z1 = z1
    MODEL._z2 = z2
    MODEL._ub = ub
    MODEL._lb = lb

    MODEL.update()


    eps = 0.001
    LB = 0
    UB = 1

    lbs = [LB]
    ubs = [UB]
    iter = 0

    while abs(UB - LB) > eps and iter < max_iter:
        objval_subproblem, alpha1, alpha2, beta1, beta2 = subproblem(MODEL, z1_old, z2_old)


        MODEL.addConstr(MODEL._lb >= LB)
        MODEL.addConstr(MODEL._ub <= objval_subproblem
                        #+ gp.quicksum(alpha1[i]* w1[i] for i in range(p1) if z1old[i] < 0.5)
                        + gp.quicksum(alpha1[i]*M*MODEL._z1[i] for i in range(p1) if z1_old[i] < 0.5)
                        #+ gp.quicksum(alpha2[j] * w2[j] for j in range(p2) if z2old[j] < 0.5)
                        + gp.quicksum(alpha2[j]*M*MODEL._z2[j] for j in range(p2) if z2_old[j] < 0.5)
                        # - gp.quicksum(beta1[i] * w1[i] for i in range(p1) if z1old[i] < 0.5)
                        + gp.quicksum(beta1[i] *M* MODEL._z1[i] for i in range(p1) if z1_old[i] < 0.5)
                        # - gp.quicksum(beta2[j] * w2[j] for j in range(p2) if z2old[j] < 0.5)
                        + gp.quicksum(beta2[j] * M * MODEL._z2[j] for j in range(p2) if z2_old[j] < 0.5))
        # MODEL.write('model.lp')
        MODEL.optimize()

        print(z1)
        for i in range(p1):
            if z1[i].X > 0.5:
                z1_old[i] = 1
            else:
                z1_old[i] = 0

        for j in range(p2):
            if z2[j].X > 0.5:
                z2_old[j] = 1
            else:
                z2_old[j] = 0

        LB = max(LB, objval_subproblem)
        UB = min(UB, MODEL.ObjVal)

        print('LB = ' + str(LB))
        print('UB = ' + str(UB))

        lbs.append(LB)
        ubs.append(UB)

        iter += 1

    print(LB)
    print(UB)

    if show:
        plt.plot(lbs)
        plt.plot(ubs)
        plt.show()

    return LB