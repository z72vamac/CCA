import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from CCA import CCA, multistart_CCA

def benders_CCA(dataset1, dataset2, k1, k2, init=True, best_reponse=False):

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

    lbs = []
    ubs = []



    def subproblem(MODEL, where):


        if where == GRB.Callback.MIPSOL:

            n, p1 = dataset1.shape
            n, p2 = dataset2.shape


            z1 = MODEL.cbGetSolution(MODEL._z1)
            z2 = MODEL.cbGetSolution(MODEL._z2)
            ub = MODEL.cbGetSolution(MODEL._ub)
            lb = MODEL.cbGetSolution(MODEL._lb)

            lbs.append(lb)
            ubs.append(ub)

            print("LB = " + str(lb))
            print("UB = " + str(ub))

            if ub - lb >= 0.05:

                print("\nSolving subproblem by fixing z1 and z2\n")

                z1_indices = [i for i in range(p1) if z1[i] > 0.5]
                z2_indices = [j for j in range(p2) if z2[j] > 0.5]

                dataset1M = dataset1.iloc[:, z1_indices]
                dataset2M = dataset2.iloc[:, z2_indices]

                print(z1)
                print(z2)

                n, u1 = dataset1M.shape
                n, u2 = dataset2M.shape

                if u1 == 0:
                    M = 1 / np.sqrt(min(LA.eig(dataset2M.T @ dataset2M)[0]))
                elif u2 == 0:
                    M = 1 / np.sqrt(min(LA.eig(dataset1M.T @ dataset1M)[0]))
                else:
                    M = max([1 / np.sqrt(min(LA.eig(dataset1M.T @ dataset1M)[0])),
                             1 / np.sqrt(min(LA.eig(dataset2M.T @ dataset2M)[0]))])

                MODEL2 = gp.Model("Subproblem")

                n, p1 = dataset1.shape

                n, p2 = dataset2.shape

                w1 = MODEL2.addMVar(p1, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w1")
                # z1 = MODEL2.addMVar(p1, vtype=GRB.BINARY, name="z1")
                # w2 = MODEL.addMVar(p1, vtype=GRB.BINARY, name="z1")

                w2 = MODEL2.addMVar(p2, vtype=GRB.CONTINUOUS, lb=-M, ub=M, name="w2")
                # z2 = MODEL2.addMVar(p2, vtype=GRB.BINARY, name="z2")

                alpha1 = MODEL2.addMVar(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha1")
                alpha2 = MODEL2.addMVar(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha2")

                prod1 = MODEL2.addMVar(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="prod1")
                prod2 = MODEL2.addMVar(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="prod2")

                beta1 = MODEL2.addMVar(p1, vtype=GRB.CONTINUOUS, lb=0.0, name="beta1")
                beta2 = MODEL2.addMVar(p2, vtype=GRB.CONTINUOUS, lb=0.0, name="beta2")


                gamma1 = MODEL2.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma1")
                gamma2 = MODEL2.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma2")

                MODEL2.addConstrs(w1[j] <=  M*z1[j] for j in range(p1))
                MODEL2.addConstrs(w1[j] >= -M*z1[j] for j in range(p1))

                MODEL2.addConstrs(w2[j] <=  M*z2[j] for j in range(p2))
                MODEL2.addConstrs(w2[j] >= -M*z2[j] for j in range(p2))

                MODEL2.addConstr(w1 @ product11 @ w1 <= 1)
                MODEL2.addConstr(w2 @ product22 @ w2 <= 1)

                product11w1 = product11 @ w1
                product22w2 = product22 @ w2

                product12w2 = product12 @ w2
                product12w1 = product12.T @ w1

                # MODEL.addConstr(np.ones(p2) @ z2 <= k2)
                MODEL2.addConstrs(beta1[i] == alpha1[i] + gp.quicksum(product12[i, j]*w2[j] for j in range(p2)) - 2*gp.quicksum(product11[i, j]*gamma1*w1[i] for j in range(p1)) for i in range(p1))
                MODEL2.addConstrs(beta2[j] == alpha2[j] + gp.quicksum(product12[i, j]*w1[i] for i in range(p1)) - 2*gp.quicksum(product22[i, j]*gamma2*w2[j] for i in range(p1)) for j in range(p2))

                MODEL2.setObjective(w1 @ product12 @ w2, GRB.MAXIMIZE)
                # MODEL.setObjective(np.ones(p1) @ w1, GRB.MAXIMIZE)

                MODEL2.Params.NonConvex = 2

                # MODEL2.write("subproblem.lp")
                # MODEL2.Params.MIPGap = 1e-2
                MODEL2.Params.MIPGap = 0.05

                MODEL2.optimize()

                objval_subproblem = MODEL2.ObjVal
                print("\nObjval Subproblem: " + str(objval_subproblem))


                MODEL.cbLazy(MODEL._lb >= objval_subproblem)

                MODEL.cbLazy(MODEL._ub <= objval_subproblem
                            #+ gp.quicksum(alpha1[i]* w1[i] for i in range(p1) if z1old[i] < 0.5)
                            + gp.quicksum(alpha1.X[i]*M*MODEL._z1[i] for i in range(p1) if z1[i] < 0.5)
                            #+ gp.quicksum(alpha2[j] * w2[j] for j in range(p2) if z2old[j] < 0.5)
                            + gp.quicksum(alpha2.X[j]*M*MODEL._z2[j] for j in range(p2) if z2[j] < 0.5)
                            # - gp.quicksum(beta1[i] * w1[i] for i in range(p1) if z1old[i] < 0.5)
                            + gp.quicksum(beta1.X[i] *M* MODEL._z1[i] for i in range(p1) if z1[i] < 0.5)
                            # - gp.quicksum(beta2[j] * w2[j] for j in range(p2) if z2old[j] < 0.5)
                            + gp.quicksum(beta2.X[j] * M * MODEL._z2[j] for j in range(p2) if z2[j] < 0.5))



    MODEL = gp.Model("Initializing Model")

    n, p1 = dataset1.shape

    n, p2 = dataset2.shape

    z1 = MODEL.addVars(p1, vtype=GRB.BINARY, name="z1")
    z2 = MODEL.addVars(p2, vtype=GRB.BINARY, name="z2")
    ub = MODEL.addVar(vtype=GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name="ub")
    lb = MODEL.addVar(vtype=GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name="lb")


    if init:
        w1, w2, objective = CCA(X1, X2, k, k, best_reponse=True)

        for i in range(p1):
            if w1[i] != 0:
                z1[i].start = 1
            else:
                z1[i].start = 0

        for j in range(p2):
            if w2[j] != 0:
                z2[j].start = 1
            else:
                z2[j].start = 0

        MODEL.addConstr(lb >= objective)


    # MODEL.addConstr(lb >= objective)
    # for i in range(p1):
    #     if w1[i] != 0:
    #         z1[i].start = 1
    #
    # for j in range(p2):
    #     if w2[j] != 0:
    #         z2[j].start = 1

    MODEL.addConstr(z1.sum('*') == k1)
    MODEL.addConstr(z2.sum('*') == k2)

    MODEL.addConstr(lb <= ub)

    # MODEL.Params.NonConvex = 2
    MODEL._z1 = z1
    MODEL._z2 = z2
    MODEL._lb = lb
    MODEL._ub = ub

    MODEL.Params.LazyConstraints = 1
    MODEL.Params.TimeLimit = 600

    MODEL.setObjective(ub, GRB.MAXIMIZE)

    MODEL.optimize(subproblem)

    MODEL.write('model.lp')

    # print(MODEL.ObjVal)

    if MODEL.Status == 3:
        MODEL.computeIIS()
        MODEL.write("infeasible.ilp")

    print(max(lbs))

    plt.plot(lbs)
    plt.plot(ubs)

    plt.show()

    return MODEL, MODEL.ObjVal, np.array(MODEL._z1), np.array(MODEL._z2)

    # def initial_model():
    #
    #
    #
    #
    #     MODEL._z1 = z1
    #     MODEL._z2 = z2
    #     MODEL._nu = nu
    #
    #
    #
    #     product12 = X1.T @ X2
    #     product11 = X1.T @ X1
    #     product22 = X2.T @ X2
    #
    #     np.random.seed(1)
    #     arr1 = np.array([1] * k1 + [0] * (p1 - k1))
    #     np.random.shuffle(arr1)
    #
    #     for i in range(p1):
    #         z1[i] = arr1[i]
    #
    #     arr2 = np.array([1] * k2 + [0] * (p2 - k2))
    #     np.random.shuffle(arr2)
    #
    #     for i in range(p2):
    #         z2[i] = arr2[i]
    #
    #     objval_subproblem, w1, w2, alpha1, alpha2, beta1, beta2 = subproblem(z1, z2)
    #     # MODEL.addConstr(w1 @ product11 @ w1 <= 1)
    #
    #     MODEL.setObjective(nu, GRB.MAXIMIZE)
    #
    #     return MODEL, objval_subproblem, w1, w2, alpha1, alpha2, beta1, beta2
    #
    # MODEL, objval_subproblem, w1, w2, alpha1, alpha2, beta1, beta2 = initial_model()
    #
    # objvals = []

    # for i in range(100):
    #     MODEL, MODEL.ObjVal, z1, z2 = master_problem(MODEL, objval_subproblem, w1, w2, alpha1, alpha2, beta1, beta2)
    #     objval_subproblem, w1, w2, alpha1, alpha2, beta1, beta2 = subproblem(z1, z2)
    #
    #     objvals.append(objval_subproblem)
    #
    # print(objvals)
    # print(max(objvals))
    # plt.plot(objvals)
    # plt.show()

# df = pd.read_csv('winequalityred_scaled.csv', sep=",")
df = pd.read_csv('music_scaled.csv', sep=",")

# training_data = df.sample(frac=0.7, random_state=6)
# X = training_data.iloc[:, 0:6]
# Y = training_data.iloc[:, 6:13]

X = df.iloc[:, 0:34]
Y = df.iloc[:, 34:68]

k = 3

benders_CCA(X, Y, k, k, init=False, best_reponse=True)


# multistart_CCA(X, Y, k, k, best_reponse=True)
