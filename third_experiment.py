"""
Checking if fixing variables takes the highest value in the objective function
"""
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




# print(X)
# print(Y)


def CCA(dataset1, dataset2, k1, k2, best_reponse=False):

    print("*******")
    print("Computing the CCA. k1 = " + str(k1) + "; k2 = " + str(k2))
    print("*******")

    M = max([1/np.sqrt(min(LA.eig(dataset1.T @ dataset1)[0])), 1/np.sqrt(min(LA.eig(dataset2.T @ dataset2)[0]))])

    def fixing_w1(w1):

        print("Fixing w1")

        MODEL = gp.Model("Fixing w1")

        n, p1 = dataset1.shape

        n, p2 = dataset2.shape

        # w1 = MODEL.addMVar(p1, vtype=GRB.CONTINUOUS, name="w1")
        # z1 = MODEL.addMVar(p1, vtype=GRB.BINARY, name="z1")

        w2 = MODEL.addMVar(p2, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w2")
        z2 = MODEL.addMVar(p2, vtype=GRB.BINARY, name="z2")

        product12 = X1.T @ X2
        product11 = X1.T @ X1
        product22 = X2.T @ X2

        MODEL.addConstr(w2 @ product22 @ w2 <= 1)

        MODEL.addConstr(np.ones(p2) @ z2 <= k2)

        MODEL.addConstrs(w2[j] <= M * z2[j] for j in range(p2))
        MODEL.addConstrs(w2[j] >= -M * z2[j] for j in range(p2))

        objective = w1 @ product12 @ w2

        MODEL.setObjective(w1 @ product12 @ w2, GRB.MAXIMIZE)

        # MODEL.setObjective(np.ones(p1) @ w1, GRB.MAXIMIZE)

        # MODEL.Params.NonConvex = 2

        MODEL.Params.MIPGap = 1e-2
        MODEL.Params.OutputFlag = 0

        MODEL.optimize()

        print(MODEL.getObjective())
        print(np.array(w2.X))

        return MODEL.ObjVal, np.array(w2.X), np.array(z2.X)

    def fixing_w2(w2):

        print("Fixing w2")

        MODEL = gp.Model("Fixing w2")

        n, p1 = dataset1.shape

        n, p2 = dataset2.shape

        w1 = MODEL.addMVar(p1, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w1")
        z1 = MODEL.addMVar(p1, vtype=GRB.BINARY, name="z1")

        # w2 = MODEL.addMVar(p2, vtype=GRB.CONTINUOUS, name="w2")
        # z2 = MODEL.addMVar(p2, vtype=GRB.BINARY, name="z2")

        product12 = X1.T @ X2
        product11 = X1.T @ X1
        product22 = X2.T @ X2

        MODEL.addConstr(w1 @ product11 @ w1 <= 1)

        MODEL.addConstr(np.ones(p1) @ z1 <= k1)

        # M = 1

        MODEL.addConstrs(w1[j] <= M * z1[j] for j in range(p1))
        MODEL.addConstrs(w1[j] >= -M * z1[j] for j in range(p1))

        objective = w1 @ (product12 @ w2)

        MODEL.setObjective(objective, GRB.MAXIMIZE)
        # MODEL.setObjective(np.ones(p1) @ w1, GRB.MAXIMIZE)

        # MODEL.Params.NonConvex = 2
        MODEL.Params.MIPGap = 1e-2
        MODEL.Params.OutputFlag = 0


        MODEL.optimize()

        print(MODEL.getObjective())
        print(np.array(w1.X))


        return MODEL.ObjVal, np.array(w1.X), np.array(z1.X)


    MODEL = gp.Model("CCA")

    n, p1 = dataset1.shape
    n, p2 = dataset2.shape


    w1 = MODEL.addMVar(p1, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w1")
    z1 = MODEL.addMVar(p1, vtype=GRB.BINARY, name="z1")

    w2 = MODEL.addMVar(p2, vtype=GRB.CONTINUOUS, lb = -M, ub = M, name="w2")
    z2 = MODEL.addMVar(p2, vtype=GRB.BINARY, name="z2")


    X1 = np.array(dataset1)
    X2 = np.array(dataset2)

    product12 = X1.T @ X2
    product11 = X1.T @ X1
    product22 = X2.T @ X2

    MODEL.addConstr(w1 @ product11 @ w1 <= 1.0)
    MODEL.addConstr(w2 @ product22 @ w2 <= 1.0)

    MODEL.addConstr(np.ones(p1) @ z1 <= k1)
    MODEL.addConstr(np.ones(p2) @ z2 <= k2)

    MODEL.addConstrs(w1[j] <= M*z1[j] for j in range(p1))
    MODEL.addConstrs(w1[j] >= -M*z1[j] for j in range(p1))

    MODEL.addConstrs(w2[j] <= M*z2[j] for j in range(p2))
    MODEL.addConstrs(w2[j] >= -M*z2[j] for j in range(p2))

    MODEL.setObjective(w1 @ product12 @ w2, GRB.MAXIMIZE)


    # MODEL.Params.FeasibilityTol = 1e-9
    MODEL.Params.NonConvex = 2
    MODEL.Params.TimeLimit = 10
    # MODEL.Params.MIPGap = 0.05

    MODEL.optimize()

    obj_old = 0
    obj_new = MODEL.ObjVal

    objectives = [obj_old, obj_new]

    if best_reponse:

        w1 = np.array(w1.X)

        it = 0
        while it <= 3: #and abs(obj_old - obj_new) >= 1e-2:
            # while it <= 10:

                obj_old = obj_new
                obj_new, w2, z2 = fixing_w1(w1)
                obj_new, w1, z1 = fixing_w2(w2)
                objectives.append(obj_new)
                it += 1

            # print(objectives)
            # print(max(objectives))
            # plt.plot(objectives[1:])
            # plt.show()
            # initial_solution(z1, z2, dataset1, dataset2)

    return w1, w2, objectives[-1]



df = pd.read_csv('music_scaled.csv', sep=",")

X = df.iloc[:, 0:34]
Y = df.iloc[:, 34:68]

CCA(X, Y, 4, 4, best_reponse=True)

# print(dataset)


# pareto_train = np.zeros((34, 34))
# pareto_test = np.zeros((34, 34))
#
# for k1 in range(1, 35):
#     for k2 in range(1, 35):
#         objvals = []
#         corrs = []
#         for i in range(5):
#             training_data = df.sample(frac=0.7, random_state=i)
#             X = training_data.iloc[:, 0:34]
#             Y = training_data.iloc[:, 34:68]
#
#             w1, w2, objval = CCA(X, Y, k1, k2, best_reponse=True)
#
#             objvals.append(objval)
#
#             testing_data = df.drop(training_data.index)
#             X_test = testing_data.iloc[:, 0:34]
#             Y_test = testing_data.iloc[:, 34:68]
#
#             norm1 = np.sqrt(w1.T @ X_test.T @ X_test @ w1)
#             norm2 = np.sqrt(w2.T @ Y_test.T @ Y_test @ w2)
#
#             cor = (w1.T @ X_test.T @ Y_test @ w2)/(norm1*norm2)
#
#
#             corrs.append(cor)
#
#         pareto_train[k1-1, k2-1] = np.mean(objvals)
#         pareto_test[k1 - 1, k2 - 1] = np.mean(corrs)
#
#         np.savetxt("pareto_train.csv", pareto_train, delimiter=",")
#         np.savetxt("pareto_test.csv", pareto_test, delimiter=",")

        # print(cor)





# from sklearn.cross_decomposition import CCA
#
# # X = dataset.iloc[:, 0:4]
# # Y = dataset.iloc[:, 4:8]
#
# cca = CCA(n_components=3)
# cca.fit(X, Y)
#
# print(cca.x_weights_)
#
# X_c, Y_c = cca.transform(X, Y)
#
# print(np.corrcoef(X_c[:, 0], Y_c[:, 0]))
#
# # print(cca.coef_)
