import numpy as np
import random
import math
import csv
from random import seed
from numpy.linalg import inv
import pickle



def loadData(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)

    # work with a smaller dataset
    #dataset = dataset[:10000]

    dataset = np.array(dataset)
    dataset = dataset.astype(np.float)

    return dataset




def divideDatasets(dataset):
    dataset_copy = np.copy(dataset)

    watch_frequency = [dataset_copy[i,0] for i in range(len(dataset_copy))]
    #print(watch_frequency)

    #removing the first column
    dataset_copy = np.delete(dataset_copy, np.s_[:1], axis=1)

    #division of datasets
    train_set = np.copy(dataset_copy)
    test_set = np.copy(dataset_copy)
    validation_set = np.copy(dataset_copy)
    train_validation_set = np.copy(dataset_copy)


    temp = 99.0

    for i in range(len(dataset_copy)):
        for j in range(len(dataset_copy[i])):
            if dataset_copy[i,j] != temp:
                rval = random.uniform(0,1)

                if rval <= 0.6:
                    test_set[i,j] = temp
                    validation_set[i,j] = temp

                elif rval > 0.6 and rval <= 0.8:
                    train_set[i,j] = temp
                    test_set[i,j] = temp

                elif rval > 0.8:
                    train_set[i,j] = temp
                    validation_set[i,j] = temp
                    train_validation_set[i,j] = temp


    return watch_frequency, train_set, validation_set, train_validation_set, test_set, dataset_copy





def doOperation(lambda_u, lambda_v, K, N, M, garbage, curr_dataset):

    lu = lambda_u
    lv = lambda_v
    kval = K

    # put random value in u(1...n)
    U = np.full((N, kval), 1.0)
    V = np.full((kval, M), 1.0)

    for ii in range(len(U)):
        for jj in range(len(U[ii])):
            U[ii, jj] = random.uniform(0, 1)

    curr_error = 0.0
    prev_error = 0.0
    while(True):
        #print('what')
        # update Vm
        Ik = np.eye(kval)
        Ik = np.multiply(Ik, lv)
        for ii in range(M):
            sums_left = np.zeros((kval, kval))

            # have to make Un here
            for jj in range(N):
                if curr_dataset[jj, ii] != garbage:
                    res = np.zeros((kval, kval))
                    Ucopy = np.reshape(U[jj], (kval, 1))
                    res = np.dot(Ucopy, Ucopy.T)
                    #print(res)
                    sums_left = np.add(sums_left, res)

            sums_left = np.add(sums_left, Ik)

            #print(sums_left)

            sums_left = inv(sums_left)

            sums_right = np.zeros((kval, 1))
            for jj in range(N):
                if curr_dataset[jj, ii] != garbage:
                    resy = np.zeros((kval, 1))
                    Ucopy = np.reshape(U[jj], (kval, 1))
                    resy = np.multiply(Ucopy, curr_dataset[jj, ii])
                    #print(resy.shape)
                    sums_right = np.add(sums_right, resy)
                    # print('shape: ', Ucopy.shape, ' ', resy.shape, ' ', sums_right.shape)

            resx = np.dot(sums_left, sums_right)
            # print('result ', resx.shape)
            for jj in range(kval):
                V[jj, ii] = resx[jj]



        # update Un
        Ik = np.eye(kval)
        Ik = np.multiply(Ik, lu)
        for ii in range(N):
            sums_left = np.zeros((kval, kval))

            # have to make Vm here
            for jj in range(M):
                if curr_dataset[ii, jj] != garbage:
                    res = np.zeros((kval, kval))
                    Vcopy = V[:, jj]
                    Vcopy = np.reshape(Vcopy, (kval, 1))
                    # print('vv ', Vcopy.shape)

                    res = np.dot(Vcopy, Vcopy.T)
                    #print('shape ', res.shape)
                    sums_left = np.add(sums_left, res)

            sums_left = np.add(sums_left, Ik)
            sums_left = inv(sums_left)

            # print('sums left: ', sums_left.shape)

            sums_right = np.zeros((kval, 1))
            for jj in range(M):
                if curr_dataset[ii, jj] != garbage:
                    resy = np.zeros((kval, 1))
                    Vcopy = V[:, jj]
                    Vcopy = np.reshape(Vcopy, (kval, 1))

                    resy = np.multiply(Vcopy, curr_dataset[ii, jj])
                    sums_right = np.add(sums_right, resy)
                    #print('shape: ', Vcopy.shape, ' ', resy.shape, ' ', sums_right.shape)

            resx = np.dot(sums_left, sums_right)
            #print('result ', resx.shape)
            for jj in range(kval):
                U[ii, jj] = resx[jj]



        #RMSE error
        prev_error = curr_error

        curr_result = np.dot(U,V)
        #print('curr result ', curr_result.shape)
        sums = 0.0
        cnt = 0
        for ii in range(N):
            for jj in range(M):
                if curr_dataset[ii,jj] != garbage:
                    sums += ((curr_dataset[ii,jj]-curr_result[ii,jj])**2)
                    cnt += 1

        sums = sums/(cnt)
        curr_error = math.sqrt(sums)

        #print(curr_error, ' ', prev_error)

        if abs(((curr_error-prev_error)/(curr_error))) < 0.01:
            break


    #print('Out of loop')
    return U, V, curr_error




def calculate_error_on_dataset(curr_dataset, U, V):

    curr_result = np.dot(U, V)
    #print('curr result ', curr_result.shape)
    sums = 0.0
    cnt = 0
    for ii in range(N):
        for jj in range(M):
            if curr_dataset[ii, jj] != garbage:
                sums += ((curr_dataset[ii, jj] - curr_result[ii, jj]) ** 2)
                cnt += 1

    sums = sums/(cnt)
    curr_error = math.sqrt(sums)

    return curr_error









#main program

seed(1)

filename = 'data/data.csv'
dataset = loadData(filename)



watch_frequency, train_set, validation_set, train_validation_set, test_set, X = divideDatasets(dataset)



lambda_u = [0.01, 0.1, 1.0, 10.0]
lambda_v = lambda_u[:]
K = [5, 10, 20, 40]
N = len(train_set)
M = len(train_set[0])

garbage = 99.0
error_dataset = 0.0

min_error = [1000000.0, 0.0, 0.0, 0.0]
for i in lambda_u:
    #for k in lambda_v:
    for j in K:
        U, V, error_dataset = doOperation(i, i, j, N, M, garbage, train_set)
        #U, V = doOperation(i, k, j, N, M, garbage, train_set)

        curr_error = calculate_error_on_dataset(validation_set, U, V)

        print('Errors: ', i, ' ', i, ' ', j, ' ', error_dataset, ' ', curr_error)

        if curr_error < min_error[0]:
            min_error[0] = curr_error
            min_error[1] = i
            min_error[2] = i
            min_error[3] = j




print('\n\n80 percent training starts\n')
U, V, error_dataset = doOperation(min_error[1], min_error[2], min_error[3], N, M, garbage, train_validation_set)
print('error: ', min_error[1],' ', min_error[2], ' ', min_error[3], ' ', error_dataset)


#pickling to save data
with open('offline.pkl', 'wb') as f:
    pickle.dump(U, f)
    pickle.dump(V, f)

with open('offline.pkl', 'rb') as f:
    U = pickle.load(f)
    V = pickle.load(f)




#final run on test set
print('Error of Model: ', calculate_error_on_dataset(test_set, U, V))
























