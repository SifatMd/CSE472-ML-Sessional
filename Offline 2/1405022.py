import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import multivariate_normal
from random import seed
import time
from matplotlib.patches import Ellipse



seed(1)

def load_data(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)

    #work with a smaller dataset
    #dataset = dataset[:400]

    dataset = np.array(dataset)
    dataset = dataset.astype(np.float)

    return dataset





filename = "data.csv"
dataset = load_data(filename)


#transpose
dataset_transpose = np.transpose(dataset)

#calculate covariance matrix
cov_matrix = np.cov(dataset_transpose)



#calculate eigen value and vectors
eigen_value, eigen_vector = np.linalg.eig(cov_matrix)


eigen_vector_2d = eigen_vector[:,0:2]

eigen_vector_2d = eigen_vector_2d.real

#print(len(dataset), len(dataset[0]), len(eigen_vector_2d), len(eigen_vector_2d[0]))

#print(dataset.shape, eigen_vector_2d.shape)
dot_product = np.dot(dataset, eigen_vector_2d)

dot_product = dot_product.real
all_points = dot_product.tolist()


#print(all_points)

#graph showing
#plt.scatter(dot_product[:,0], dot_product[:,1])
#plt.show()



#EM algorithm

def close_event():
    plt.close()

def drawGraph():
    for i in range(len(Pik)):
        index = Pik[i].index(max(Pik[i]))
        if index == 0:
            plt.scatter(all_points[i][0], all_points[i][1], color='orange')

        elif index == 1:
            plt.scatter(all_points[i][0], all_points[i][1], color='lime')

        elif index == 2:
            plt.scatter(all_points[i][0], all_points[i][1], color='cyan')


    for i in range(3):
        plt.scatter(meu[i][0], meu[i][1], color="black")


    for i in range(3):
        nparr = []
        cc = 2
        nparr.append(np.array([cc*gmm_cov[i][0], cc*gmm_cov[i][1]]))
        nparr.append(np.array([cc*gmm_cov[i][2], cc*gmm_cov[i][3]]))
        nparr = np.array(nparr)


        vals, vecs = np.linalg.eigh(nparr)

        # Compute "tilt" of ellipse using first eigenvector
        x, y = vecs[:, 0]
        theta = np.degrees(np.arctan2(y, x))

        # Eigenvalues give length of ellipse along each eigenvector
        w, h = 2 * np.sqrt(vals)

        ax = plt.subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ellipse = Ellipse(meu[i], w, h, theta, color="purple")
        ellipse.set_clip_box(ax.bbox)
        ellipse.set_alpha(0.2)
        ax.add_artist(ellipse)




    #plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()





def prob_sample(cov_mat, xi, meu_k):
    #print('res: ', abs(cov_mat[0]*cov_mat[3] - cov_mat[1]*cov_mat[2]))
    res = 1.0/float(math.sqrt(((2*math.pi)**2)*(abs(float(cov_mat[0]*cov_mat[3]) - float(cov_mat[1]*cov_mat[2])))))
    p1 = float(xi[0] - meu_k[0])
    p2 = float(xi[1] - meu_k[1])

    #Inverse the cov_mat
    det = float(cov_mat[0]*cov_mat[3] - cov_mat[1]*cov_mat[2])
    temp = cov_mat[0]
    cov_mat[0] = cov_mat[3]
    cov_mat[3] = temp
    cov_mat[1] = -cov_mat[1]
    cov_mat[2] = -cov_mat[2]
    for ii in range(4):
        cov_mat[ii] = float(cov_mat[ii]/det)

    t1 = float(p1*cov_mat[0] + p2*cov_mat[2])
    t2 = float(p1*cov_mat[1] + p2*cov_mat[3])
    t3 = float(t1*p1) + float(t2*p2)

    e = float((-0.5)*(t3))

    if t3 > 500:
        e = 10**50
        res = res*(10**60)

    elif t3 < (-500):
        e = (-10)*50
        res = res *((-10)**60)

    else:
       res = res*math.exp(e)

    #print('res ', res)
    return res



#random initialization

D = 3
cnt = 0
meu = []
gmm_cov = []
weights = []
minx = 1000.0
maxx = (-1000.0)
miny = 1000.0
maxy = (-1000.0)

for i in range(len(all_points)):
    if all_points[i][0] < minx:
        minx = all_points[i][0]
    if all_points[i][0] > maxx:
        maxx = all_points[i][0]

    if all_points[i][1] < miny:
        miny = all_points[i][1]
    if all_points[i][1] > maxy:
        maxy = all_points[i][1]


temp = []
for i in range(D):
    temp = []
    temp.append(random.uniform(minx,maxx))
    temp.append(random.uniform(miny,maxy))
    meu.append(temp)


#print('meu: ', meu)


for i in range(D):
    temp = []
    temp.append(random.uniform(0,1))
    temp.append(random.uniform(0,1))
    temp.append(temp[1])
    temp.append(random.uniform(0,1))
    gmm_cov.append(temp)





weights.append(.33)
weights.append(.33)
weights.append(.33)
sumw = 0.0
for i in range(D):
    sumw += weights[i]
for i in range(D):
    weights[i] /= sumw



sums = 0.0

for i in range(len(all_points)):
    sumProd = 0.0
    for j in range(D):
        c_m = gmm_cov[j]
        xi = all_points[i]
        meu_k = meu[j]
        sumProd += float(weights[j]*(prob_sample(c_m[:], xi[:], meu_k[:])))

    sums += float(np.log(sumProd))


prev_val = 0.0
curr_val = sums




Pik = []

for i in range(len(all_points)):
    temp = [0.0,0.0,0.0]
    Pik.append(temp)





while(True):

    # E step

    for i in range(len(all_points)):
        sums = 0.0
        for j in range(D):
            c_m = gmm_cov[j]
            xi = all_points[i]
            meu_k = meu[j]

            num = prob_sample(c_m[:], xi[:], meu_k[:])
            # print('num ', num)
            num = float(num * weights[j])
            sums += num
            Pik[i][j] = num

        for j in range(D):
            Pik[i][j] /= sums

        # print('sum: ', sums)







    #M step
    #meu update
    for i in range(D):
        for k in range(2):
            sumPik = 0.0
            for j in range(len(Pik)):
                  sumPik += float(Pik[j][i])

            sumProd = 0.0
            for j in range(len(Pik)):
                sumProd += float(Pik[j][i]*all_points[j][k])

            meu[i][k] = float(sumProd/sumPik)



    #cov-matrix update
    for i in range(D):
        sumPik = 0.0
        for j in range(len(Pik)):
            sumPik += float(Pik[j][i])

        res = [0.0,0.0,0.0,0.0]

        for j in range(len(Pik)):
            p1 = Pik[j][i]

            t1 = float(all_points[j][0] - meu[i][0])
            t2 = float(all_points[j][1] - meu[i][1])

            q1 = float(t1*t1)
            q2 = float(t1*t2)
            q3 = float(t2*t1)
            q4 = float(t2*t2)

            q1 *= p1
            q2 *= p1
            q3 *= p1
            q4 *= p1

            res[0] += float(q1)
            res[1] += float(q2)
            res[2] += float(q3)
            res[3] += float(q4)


        for j in range(4):
            res[j] = float(res[j]/sumPik)

        gmm_cov[i] = res





    #weights udpate
    for i in range(D):
        sumPik = 0.0
        for j in range(len(Pik)):
            sumPik += float(Pik[j][i])

        weights[i] = float(sumPik/len(all_points))

    sumw = 0.0
    for i in range(D):
        sumw += weights[i]
    for i in range(D):
        weights[i] /= sumw




    #check convergence

    prev_val = curr_val
    sums = 0.0
    for i in range(len(all_points)):
        sumProd = 0.0
        for j in range(D):
            c_m = gmm_cov[j]
            xi = all_points[i]
            meu_k = meu[j]
            sumProd += float(weights[j]*(prob_sample(c_m[:], xi[:], meu_k[:])))

        sums += float(np.log(sumProd))


    curr_val = sums

    #print((curr_val))
    if float(abs(curr_val - prev_val)) < 0.0000001:
        #print('here')
        break


    cnt += 1

    #print('cnt: ', cnt, ' ', (cnt%10))
    if (cnt%5) == 0:
        drawGraph()








print('complete ', cnt)

for i in range(len(Pik)):
    index = Pik[i].index(max(Pik[i]))
    if index == 0:
        plt.scatter(all_points[i][0], all_points[i][1], color='orange')

    elif index == 1:
        plt.scatter(all_points[i][0], all_points[i][1], color='lime')

    elif index == 2:
        plt.scatter(all_points[i][0], all_points[i][1], color='cyan')

for i in range(3):
    plt.scatter(meu[i][0], meu[i][1], color="black")

for i in range(3):
    nparr = []
    cc = 2
    nparr.append(np.array([cc*gmm_cov[i][0], cc*gmm_cov[i][1]]))
    nparr.append(np.array([cc*gmm_cov[i][2], cc*gmm_cov[i][3]]))
    nparr = np.array(nparr)


    vals, vecs = np.linalg.eigh(nparr)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)

    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ellipse = Ellipse(meu[i], w, h, theta, color="purple")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

plt.show()


print(weights)
print(meu)
print(gmm_cov)








