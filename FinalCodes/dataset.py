import numpy as np
import csv
import os
from ADMM_primal import *

def EuclidCost(n,p):
    C=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            C[i,j]=pow(i-j,p)
            if i==j:
                C[i,j]=2*C[i,j]
    return C+C.T


def DOTmark_CostWeight(n,ImageClass):
    path=os.getcwd()
    index=np.random.choice(range(1,10),2,replace=None)
    print(index)
    w=[]
    for i in index:
        with open(path+'/dataset/'+ImageClass+'/data32_100'+str(i)+'.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                w.append(row)
        csvfile.close()
    w=np.array(w, np.float32).reshape((2, n*n))
    v=w[0,:]/sum(w[0,:])
    u=w[1,:]/sum(w[1,:])
    c=np.zeros((n*n,n*n))
    for i in range(n*n):
        for j in range(i,n*n):
            c[i,j]=(v[i]-u[j])**2
            if i==j:
                c[i,j]=2*c[i,j]
    return v,u,c+c.T


def Random_Cost(n):
    return np.random.random((n,n))

def Random_Weight(n):
    v=np.random.random(n)
    return v/sum(v)

#c=Random_Cost(10)
#mu=Random_Weight(10)
#nu=Random_Weight(10)
mu,nu,c=DOTmark_CostWeight(32,'Shapes')
ADMM_primal(c,mu,nu,1000,1e-3,1e-1)

