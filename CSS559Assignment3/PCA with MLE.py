#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#Importing the data
pima=pd.read_csv("diabetes.csv")


# In[3]:


#function which takes "mean" and "covariance" as the parameters and returns the likelihood of the Feature Vector
def likelihood(x,mu,co):
    #inverse of the covariance matrix
    inv=np.linalg.inv(co)
    p1=1/(np.sqrt(((2*np.pi)**3)*np.linalg.det(co)))
    p2=np.exp(-0.5*np.dot(np.dot((x-mu).T,inv),(x-mu)))
    p=p1*p2
    return p


# In[4]:


#function to find the accuracy of the model upon test data
def accuracy(X_test,MeanVectorA,meanVectorB,covA,covB,prior_prob_A,prior_prob_B):
    X_testA=X_test.iloc[:,:8]
    Xtest_pca=np.dot(X_testA,Eigen_vectors)
    count=0
    ##Testing one test sample on the classsifier
    for i in range(X_test.shape[0]):
        #a=X_trainA.iloc[25,:3].to_numpy().reshape(X_trainA.shape[1]-1,1)
        a=Xtest_pca[i,:].reshape(Xtest_pca.shape[1],1)
        postA=likelihood(a,MeanVectorA.T,covA)*prior_prob_A
        postB=likelihood(a,MeanVectorB.T,covB)*prior_prob_B
        if(postA<postB):
            if(X_test.iloc[i,8]==1):
                count+=1
        else:
            if(X_test.iloc[i,8]==0):
                count+=1
    accurate=count/X_test.shape[0]
    return accurate
    


# In[8]:


A=[]
for i in range(10):
    X_train,X_test=train_test_split(pima, test_size=0.5)
    X=X_train.iloc[:,0:8]
    #Re-center:subtracting mean from each row of X
    MeanVector=np.mean(X,axis=0)
    MeanVector=MeanVector.to_numpy().reshape(MeanVector.shape[0],1)
    X_1=X-MeanVector.T
    #calculating the variance of the dataset
    cov_X=np.cov(X_1.T)
    eigen_vectors=np.linalg.eig(cov_X)
    [v,ev,u]=np.linalg.svd(cov_X)
    #we have calculated the eigen vectors using SVD and took first 3 eigen vectors which corresponds ot highest eigen values
    Eigen_vectors=v[:,:3]
    #we transforming the normal components of the samples onto principal vectors
    X_PCA=np.dot(X,Eigen_vectors)
    data = {'Feature1':X_PCA[:,0],'Feature2':X_PCA[:,1],'Feature3':X_PCA[:,2],'Outcome':X_train["Outcome"].to_numpy()}
    df = pd.DataFrame(data)
    #performing MLE classification upon transformed points
    X_trainA=df[df["Outcome"]==0]
    X_trainB=df[df["Outcome"]==1]
    #calculating the prior probability of the classes
    prior_prob_A=X_trainA.shape[0]/(X_trainA.shape[0]+X_trainB.shape[0])
    prior_prob_B=X_trainB.shape[0]/(X_trainA.shape[0]+X_trainB.shape[0])
    #mean of the features of classA
    meanVarA1=np.mean(X_trainA.iloc[:,0],axis=0)
    meanVarA2=np.mean(X_trainA.iloc[:,1],axis=0)
    meanVarA3=np.mean(X_trainA.iloc[:,2],axis=0)
    #mean of the features of classB
    meanVarB1=np.mean(X_trainB.iloc[:,0],axis=0)
    meanVarB2=np.mean(X_trainB.iloc[:,1],axis=0)
    meanVarB3=np.mean(X_trainB.iloc[:,2],axis=0)
    #defining the meanVector which stacked all the means of the taken Features
    MeanVectorA=np.array([[meanVarA1,meanVarA2,meanVarA3]])
    MeanVectorB=np.array([[meanVarB1,meanVarB2,meanVarB3]])
    #calculating the covariance matrix for both classes(A and B)
    covA=np.cov(X_trainA.iloc[:,:3].T)
    covB=np.cov(X_trainB.iloc[:,:3].T)
    #Testing on the test dataset
    X_testA=X_test.iloc[:,:8]
    kk=accuracy(X_test,MeanVectorA,MeanVectorB,covA,covB,prior_prob_A,prior_prob_B)
    A.append(kk)
print("Average Accuracy of the model for 10 runs:")
print(np.mean(A))

