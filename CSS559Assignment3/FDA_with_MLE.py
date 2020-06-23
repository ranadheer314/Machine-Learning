#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# In[311]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[312]:


#Importing the data
pima=pd.read_csv("diabetes.csv")
X_train,X_test=train_test_split(pima, test_size=0.5)


# In[313]:


class1=X_train[X_train["Outcome"]==0]
class2=X_train[X_train["Outcome"]==1]


# In[314]:


def mean(X):
    X_mean=np.mean(X,axis=0)
    return X_mean


# In[315]:


MeanC1=mean(class1.iloc[:,:8])
MeanC2=mean(class2.iloc[:,:8])
tMean=mean(X_train.iloc[:,:8])


# In[316]:


MeanC1=MeanC1.to_numpy().reshape(MeanC1.shape[0],1)
MeanC2=MeanC2.to_numpy().reshape(MeanC1.shape[0],1)
tMean=tMean.to_numpy().reshape(tMean.shape[0],1)


# In[317]:


S1=(class1.shape[0]-1)*np.dot((class1.iloc[:,:8]-MeanC1.T).T,(class1.iloc[:,:8]-MeanC1.T))
S2=(class2.shape[0]-1)*np.dot((class2.iloc[:,:8]-MeanC2.T).T,(class2.iloc[:,:8]-MeanC2.T))
Sw=S1+S2


# In[318]:


Sw


# In[319]:


tMean.shape


# In[320]:


Sb1=(class1.shape[0]-1)*np.dot((tMean-MeanC1.T),(tMean-MeanC1.T))
Sb2=(class2.shape[0]-1)*np.dot((tMean-MeanC2.T),(tMean-MeanC2.T))
Sb=Sb1+Sb2


# In[321]:


inv_Sw=np.linalg.inv(Sw)


# In[322]:


MeanC1.shape


# In[323]:


optimal_direction=np.dot(inv_Sw,MeanC1-MeanC2)


# In[324]:


C1_LDA=class1.iloc[:,:8]*optimal_direction.T
C2_LDA=class2.iloc[:,:8]*optimal_direction.T


# In[325]:


X_trainA=C1_LDA
X_trainB=C2_LDA


# In[326]:


#calculating the prior probability of the classes
prior_prob_A=X_trainA.shape[0]/(X_trainA.shape[0]+X_trainB.shape[0])
prior_prob_B=X_trainB.shape[0]/(X_trainA.shape[0]+X_trainB.shape[0])


# In[327]:


#function which takes "mean" and "covariance" as the parameters and returns the likelihood of the Feature Vector
def likelihood(x,mu,co):
    #inverse of the covariance matrix
    inv=np.linalg.inv(co)
    p1=1/(np.sqrt(((2*np.pi)**3)*np.linalg.det(co)))
    p2=np.exp(-0.5*np.dot(np.dot((x-mu).T,inv),(x-mu)))
    p=p1*p2
    return p


# In[328]:


#calculating the covariance matrix for both classes(A and B)
covA=np.cov(X_trainA.iloc[:,:].T)
covB=np.cov(X_trainB.iloc[:,:].T)


# In[329]:


MeanVectorA=np.mean(X_trainA,axis=0)
MeanVectorB=np.mean(X_trainB,axis=0)


# In[330]:


MeanVectorA=MeanVectorA.to_numpy().reshape(1,MeanVectorA.shape[0])
MeanVectorB=MeanVectorB.to_numpy().reshape(1,MeanVectorB.shape[0])


# In[331]:


X_test


# In[332]:


a.iloc[:,]


# In[333]:


postA=likelihood(k,MeanVectorA.T,covA)*prior_prob_A
postB=likelihood(k,MeanVectorB.T,covB)*prior_prob_B


# In[334]:


#Testing on the test dataset
X_testA=X_test.iloc[:,:8]
def accuracy(X_test,Optimal_direction,MeanVectorA,meanVectorB,covA,covB,prior_prob_A,prior_prob_B):
    a=X_test.iloc[:,:8]*Optimal_direction.T
    count=0
    for i in range(X_test.shape[0]):
        k=a.iloc[i,:].to_numpy().reshape(a.shape[1],1)
        #a=X_trainA.iloc[25,:3].to_numpy().reshape(X_trainA.shape[1]-1,1)
        #a=Xtest_pca[i,:].reshape(Xtest_pca.shape[1],1)
        postA=likelihood(k,MeanVectorA.T,covA)*prior_prob_A
        postB=likelihood(k,MeanVectorB.T,covB)*prior_prob_B
        if(postA<postB):
            if(X_test.iloc[i,8]==1):
                count+=1
        else:
            if(X_test.iloc[i,8]==0):
                count+=1
    accurate=count/X_test.shape[0]
    return accurate


# In[335]:


accuracy(X_test,optimal_direction,MeanVectorA,MeanVectorB,covA,covB,prior_prob_A,prior_prob_B)

