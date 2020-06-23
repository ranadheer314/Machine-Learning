#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ![image.png](attachment:image.png)

# In[105]:


def mean(X):
    X_mean=np.mean(X,axis=0)
    return X_mean


# In[106]:


C1=np.array([[-2,1],[-5,-4],[-3,1],[0,-3],[-8,-1]])
C2=np.array([[2,5],[1,0],[5,-1],[-1,-3],[6,1]])


# In[107]:


M1=np.mean(C1,axis=0)
M2=np.mean(C2,axis=0)


# In[108]:


S1=(C1.shape[0]-1)*np.cov(C1.T)
S2=(C2.shape[0]-1)*np.cov(C2.T)


# In[109]:


S1


# In[110]:


S_W=S1+S2


# In[111]:


np.linalg.inv(S_W)


# In[112]:


Optimal_components=np.dot(np.linalg.inv(S_W),M1-M2)


# In[113]:


print("Optimal Line Direction: ")
print(Optimal_components)


# In[114]:


test_class1=np.dot(class1,Optimal_components)
test_class2=np.dot(class2,Optimal_components)


# In[118]:


test_class2


# In[115]:


print("correctly clasified points :")
for i in range(class1.shape[0]):
    if(test_class1[i]>0):
        print(class1[i])
for i in range(class2.shape[0]):
    if(test_class2[i]<0):
        print(class2[i])


# In[116]:


print("incorrectly clasified points :")
for i in range(class1.shape[0]):
    if(test_class1[i]<0):
        print(class1[i])
for i in range(class2.shape[0]):
    if(test_class2[i]>0):
        print(class2[i])

