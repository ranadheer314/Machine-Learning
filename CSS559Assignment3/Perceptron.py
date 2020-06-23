#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ![image.png](attachment:image.png)

# In[116]:


#store all the arrays in the numpy array
X_init=np.array([[2,1,1,-1,0,2],[1,0,0,1,2,0],[2,-1,-1,1,1,0],[1,4,0,1,2,1],[1,-1,1,1,1,0],[1,-1,-1,-1,1,0],[2,-1,1,1,2,1]])


# In[117]:


#creating a new column with all ones
new_column=np.ones((X_init.shape[0],1))


# In[118]:


X_in=X_init[:,0:1]


# In[119]:


# Last Row of the Numpy array is the extra column with ones 
X_shift=np.append(X_in,new_column,axis=1)


# In[120]:


for i in range(1,X_init.shape[1]):
    X_col=X_init[:,i].reshape(X_init.shape[0],1)
    X_shift=np.append(X_shift,X_col,axis=1)


# In[121]:


X_init


# In[122]:


#added a one colummn at the initial position
X_shift


# In[123]:


# Multiplied by -1 for rows which belongs to class 2
for i in range(X_shift.shape[0]):
    if(X_shift[i,0]==2):
        for j in range(1,X_shift.shape[1]):
            X_shift[i,j]*=-1
    


# In[124]:


X_shift


# In[125]:


X=X_shift[:,1:]


# In[126]:


#final Matrix after all necessary operations
#added one one's column at the front 
# Multiplied by -1 for rows which belongs to class 2
print(X)


# In[127]:


initial_weight=np.array([[3,1,1,-1,2,-7]])


# In[128]:


initial_weight


# In[129]:


Learning_rate=1


# In[130]:


iterations=1000


# In[131]:


np.dot(X,initial_weight.T)


# In[132]:


weights=initial_weight
br=1
while(br):
    #if all the samples are correctly classified break the loop
    if(sum(np.dot(X,weights.T)>0)):
        br=0
    for j in range(X.shape[0]):
        if(np.dot(weights,X[j,:].T)<0):
             #update the weights 
            weights=weights+(Learning_rate*X[j,:])
           

print("final weight matrix:")
print(weights)


# In[133]:


print("performing X*W.T on final weightss")
print(np.dot(X,weights.T))


# In[ ]:




