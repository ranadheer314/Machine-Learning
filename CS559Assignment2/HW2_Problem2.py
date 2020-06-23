#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

pima=pd.read_csv("diabetes.csv")

#storing the accuracy of 10 iterations in a list
ListAccuracy_k1=[]
ListAccuracy_k5=[]
ListAccuracy_k11=[]


#Method to calculating Euclidean Distance for two Vectors
def dist(x,y):
    temp=(x-y)**2
    s=np.sum(temp,axis=0)
    d=np.sqrt(s) 
    return d


#Method to Perform KNN 
def knn(x,k,x_old):
    #Making a copy of the Dataset to temparily store the distance values as a separate column
    x1=x_old.copy()
    #Temp List stores the distance of the numpy array "x" to all the sample from the given Dataset"x_old"
    temp=[]
    for i in range(x_old.shape[0]):
        d=dist(x,x_old.iloc[i,:].to_numpy().reshape(X_train.shape[1],1))
        temp.append(d)
    #Adding new column as the distance to the "x1" DataFrame
    x1["distance"]=temp
    #Sorts the DataFrame based upon the distance values 
    x2=x1.sort_values(by=['distance'])
    #sorted_k_indexes stores the K Nearest Neighbours to the test sample
    sorted_k_indexes=[]
    for j in range(k):
        sorted_k_indexes.append(x2.iloc[j,4])
    classA=0
    classB=0
    #calculating how many classA and ClassB outcomes were there in these K values
    for p in range(len(sorted_k_indexes)):
        if(sorted_k_indexes[p]==1):
            classB+=1
        else:
            classA+=1
    #if ClassB values were more than assign the test sample to ClassB (i.e outcome 1) else ClassA
    if(classA>classB):
        return 0
    else:
        return 1


for m in range(11):

    #Taking required features for training into a list
    features=["Glucose","BloodPressure","SkinThickness","Outcome"]
    #Creating a DataFrame with the list features
    X_temp=pima[features]
    #Spliting the Datasets into 2 parts i.e Training Set and Test Set
    X_train,X_test = train_test_split(X_temp, test_size=0.5)


    #Accuracy for K=1
    predicted_outcome=[]
    for k in range(X_test.shape[0]):
        #Changing the test sample from Pandas Series to numpy Array and reshaping it
        a=X_test.iloc[k,:].to_numpy().reshape(X_train.shape[1],1)
        #Calling the knn method and storing the return in a label
        label=knn(a,1,X_train)
        if(label==1):
            predicted_outcome.append(1)
        else:
            predicted_outcome.append(0)
    XTest=X_test.copy()
    #adding a column to the test set DataFrame with Predicted values
    XTest["predictedOutcome"]=predicted_outcome
    correct=0
    wrong=0
    for k in range(XTest.shape[0]):
        #If predicted values and actual outcome is same then increment the correct variable
        if(XTest.iloc[k,3]==XTest.iloc[k,4]):
            correct+=1
        else:
            wrong+=1
    print("Total no of correctly predicted values:" +str(correct))
    Accuracy_k1=correct/X_test.shape[0]
    print("Accuracy: "+str(Accuracy_k1))


    #Accuracy for K=5
    predicted_outcome=[]
    for k in range(X_test.shape[0]):
        a=X_test.iloc[k,:].to_numpy().reshape(X_train.shape[1],1)
        label=knn(a,5,X_train)
        if(label==1):
            predicted_outcome.append(1)
        else:
            predicted_outcome.append(0)
    XTest=X_test.copy()
    XTest["predictedOutcome"]=predicted_outcome
    correct=0
    wrong=0
    for k in range(XTest.shape[0]):
        if(XTest.iloc[k,3]==XTest.iloc[k,4]):
            correct+=1
        else:
            wrong+=1
    print("Total no of correctly predicted values:" +str(correct))
    Accuracy_k5=correct/X_test.shape[0]
    print("Accuracy: "+str(Accuracy_k5))


    #Accuracy for K=11
    predicted_outcome=[]
    for k in range(X_test.shape[0]):
        a=X_test.iloc[k,:].to_numpy().reshape(X_train.shape[1],1)
        label=knn(a,11,X_train)
        if(label==1):
            predicted_outcome.append(1)
        else:
            predicted_outcome.append(0)
    XTest=X_test.copy()
    XTest["predictedOutcome"]=predicted_outcome
    correct=0
    wrong=0
    for k in range(XTest.shape[0]):
        if(XTest.iloc[k,3]==XTest.iloc[k,4]):
            correct+=1
        else:
            wrong+=1
    print("Total no of correctly predicted values:" +str(correct))
    Accuracy_k11=correct/X_test.shape[0]
    print("Accuracy: "+str(Accuracy_k11))


    ListAccuracy_k1.append(Accuracy_k1)
    ListAccuracy_k5.append(Accuracy_k5)
    ListAccuracy_k11.append(Accuracy_k11)


#mean accuracy
Mean_k1=sum(ListAccuracy_k1)/len(ListAccuracy_k1)
Mean_k5=sum(ListAccuracy_k5)/len(ListAccuracy_k5)
Mean_k11=sum(ListAccuracy_k11)/len(ListAccuracy_k11)

print("Mean for K==1: "+str(Mean_k1))
print("Mean for K==5: "+str(Mean_k5))
print("Mean for K==11: "+str(Mean_k11))



std_dev=0
for p in range(len(ListAccuracy_k1)):
    std_dev+=(ListAccuracy_k1[p]-Mean_k1)**2
std_d=std_dev/(len(ListAccuracy_k1)-1)
std_deviation_k1=math.sqrt(std_d)
#Standard Deviation of the accuracy
print(std_deviation_k1)


std_dev=0
for p in range(len(ListAccuracy_k5)):
    std_dev+=(ListAccuracy_k5[p]-Mean_k5)**2
std_d=std_dev/(len(ListAccuracy_k5)-1)
std_deviation_k5=math.sqrt(std_d)
#Standard Deviation of the accuracy
print(std_deviation_k5)


std_dev=0
for p in range(len(ListAccuracy_k11)):
    std_dev+=(ListAccuracy_k11[p]-Mean_k11)**2
std_d=std_dev/(len(ListAccuracy_k11)-1)
std_deviation_k11=math.sqrt(std_d)
#Standard Deviation of the accuracy
print(std_deviation_k11)

