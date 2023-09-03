#!/usr/bin/env python
# coding: utf-8

# In[356]:


#Importing python packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
#Creating Lists for storing data
TrainTime = []
TestTime = []
Accuracy_List = []
Algorithems = ['SVM','KNN','Decision Tree','Neural Network','Neural Network Tuned']


# In[357]:


#Reading CSV
dataset = pd.read_csv('heart.csv')


# In[358]:


dataset.head()


# In[359]:


#Checking Information of CSV
dataset.info()


# In[360]:


#Checking Unique Values in data
dataset.nunique()


# In[361]:


#Checking Classification on Sex
sns.barplot(data = dataset, x = 'Sex' , y = 'HeartDisease')


# In[362]:


#Checking Classification on ChestPainType
sns.barplot(data = dataset, x = 'ChestPainType' , y = 'HeartDisease')


# In[363]:


#Checking Classification on ST_Slope
sns.barplot(data = dataset, x = 'ST_Slope' , y = 'HeartDisease')


# In[364]:


#Checking Classification on RestingECG
sns.barplot(data = dataset, x = 'RestingECG' , y = 'HeartDisease')


# In[365]:


#Checking Classification on ExerciseAngina
sns.barplot(data = dataset, x = 'ExerciseAngina' , y = 'HeartDisease')


# In[366]:


#Checking Classification on FastingBS
sns.barplot(data = dataset, x = 'FastingBS' , y = 'HeartDisease')


# In[367]:


#Checking Classification on Age
sns.lineplot(data = dataset, x = 'Age' , y = 'HeartDisease')


# In[368]:


#Replacing all string data with numerical Data
dataset['Sex'].replace({'M':1,'F':0}, inplace = True)
dataset['ChestPainType'].replace({'TA':0,'ASY':1,'ATA':2,'NAP':3}, inplace = True)
dataset['ST_Slope'].replace({'Down':0,'Flat':1,'Up':2}, inplace = True)
dataset['RestingECG'].replace({'Normal':0,'LVH':1,'ST':2}, inplace = True)
dataset['ExerciseAngina'].replace({'N':1,'Y':0}, inplace = True)


# In[369]:


dataset.head()


# In[370]:


#Seperating Labels from Data
x = dataset.drop('HeartDisease', axis=1)
y = dataset['HeartDisease']


# In[371]:


#Spliting Data in train and test
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=0)


# In[372]:


#SVM Training and Testing and Execution time Calculation
Model = svm.SVC(kernel = 'linear')
start = time.time()
Model.fit(train_x,train_y)
end = time.time()
time_ = end - start
TrainTime.append(time_)
start = time.time()
predict_y = Model.predict(test_x)
end = time.time()
time_ = end - start
TestTime.append(time_)


# In[373]:


# SVM Accuracy Calculation
Accuracy = Model.score(test_x,test_y)*100
print('Accuracy: ',"{:.2f}".format(Accuracy),'%')
Accuracy_List.append(Accuracy)


# In[374]:


# SVM Confusion matrix
confusion_matrix(test_y,predict_y)


# In[375]:


# KNN checking for best Accuracy
acc = []
for k in range(1,30):
    Model = KNeighborsClassifier(k)
    Model.fit(train_x,train_y)
    predict_y = Model.predict(test_x)
    acc.append(Model.score(test_x,test_y)*100)
plt.figure(figsize=(12,6))
plt.plot(range(1,30),acc)
plt.title('Accuracy Rate K Values')
plt.xlabel('K-Values')
plt.ylabel('Mean Accuracy')
plt.show()
print("Maximum Accuracy: ",max(acc),"at K =",acc.index(max(acc))+1)


# In[376]:


#KNN Training and Testing and Execution Time Calculation 
Model = KNeighborsClassifier(n_neighbors=11)  
start = time.time()
Model.fit(train_x, train_y)
end = time.time()
time_ = end - start
TrainTime.append(time_)
start = time.time()
predict_y = Model.predict(test_x)
end = time.time()
time_ = end - start
TestTime.append(time_)


# In[377]:


# KNN Accuracy
Accuracy = Model.score(test_x,test_y)*100
print('Accuracy: ',"{:.2f}".format(Accuracy),'%')
Accuracy_List.append(Accuracy)


# In[378]:


# KNN Confuxion Matrix
confusion_matrix(test_y, predict_y)


# In[379]:


#Decision Tree Training and Testing Execution Time Calculation
Model = DecisionTreeClassifier(max_depth=3)
start = time.time()
Model.fit(train_x, train_y)
end = time.time()
time_ = end - start
TrainTime.append(time_)
start = time.time()
predict_y = Model.predict(test_x)
end = time.time()
time_ = end - start
TestTime.append(time_)


# In[380]:


# Decision Tree Accuracy
Accuracy = Model.score(test_x, test_y)*100
print('Accuracy: ',"{:.2f}".format(Accuracy),'%')
Accuracy_List.append(Accuracy)


# In[381]:


confusion_matrix(test_y, predict_y)


# In[382]:


Model = MLPClassifier(random_state=1)
start = time.time()
Model.fit(train_x, train_y)
end = time.time()
time_ = end - start
TrainTime.append(time_)
start = time.time()
predict_y = Model.predict(test_x)
end = time.time()
time_ = end - start
TestTime.append(time_)


# In[383]:


Accuracy = tree.score(test_x, test_y)*100
print('Accuracy: ',"{:.2f}".format(Accuracy),'%')
Accuracy_List.append(Accuracy)


# In[384]:


confusion_matrix(test_y, predict_y)


# In[385]:


Model = MLPClassifier(activation='tanh',solver='sgd',alpha=0.05,hidden_layer_sizes=(4000,10),random_state=1)
start = time.time()
Model.fit(train_x, train_y)
end = time.time()
time_ = end - start
TrainTime.append(time_)
start = time.time()
predict_y = Model.predict(test_x)
end = time.time()
time_ = end - start
TestTime.append(time_)


# In[386]:


Accuracy = tree.score(test_x, test_y)*100
print('Accuracy: ',"{:.2f}".format(Accuracy),'%')
Accuracy_List.append(Accuracy)


# In[387]:


confusion_matrix(test_y, predict_y)


# In[388]:


c = ["red", "green", "orange", "black", "yellow"]
plt.figure(figsize=(12,6))
plt.bar(Algorithems,Accuracy_List, color=c)
plt.xlabel("Classifications")
plt.ylabel("Percentage")
plt.show()


# In[389]:


c = ["red", "green", "orange", "black", "yellow"]
plt.figure(figsize=(12,6))
plt.bar(Algorithems,TrainTime, color=c)
plt.xlabel("Classifications")
plt.ylabel("Time (s)")
plt.show()


# In[390]:


c = ["red", "green", "orange", "black", "yellow"]
plt.figure(figsize=(12,6))
plt.bar(Algorithems,TestTime, color=c)
plt.xlabel("Classifications")
plt.ylabel("Time (s)")
plt.show()


# In[391]:


result = pd.DataFrame({'Titles': Algorithems,
             'Accuracy (%)':Accuracy_List,
             'Train Time (s)': TrainTime,
             'Test Time (s)':TestTime})


# In[392]:


result

