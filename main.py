import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
#machine learning algorithims 
#splitting the data into training and testing
from sklearn.model_selection import train_test_split

#Algorithim that we would be using 
from sklearn.tree import DecisionTreeClassifier
#for finding accuracy 
from sklearn import metrics 

data = pd.read_csv("iris.csv")

print(data.head())#displays the first 5 rows 
print(data.info())# summarises the dataset

#data preprocessing

data["species"] = data["species"].replace({"setosa":0,"versicolor":1,"virginica":2})
print(data.head())

#input and output

X = data.drop("species", axis = 1)
y = data["species"]

mp.scatter(data["petal_length"],data["species"], s = 10, c = "green", marker = "o")
mp.show()

