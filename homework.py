import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
 
from sklearn import metrics 

data = pd.read_csv("titanic.csv", usecols=["Survived", "Pclass", "Name", "Sex", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare"])

print(data.head())
print(data.info())


data["Sex"] = data["Sex"].replace({"male": 0, "female": 1})
data = data.dropna(axis=0)
print(data.head())



X = data.drop(["Survived", "Name"], axis=1)
y = data["Survived"]


mp.scatter(data["Age"],data["Survived"], s = 10, c = "green", marker = "o")
mp.show()

mp.scatter(data["Fare"], data["Survived"], s=10, c = "gray", marker="*")
mp.show()

mp.scatter(data["Pclass"], data["Survived"], s=10, marker="v")
mp.show()

mp.scatter(data["Sex"], data["Survived"], s=20, marker="^")
mp.show()  

mp.scatter(data["Survived"], data["Survived"], s=20, marker="^")
mp.show()  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape) 


model = DecisionTreeClassifier(max_depth=3,random_state=1)
 
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy ", metrics.accuracy_score(predictions, y_test))
print("or")
a = metrics.accuracy_score(predictions, y_test)
print("Accuracy: ", a*100,"%")
