#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\1.SVR\Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Applying the SVR from SVM
from sklearn.svm import SVR
regressor = SVR(kernel="poly", degree=4, gamma="auto")
regressor.fit(x,y)
y_pred = regressor.predict([[6.5]])


#Visualising the SVR results
plt.scatter(x, y, color = "red")
plt.plot(x, regressor.predict(x), color = "blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
