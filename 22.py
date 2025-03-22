import numpy as np
import pandas as pd
import streamlit as st
import pickle #pickle make a 60line code in to 1 file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ds=pd.read_csv(r'c:\Users\ADMIN\Downloads\Salary_Data.csv')
print(ds)

print(ds.shape)

x=ds.iloc[:,:-1]
y=ds.iloc[:,-1]
print(x,y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=0)

y_test=y_test.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

#future prediction
y_12=regressor.predict([[12]])
y_30=regressor.predict([[30]])
print("predicted salary for 12 yrs of exp: ${y_12[0]:,.2f}")
print("predicted salary for 30 yrs exp: ${y_30[0]:,.2f}")

#check model perfomance
bias=regressor.score(x_train,y_train)
variance=regressor.score(x_test,y_test)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())
