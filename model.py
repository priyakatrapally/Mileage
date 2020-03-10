# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('D:\\CQ\\Linear Regression\\KPL.csv')

#Since it is a small dataset, we will train our model with all availabe data.
x=dataset.iloc[:,1:4]
y=dataset.iloc[:,0]

from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()

#Fitting model with trainig data
reg_model.fit(x, y)
# Saving model to disk
pickle.dump(reg_model, open('reg_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('reg_model.pkl','rb'))
print(model.predict([[90,90 ,100 ]]))
