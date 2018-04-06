import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def np_sig(x): return 1/(1 + np.exp(-1 * x))
data = np_sig(np.array([-62,-7.400,110.5,9790.5,0.4,1]))
print(len(data))
print('Does it belong to class Z : ',data>0.5)
print('----------------------')


x = np.reshape([1,2,3,4,5],(-1,1))
con = np.reshape([10,20,33,40,54],(-1,1))
regr = linear_model.LinearRegression()
regr.fit(x,con)
print('Predict on 6 :',regr.predict(6))
print('----------------------')

out_data = np.array([1,3,999,-786,2,49,2])
std = out_data.std()
mean = out_data.mean()
for x in range(len(out_data)):
    current_data = out_data[x]
    if not current_data < mean + std or not current_data > mean - std:
        print('Outliers : ',current_data)
print('----------------------')


# --- end code --