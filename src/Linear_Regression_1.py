import numpy as np
import matplotlib
import pandas
import sklearn
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
#load Dataset
path_file = "/Users/priyambadajain/Desktop/Surgical_Cases_fm_3.3.csv"
dataset = pandas.read_csv(path_file)


dataset1=dataset.drop(dataset.columns[[0,2,4,6,7,11,12,13,14,15,17,18]],axis=1)
dataset1=dataset1[(dataset1['SN - BM - PRBC Ordered']>=0)]
dataset1=dataset1[(dataset1['age']>=0)]
dataset1=dataset1[(dataset1['SN - BM - Pre-Op INR']>=0)]
dataset1=dataset1[(dataset1['SN - BM - Pre-Op Platelet Count']>=0)]
dataset1[dataset1.columns[3]] = pandas.Series(dataset1[dataset1.columns[3]]).convert_objects(convert_numeric=True)
dataset1[dataset1.columns[4]] = pandas.Series(dataset1[dataset1.columns[4]]).convert_objects(convert_numeric=True)
dataset1[dataset1.columns[2]] = dataset1[dataset1.columns[2]].astype('category')
dataset1[dataset1.columns[1]] = dataset1[dataset1.columns[1]].astype('category')
cat_columns = dataset1.select_dtypes(['category']).columns
dataset1[cat_columns] = dataset1[cat_columns].apply(lambda x: x.cat.codes)
dataset1 = dataset1.fillna(0)

target = dataset1['SN - BM - PRBC Ordered']
dataset1=dataset1.drop(dataset1.columns[[5]],axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(dataset1)
df_normalized = pd.DataFrame(np_scaled)
target = df_normalized[df_normalized.columns[[5]]]
df_normalized=df_normalized.drop(df_normalized.columns[[5]],axis=1)
X_train = dataset1[:-600]
X_test = dataset1[-600:]

y_train = target[:-600]
y_test = target[-600:]

model_lm = linear_model.LinearRegression()

model_lm.fit(X_train,y_train)
prediction = model_lm.predict(X_test)



from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, prediction)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, prediction)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,prediction)
from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, prediction)
from sklearn.metrics import r2_score
r2_score(y_test,prediction)


