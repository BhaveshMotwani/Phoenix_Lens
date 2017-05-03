import pandas
import numpy
import math
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import  cross_val_score
import sklearn.ensemble
import sys
csvFile = "/home/sovan/Desktop/Surgical_Cases_3.3.csv"
surge_data = pandas.read_csv(csvFile)


def ageCleaner(x):
    if isinstance(x,str) or x == None or math.isnan(x):
        return 60.0
    else:
        return x

def timeToMinutes(x):
    myArr = x.strip().split(':')
    if len(myArr) < 2:
        return 0
    else:
        return int(myArr[0])*60 + int(myArr[1])

surge_data_reqd = surge_data
surge_data_reqd['Age'] = surge_data_reqd['Age'].map(ageCleaner)


surge_data_reqd[surge_data_reqd.columns[4]] = surge_data_reqd[surge_data_reqd.columns[4]].map(timeToMinutes)

surge_procedureNames = surge_data_reqd[surge_data_reqd.columns[3]].tolist()

surge_data_reqd[surge_data_reqd.columns[3]] = surge_data_reqd[surge_data_reqd.columns[3]].astype('category')
cat_columns = surge_data_reqd.select_dtypes(['category']).columns
surge_data_reqd[cat_columns] = surge_data_reqd[cat_columns].apply(lambda x: x.cat.codes)

surge_process_numbers = surge_data_reqd[surge_data_reqd.columns[3]].tolist()

#print(list(surge_data_reqd[surge_data_reqd.columns[3]].index))



surgeProcessDict = {}

for x in range(len(surge_procedureNames)):
	surge_procedureNames[x] = str(surge_procedureNames[x])  
	p = surge_procedureNames[x].strip().lower()
	surgeProcessDict[p] = surge_process_numbers[x]




surge_specialityName = surge_data_reqd[surge_data_reqd.columns[5]].tolist()

surge_data_reqd[surge_data_reqd.columns[5]] = surge_data_reqd[surge_data_reqd.columns[5]].astype('category')
cat_columns = surge_data_reqd.select_dtypes(['category']).columns
surge_data_reqd[cat_columns] = surge_data_reqd[cat_columns].apply(lambda x: x.cat.codes)

surge_speciality_numbers = surge_data_reqd[surge_data_reqd.columns[5]].tolist()

#print(list(surge_data_reqd[surge_data_reqd.columns[3]].index))



surgeSpeciality = {}

for x in range(len(surge_specialityName)):
        surge_specialityName[x] = str(surge_specialityName[x])
        p = surge_specialityName[x].strip().lower()
        surgeSpeciality[p] = surge_speciality_numbers[x]


patientTypeName = surge_data_reqd[surge_data_reqd.columns[7]].tolist()

surge_data_reqd[surge_data_reqd.columns[7]] = surge_data_reqd[surge_data_reqd.columns[7]].astype('category')
cat_columns = surge_data_reqd.select_dtypes(['category']).columns
surge_data_reqd[cat_columns] = surge_data_reqd[cat_columns].apply(lambda x: x.cat.codes)

ptn = surge_data_reqd[surge_data_reqd.columns[7]].tolist()

#print(list(surge_data_reqd[surge_data_reqd.columns[3]].index))



pt_dict = {}

for x in range(len(patientTypeName)):
        patientTypeName[x] = str(patientTypeName[x])
        p = patientTypeName[x].strip().lower()
        pt_dict[p] = ptn[x]





def myINRFilter(x):
    if math.isnan(x) or x == 0.0:
        x = random.uniform(0.8,1.1)
    return x


surge_data_reqd[surge_data_reqd.columns[8]] = pandas.Series(surge_data_reqd[surge_data_reqd.columns[8]]).convert_objects(convert_numeric=True)
surge_data_reqd[surge_data_reqd.columns[8]] = surge_data_reqd[surge_data_reqd.columns[8]].map(myINRFilter)



surge_data_reqd[surge_data_reqd.columns[9]] = pandas.Series(surge_data_reqd[surge_data_reqd.columns[9]]).convert_objects(convert_numeric=True)

def myPlateletsFilter(x):
    if math.isnan(x) or x <= 34.009:
        x = 34.009
    elif x >= 524.194:
        x = 524.194
    return x

surge_data_reqd[surge_data_reqd.columns[9]] = surge_data_reqd[surge_data_reqd.columns[9]].map(myPlateletsFilter)

def RBC_Cleaner(x):
    if isinstance(x,str) or x == None or math.isnan(x):
        return 0.0
    else:
        return x


surge_data_reqd[surge_data_reqd.columns[10]] = pandas.Series(surge_data_reqd[surge_data_reqd.columns[10]]).convert_objects(convert_numeric=True)
surge_data_reqd[surge_data_reqd.columns[10]] = surge_data_reqd[surge_data_reqd.columns[10]].map(RBC_Cleaner)





def ResultsBeforeSurgery(x):
    if isinstance(x,str) or x == None or math.isnan(x):
        return 0.0
    else:
        return x

surge_data_reqd[surge_data_reqd.columns[16]] = pandas.Series(surge_data_reqd[surge_data_reqd.columns[16]]).convert_objects(convert_numeric=True)
surge_data_reqd[surge_data_reqd.columns[16]] = surge_data_reqd[surge_data_reqd.columns[16]].map(ResultsBeforeSurgery)


print(surge_data_reqd.head(5))

'''
cleaned_data = surge_data_reqd.drop(surge_data.columns[[0,2,4,6,11,12,13,14,15,17,18]],axis=1)




mean = cleaned_data.mean(axis=0)

std =  cleaned_data.std(axis=0)



cols_to_norm = ['SN - BM - Pre-Op INR','SN - BM - Pre-Op Platelet Count','Age','SURG_PROCEDURE','SURGICAL_SPECIALTY','PATIENT_TYPE','ResultsBeforeSurgery']
cleaned_data[cols_to_norm] = cleaned_data[cols_to_norm].apply(lambda x: (x - x.mean()) / x.std())


cleaned_data = cleaned_data[cleaned_data.apply(lambda x : numpy.abs(x -x.mean()) <= (3.0 * x.std())).all(axis=1)]


train, validate = numpy.split(cleaned_data.sample(frac=1), [int(.6*len(cleaned_data))])





X_train = train.drop(train.columns[[6]],axis=1)

Y_train = train.drop(train.columns[[0,1,2,3,4,5,7]],axis=1)

regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=150)

regressor.fit(X_train, Y_train)

X_test = validate.drop(train.columns[[6]],axis=1)

Y_test = validate.drop(train.columns[[0,1,2,3,4,5,7]],axis=1)

Y_predict = regressor.predict(X_test)


print(sklearn.metrics.mean_absolute_error(Y_test, Y_predict))


	
def mainModel(age,sp,ss,pt,inr,plt,hemo):
	age = int(age)
	age = float((float(age) - mean['Age'])/std['Age'])
	sp = sp.strip().lower()
	sp = float((int(surgeProcessDict[sp]) - mean['SURG_PROCEDURE'])/std['SURG_PROCEDURE'])
	ss = ss.strip().lower()
	ss = float((int(surgeSpeciality[ss]) - mean['SURGICAL_SPECIALTY'])/std['SURGICAL_SPECIALTY'])
	pt = pt.strip().lower()
	pt = float((int(pt_dict[pt]) - mean['PATIENT_TYPE'])/std['PATIENT_TYPE'])
	inr = float(inr)
	inr = float((float(age) - mean['SN - BM - Pre-Op INR'])/std['SN - BM - Pre-Op INR'])
	plt = float(plt)
	plt = float((float(age) - mean['SN - BM - Pre-Op Platelet Count'])/std['SN - BM - Pre-Op Platelet Count'])
	hemo = float(hemo)
	hemo = float((float(hemo) - mean['ResultsBeforeSurgery'])/std['ResultsBeforeSurgery'])
	to_predict = X_test[0:0]
	to_predict = to_predict.append({'Age':age,'SURG_PROCEDURE':sp,'SURGICAL_SPECIALTY':ss,'PATIENT_TYPE':pt,'SN - BM - Pre-Op INR':inr,'SN - BM - Pre-Op Platelet Count':plt,'ResultsBeforeSurgery':hemo}, ignore_index=True)
	Y_predict = regressor.predict(to_predict)
	return Y_predict[0]
'''
