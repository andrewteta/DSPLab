import numpy as np
from sklearn import svm
from tkinter import filedialog
import input_titanic_data as titIn

def predict(data, labels, svc):
    svc.fit(data, labels)
    p_val = svc.predict(data)
    p_success = (1 - p_val[labels != p_val].size / labels.size) * 100
    return p_val, p_success

#filepath = filedialog.askopenfilenames()[0]

# create SVM module
clf = svm.SVC(gamma='scale', kernel='linear')
# extract all titanic training data into data matrix and labels vector
titanic_data, survived = titIn.get_titanic_all('titanic_tsmod.csv')

# extract fare data column and run predictions
fareData = np.reshape(titanic_data[:,-4], (-1,1))
pred_val, percent_success = predict(fareData, survived, clf)
print(f'percent correct (fare) = {percent_success}')

# extract class data columns and run predictions
classData = titanic_data[:,0:3]
pred_val, percent_success = predict(classData, survived, clf)
print(f'percent correct (class) = {percent_success}')

# extract sex data column and run predictions
sexData = np.reshape(titanic_data[:,3], (-1,1))
pred_val, percent_success = predict(sexData, survived, clf)
print(f'percent correct (sex) = {percent_success}')

# extract age data columns and run predictions
ageData = titanic_data[:,4:85]
pred_val, percent_success = predict(ageData, survived, clf)
print(f'percent correct (age) = {percent_success}')

# extract siblings/parents data columns and run predictions
sibspData = titanic_data[:,85:94]
pred_val, percent_success = predict(sibspData, survived, clf)
print(f'percent correct (sibsp) = {percent_success}')

# extract parents/children data columns and run predictions
parchData = titanic_data[:,94:104]
pred_val, percent_success = predict(parchData, survived, clf)
print(f'percent correct (parch) = {percent_success}')

# extract parents/children data columns and run predictions
embarkedData = titanic_data[:,105:108]
pred_val, percent_success = predict(embarkedData, survived, clf)
print(f'percent correct (embarked) = {percent_success}')

# extract highest prediction value data columns and run predictions
combData = np.hstack((ageData, classData, sexData))
pred_val, percent_success = predict(combData, survived, clf)
print(f'percent correct (age, sex, class) = {percent_success}')

print('done')
