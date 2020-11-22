import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
import input_titanic_data as titIn
import csv
import matplotlib.pyplot as plt
from scipy.io import arff
import pickle
from sklearn.neural_network import MLPClassifier

def predict(data, labels, svc):
    # fit SVM object to data set with labeled outcomes
    svc.fit(data, labels)
    # run predictions on training set
    p_val = svc.predict(data)
    # calculate percentage of success of predicted outcomes
    p_success = (1 - p_val[labels != p_val].size / labels.size) * 100
    return p_val, p_success

def linRegPredict(data, labels):
    # fit linear regression model to data set with labeled outcomes
    linReg = LinearRegression().fit(data, labels)
    # run predictions on training set
    p_vals = linReg.predict(data)
    # compute errors
    mu = np.mean(labels)
    tss = np.sum((labels - mu)**2)
    rss = np.sum((labels - p_vals)**2)
    ess = np.sum((p_vals - mu)**2)
    r_squared = 1 - (rss / tss)
    return p_vals, r_squared

def importYacht(filename):
    fd_r = open(filename, 'r')
    datareader = csv.reader(fd_r, dialect='excel')
    X = np.zeros( (0,6), np.float)
    y = np.zeros(0, np.float)

    # Read the labels from the file
    a = next(datareader)

    # extract data from csv into feature matrix and prediction vector
    for ctr, line in enumerate(datareader):
        # --------- X --------- #
        # col_1 = longitudinal position center of buoyancy
        # col_2 = prismatic coefficient
        # col_3 = length-displacement ratio
        # col_4 = beam-draught ratio
        # col_5 = length-beam ratio
        # col_6 = Froude number

        # --------- y --------- #
        # residuary resistance per unit weight of displacement
        X = np.vstack( [X, np.array(line[0:-1], dtype=np.float64)] )
        y = np.hstack( [y, np.array(line[-1:], dtype=np.float64)] )

    return X, y

def loadNNData():
    # load data
    print('loading NN data...')
    X = pickle.load(open('mnist_X_uint8.sav', 'rb'))
    y = pickle.load(open('mnist_y_uint8.sav', 'rb'))
    print('done loading NN data')
    X = X/255
    return X, y

def sortNNData(data, labels):
    X_training = np.zeros((10, 6000, 784))
    X_test = np.zeros((10, 1000, 784))
    y_training = np.zeros((10, 6000, 1))
    y_test = np.zeros((10, 1000, 1))

    # loop over digits
    print('sorting NNData...')
    for digit in range(10):
        imageIndex = 0
        for label in range(len(labels)):
            # find all elements equal to digit (in training set)
            if labels[label] == digit and imageIndex < 6000:
                X_training[digit, imageIndex, :] = data[label]
                y_training[digit, imageIndex] = digit
                imageIndex += 1
            # find all elements equal to digit (in test set)
            elif labels[label] == digit and (imageIndex >= 6000 and imageIndex < 7000):
                X_test[digit, imageIndex - 6000, :] = data[label]
                y_test[digit, imageIndex - 6000] = digit
                imageIndex += 1
    print('done sorting NNData')
    return X_training, X_test, y_training, y_test

# extract raw data for NN processing
NNData = loadNNData()
NNData_sorted = sortNNData(NNData[0], NNData[1])

X_training = np.reshape(NNData_sorted[0], (-1,784))
X_test = np.reshape(NNData_sorted[1], (-1,784))
y_training = np.flatten(NNData_sorted[2])
y_test = np.flatten(NNData_sorted[3])

# Define and train NN model and make predictions
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp.fit(NNData_sorted[0], NNData_sorted[2])
yhat = mlp.predict(NNData_sorted[1])

# import data for yacht design into 2D table
yacht_data, vals = importYacht('yacht_data.csv')

# compute regression model for entire data set
yhat, r2 = linRegPredict(yacht_data, vals)

yhat_lpcb, r2_lpcb = linRegPredict(np.reshape(yacht_data[:,0], (-1,1)), vals)
yhat_pc, r2_pc = linRegPredict(np.reshape(yacht_data[:,1], (-1,1)), vals)
yhat_ldr, r2_ldr = linRegPredict(np.reshape(yacht_data[:,2], (-1,1)), vals)
yhat_bdr, r2_bdr = linRegPredict(np.reshape(yacht_data[:,3], (-1,1)), vals)
yhat_lbr, r2_lbr = linRegPredict(np.reshape(yacht_data[:,4], (-1,1)), vals)
yhat_fn, r2_fn = linRegPredict(np.reshape(yacht_data[:,5], (-1,1)), vals)
print(f'r2 = \n {r2_lpcb}\n{r2_pc}\n{r2_ldr}\n{r2_bdr}\n{r2_lbr}\n{r2_fn}\n')

# best predictors:
# fn    (col 5)
# pc    (col 1)
# lpcb  (col 0)

best_data = np.delete(yacht_data, [2,3,4], axis=1)
yhat_best_linear, r2_best_linear = linRegPredict(best_data, vals)
print(np.shape(np.delete(yacht_data, [2,3,4], axis=1)))

plt.figure()
plt.scatter(np.reshape(yacht_data[:,5], (-1,1)), vals)
plt.title('Froude Number vs. Outcome')
plt.xlabel('FN value')
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.savefig('figures/fn.png')

plt.figure()
plt.scatter(np.reshape(yacht_data[:,0], (-1,1)), vals)
plt.title('Longitudinal Position Center of Buoyancy vs. Outcome')
plt.xlabel('LPCB value')
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.savefig('figures/lpcb.png')

plt.figure()
plt.scatter(np.reshape(yacht_data[:,1], (-1,1)), vals)
plt.title('Prismatic Coefficient vs. Outcome')
plt.xlabel('PC value')
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.savefig('figures/pc.png')

# try a few other models
fn_exp = np.exp(20 * np.reshape(yacht_data[:,5], (-1,1)))
yhat_fn_exp, r2_fn_exp = linRegPredict(np.reshape(fn_exp, (-1,1)), vals)
print(f'exp(20*fn) r2 = {r2_fn_exp}')

fn_exp = np.exp(np.reshape(yacht_data[:,5], (-1,1)))
yhat_fn_exp, r2_fn_exp = linRegPredict(np.reshape(fn_exp, (-1,1)), vals)
print(f'exp(fn) r2 = {r2_fn_exp}')

#lpcb_data = yacht_data[:,0]
#pc_data = yacht_data[:,1]
comb_data1 = np.exp(20 * np.delete(yacht_data, [2,3,4], axis=1))
yhat_comb, r2_comb = linRegPredict(comb_data1, vals)
print(f'combined r2 = {r2_comb}')


plt.figure(dpi=170)
plt.plot(range(len(vals)), vals, linewidth=0.25, label='vals', marker='.')
plt.plot(range(len(yhat)), yhat_comb, linewidth=0.25, label='y_hat', marker='.')
plt.legend()
plt.xlabel('Instances')
plt.ylabel('Predicted Value')
plt.title('Yacht Resistance')
plt.savefig('figures/yacht_best.png')

print('yacht...')

# create SVM module
clf = svm.SVC(gamma='scale', kernel='linear')
# extract all titanic training data into data matrix and labels vector
titanic_data, survived = titIn.get_titanic_all('titanic_tsmod.csv')

# train and run predictions on all data
pred_val, percent_success = predict(titanic_data, survived, clf)
print(f'percent correct (all) = {percent_success}')

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
