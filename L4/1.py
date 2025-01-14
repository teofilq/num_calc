import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

fisier = "iris.csv"
col = ['lung_sepala', 'lat_sepala', 'lung_petala', 'lat_petala', 'clasa']
date = pd.read_csv(fisier, header=None, names=col)
date['clasa'] = date['clasa'].astype('category').cat.codes

X = date.iloc[:, :-1].values
y = date.iloc[:, -1].values

idx = np.arange(len(X))
random.shuffle(idx)
X = X[idx]
y = y[idx]

N_train = 100
train_idx = random.sample(range(len(X)), N_train)
test_idx = np.setdiff1d(range(len(X)), train_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

lin_svm = svm.SVC(kernel='linear')
lin_svm.fit(X_train, y_train)
lin_pred = lin_svm.predict(X_test)
lin_acc = accuracy_score(y_test, lin_pred)
lin_conf = confusion_matrix(y_test, lin_pred)

rbf_svm = svm.SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)
rbf_acc = accuracy_score(y_test, rbf_pred)
rbf_conf = confusion_matrix(y_test, rbf_pred)

print("Liniar:", lin_acc, "\n", lin_conf)
print("RBF:", rbf_acc, "\n", rbf_conf)