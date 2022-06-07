from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from geneticAlgorithm import geneticAlgorithm
from bestIndividual import bestIndividual
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from joblib import dump
import os
import numpy as np
import pickle


df = pd.read_csv("../Datasets/TEP_AllCases_accumulated_winlen_50_Trainval_norm_20_percent.csv")
dfTest = pd.read_csv("../Datasets/TEP_AllCases_accumulated_winlen_50_Test_norm_20_percent.csv")


X = df.drop(['Fault_Class','simulationRun','window'], axis=1)
y = df['Fault_Class']

X_test = dfTest.drop(['Fault_Class','simulationRun','window'], axis=1)
y_test = dfTest['Fault_Class']

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
rfc = RandomForestClassifier(max_depth=12, 
                            min_samples_split=6, 
                            n_estimators = 100,
                            criterion="gini",
                            max_features=52,
                            min_samples_leaf=20,
                            n_jobs=-1)
scores = []
train_indexes = []
models = []
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_test)
    models.append(pickle.dumps(rfc))
    scores.append(accuracy_score(y_test, pred))
    train_indexes.append(train_index)
scores = np.array(scores)
train_index = train_indexes[scores.argmax()]
rfc = pickle.loads(models[scores.argmax()])


acc_score = accuracy_score(y_test, pred)
str_acc_score = str(round(acc_score, 2)).replace('.','')
dump(rfc, '../Models/allFeatures_acc_'+str_acc_score+'.joblib')

filename = str('../Reports/allFeatures/results.txt')
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "a") as f:
    f.write('\nAll Features Result:')
    f.write('\nTest Accuracy Score: \t' + str(acc_score))
    f.close()

cf_matrix = confusion_matrix(y_test, pred, labels = rfc.classes_)
font = {'size'   : 6}
plt.rc('font', **font)
fig, axes = plt.subplots(figsize=((200,200)))
disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=rfc.classes_)
disp.plot()
plt.savefig('../Reports/allFeatures/confmatrix.png') 
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()








