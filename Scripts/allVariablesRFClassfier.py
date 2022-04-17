import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, fbeta_score
from sklearn.model_selection import StratifiedShuffleSplit
from pylab import savefig


df = pd.read_csv("/home/marcos/Documentos/Desenvolvimento/Projeto/Datasets/TEP_AllCases_accumulated_winlen_50_Trainval_norm.csv")
dfTest = pd.read_csv("/home/marcos/Documentos/Desenvolvimento/Projeto/Datasets/TEP_AllCases_accumulated_winlen_50_Test_norm.csv")
X = df.drop(['Fault_Class','simulationRun','window'], axis=1)
y = df['Fault_Class']
df = df.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.1))
dfTest = dfTest.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.1))

rfc = RandomForestClassifier(max_depth=12, min_samples_split=6, min_samples_leaf=20, max_features=30 )
val_sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
scores = []
train_indexes = []
for train_index, val_index in val_sss.split(X, y):
    X_train, X_val = X.loc[train_index], X.loc[val_index]
    y_train, y_val = y.loc[train_index], y.loc[val_index]
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_val)
    scores.append(accuracy_score(y_val, pred))
    train_indexes.append(train_index)

scores = np.array(scores)
train_index = train_indexes[scores.argmax()]
X_train = X.loc[train_index]
y_train = y.loc[train_index]
X_test = dfTest.drop(['Fault_Class','simulationRun','window'], axis=1)
y_test = dfTest['Fault_Class']
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)
cf_matrix = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(20,20))
results = sns.heatmap(cf_matrix, annot=True, cmap='coolwarm')
figure = results.get_figure()
figure.savefig('../Reports/rfc_conf.png') 