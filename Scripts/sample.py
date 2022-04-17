import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import dump, load
df = pd.read_csv('/home/marcos/Documentos/Desenvolvimento/Projeto/Datasets/kyphosis.csv')


X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
test_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

for train_val_index, test_index in test_sss.split(X, y):
    X_train_val, X_test = X.loc[train_val_index], X.loc[test_index]
    y_train_val, y_test = y.loc[train_val_index], y.loc[test_index]
rfc = RandomForestClassifier(n_estimators = 200)
val_sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
X_train_val.reset_index(drop=True, inplace=True)
y_train_val.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
scores = []
train_indexes = []
for train_index, val_index in val_sss.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val.loc[train_index], X_train_val.loc[val_index]
    y_train, y_val = y_train_val.loc[train_index], y_train_val.loc[val_index]
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_val)
    scores.append(accuracy_score(y_val, pred))
    train_indexes.append(train_index)

scores = np.array(scores)
train_index = train_indexes[scores.argmax()]
X_train = X.loc[train_index]
y_train = y.loc[train_index]
f_size = y_test.nunique()
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)
cf_matrix = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(f_size,f_size))
results = sns.heatmap(cf_matrix, annot=True, cmap='coolwarm')
figure = results.get_figure()
figure.savefig('../Reports/sample_rfc_conf.png') 