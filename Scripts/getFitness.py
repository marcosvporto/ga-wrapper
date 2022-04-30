import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

def getFitness(individual, X, y, target, real, *args, **kwargs):
    if(individual.count(0) < len(individual)):
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        X = X.drop(X.columns[cols], axis = 1)
        sss = StratifiedShuffleSplit(n_splits=1 if real else 1, test_size=0.2)
        rfc = RandomForestClassifier(max_depth=12, 
                                     min_samples_split=6, 
                                     n_estimators = 100,
                                     criterion="gini",
                                     max_features=individual.count(1),
                                     min_samples_leaf=20,
                                     n_jobs=-1)

        scores = []
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            rfc.fit(X_train, y_train)
            pred = rfc.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        scores = np.array(scores)
        if (target == "a"):
            return (scores.mean(),)
        elif (target == "as"):
            return (scores.mean(),scores.std())
        elif (target == "an"):
            return (scores.mean(),individual.count(1)/52)
        elif (target == "asn"):    
            return (scores.mean(),scores.std(),individual.count(1)/52)
        else:
            raise TypeError('Invalid Targed Value')
    else:
        if (target == "a"):
            return (0,)
        elif (target == "as"):
            return (0,1)
        elif (target == "an"):
            return (0,1)
        elif (target == "asn"):    
            return (0,1,1)
        else:
            raise TypeError('Invalid Targed Value')