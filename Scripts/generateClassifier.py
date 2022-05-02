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

def generateClassifier(df, dfTest, real, target, selection, crossover, population, generations, mutationprob, xprob, elitism, *args, **kwargs):

    
    X = df.drop(['Fault_Class','simulationRun','window'], axis=1)
    y = df['Fault_Class']

    X_test = dfTest.drop(['Fault_Class','simulationRun','window'], axis=1)
    y_test = dfTest['Fault_Class']

    hof, pop, log = geneticAlgorithm(X, y, target, crossover,selection, population, xprob, mutationprob, generations,elitism,real)

    score, individual, header = bestIndividual(hof, X, y)


    filename = str('../Reports/target_'+str(target)
                                                                                +'_sel_'+str(selection)
                                                                                +'_cross_'+str(crossover)
                                                                                +'_p_'+str(population)
                                                                                +'_g_'+str(generations)
                                                                                +'_mp_'+str(mutationprob)
                                                                                +'_xp_'+str(xprob)
                                                                                +'_elt_'+str(elitism)
                                                                                +('_real' if real else '_dev')
                                                                                +'/results.txt')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write('\nBest Score: \t' + str(score))
        f.write('\nNumber of Features in Subset: \t' + str(individual.count(1)))
        f.write('\nIndividual: \t\t' + str(individual))
        f.write('\nFeature Subset\t: ' + str(header))
        f.close()
    n_features =  individual.count(1)
    sss = StratifiedShuffleSplit(n_splits=5 if real else 1, test_size=0.2)
    rfc = RandomForestClassifier(max_depth=12, 
                                min_samples_split=6, 
                                n_estimators = 100,
                                criterion="gini",
                                max_features=n_features,
                                min_samples_leaf=20,
                                n_jobs=-1)
    cols = [index for index in range(len(individual)) if individual[index] == 0]
    X = X.drop(X.columns[cols], axis = 1)
    X_test = X_test.drop(X_test.columns[cols], axis = 1)
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
    #rfc.fit(X.loc[train_index],y.loc[train_index])
    #rfc.fit(X,y)
    str_ind = [str(elem) for elem in individual]
    pred = rfc.predict(X_test)
    
    acc_score = accuracy_score(y_test, pred)
    str_acc_score = str(round(acc_score, 2)).replace('.','')
    dump(rfc, '../Models/'+(''.join(str_ind))+'_acc_'+str_acc_score+'.joblib')
    with open(filename, "a") as f:
        f.write('\nTest Accuracy Score: \t' + str(acc_score))
        f.close()
    
    cf_matrix = confusion_matrix(y_test, pred, labels = rfc.classes_)
    font = {'size'   : 6}
    plt.rc('font', **font)
    fig, axes = plt.subplots(figsize=((200,200)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=rfc.classes_)
    disp.plot()
    plt.savefig('../Reports/target_'+str(target)
                +'_sel_'+str(selection)
                +'_cross_'+str(crossover)
                +'_p_'+str(population)
                +'_g_'+str(generations)
                +'_mp_'+str(mutationprob)
                +'_xp_'+str(xprob)
                +'_elt_'+str(elitism)
                +('_real' if real else '_dev')
                +'/confmatrix.png') 
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    gen_performance = [item[0] for item in log.select("max")]
    plt.plot(gen_performance)
    plt.title("Acompanhamento da Performance")
    plt.savefig('../Reports/target_'+str(target)
                +'_sel_'+str(selection)
                +'_cross_'+str(crossover)
                +'_p_'+str(population)
                +'_g_'+str(generations)
                +'_mp_'+str(mutationprob)
                +'_xp_'+str(xprob)
                +'_elt_'+str(elitism)
                +('_real' if real else '_dev')
                +'/perform.png')

    df_log = pd.DataFrame(log)

    df_log.to_csv('../Reports/target_'+str(target)
                +'_sel_'+str(selection)
                +'_cross_'+str(crossover)
                +'_p_'+str(population)
                +'_g_'+str(generations)
                +'_mp_'+str(mutationprob)
                +'_xp_'+str(xprob)
                +'_elt_'+str(elitism)
                +('_real' if real else '_dev')
                +'/ga.csv')
    return acc_score, n_features, header