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

def generateClassifier(real, target, selection, crossover, population, generations, mutationprob, xprob, elitism, *args, **kwargs):

    df = pd.read_csv("/home/marcos/Documentos/Desenvolvimento/Projeto/Datasets/TEP_AllCases_accumulated_winlen_50_Trainval_norm_group_marcos.csv")
    dfTest = pd.read_csv("/home/marcos/Documentos/Desenvolvimento/Projeto/Datasets/TEP_AllCases_accumulated_winlen_50_Test_norm_group_marcos.csv")

    if (not real):
        print("OBS: Running a reduced version of the Data")
        df = df.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))
        dfTest = dfTest.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))

    X = df.drop(['Fault_Class','simulationRun','window'], axis=1)
    y = df['Fault_Class']

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X_test = dfTest.drop(['Fault_Class','simulationRun','window'], axis=1)
    y_test = dfTest['Fault_Class']

    hof, pop, log = geneticAlgorithm(X, y, target, crossover,selection, population, xprob, mutationprob, generations,elitism,real)

    score, individual, header = bestIndividual(hof, X, y)


    filename = str('/home/marcos/Documentos/Desenvolvimento/Projeto/Reports/target_'+str(target)
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
                                min_samples_leaf=20)
    cols = [index for index in range(len(individual)) if individual[index] == 0]
    X = X.drop(X.columns[cols], axis = 1)
    X_test = X_test.drop(X_test.columns[cols], axis = 1)
    scores = []
    train_indexes = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        rfc.fit(X_train, y_train)
        pred = rfc.predict(X_test)
        scores.append(accuracy_score(y_test, pred))
        train_indexes.append(train_index)
    scores = np.array(scores)
    train_index = train_indexes[scores.argmax()]

    rfc.fit(X.loc[train_index],y.loc[train_index])
    str_ind = [str(elem) for elem in individual]
    dump(rfc, '../Models/'+(''.join(str_ind))+'.joblib')
    pred = rfc.predict(X_test)
    
    acc_score = accuracy_score(y_test, pred)
    with open(filename, "a") as f:
        f.write('\nTest Accuracy Score: \t' + str(acc_score))
        f.close()
    
    cf_matrix = confusion_matrix(y_test, pred, labels = rfc.classes_)
    fig, axes = plt.subplots(figsize=((50,50)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=rfc.classes_)
    disp.plot()
    plt.savefig('/home/marcos/Documentos/Desenvolvimento/Projeto/Reports/target_'+str(target)
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
    plt.savefig('/home/marcos/Documentos/Desenvolvimento/Projeto/Reports/target_'+str(target)
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

    df_log.to_csv('/home/marcos/Documentos/Desenvolvimento/Projeto/Reports/target_'+str(target)
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