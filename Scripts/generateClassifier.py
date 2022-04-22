from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from geneticAlgorithm import geneticAlgorithm
from bestIndividual import bestIndividual
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report



def generateClassifier(real, target, selection, crossover, population, generations, mutationprob, xprob, *args, **kwargs):

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

    hof, pop, log = geneticAlgorithm(X, y, target, crossover,selection, population, xprob, mutationprob, generations)

    score, individual, header = bestIndividual(hof, X, y)
    print('Best Score: \t' + str(score))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print('\n\ncreating a new classifier with the result')
    max_feat = individual.count(1) 
    rfc = RandomForestClassifier(max_depth=12, 
                                min_samples_split=6, 
                                n_estimators = 100,
                                criterion="gini",
                                max_features=max_feat,
                                min_samples_leaf=20)
    cols = [index for index in range(len(individual)) if individual[index] == 0]
    X = X.drop(X.columns[cols], axis = 1)
    X_test = X_test.drop(X_test.columns[cols], axis = 1)
    scores = []
    train_indexes = []

    print("Variables:",X.columns)
    rfc.fit(X,y)
    pred = rfc.predict(X_test)
    print("Accuracy Score:",accuracy_score(y_test, pred))
    cf_matrix = confusion_matrix(y_test, pred, labels = rfc.classes_)
    fig, axes = plt.subplots(figsize=((max_feat,max_feat)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=rfc.classes_)
    disp.plot()
    plt.savefig('../Reports/'+target
                                +'_sel_'+str(selection)
                                +'_cross_'+str(crossover)
                                +'_p_'+str(population)
                                +'_g_'+str(generations)
                                +'_mp_'+str(mutationprob)
                                +'_xp_'+str(xprob)
                                +'rfc_conf.png') 
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    gen_performance = [item[0] for item in log.select("max")]
    plt.plot(gen_performance)
    plt.title("Acompanhamento da Performance")
    plt.savefig("../Reports/"+target
                                +'_sel_'+str(selection)
                                +'_cross_'+str(crossover)
                                +'_p_'+str(population)
                                +'_g_'+str(generations)
                                +'_mp_'+str(mutationprob)
                                +'_xp_'+str(xprob)
                                +"ga.png")