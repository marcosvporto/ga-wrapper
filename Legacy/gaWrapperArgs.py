import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from pylab import savefig
from joblib import dump, load
from deap import base, creator, algorithms, tools
import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

'''
t - target = Objetivo  
 - a -> acurácia
 - as -> acurácia média e desvio padrao médio
 - an -> acuária média e número de variáveis
 - asn -> acuárica média, desvio padrão médio e número de variáveis

c - crossover = cruzamento
 - default = 1 ponto
 - 2 = 2 pontos
 - 3  = Uniforme com probabilidade 0.03% de um atributo ser modificado
 - 4  = Uniforme com probabilidade 0.04% de um atributo ser modificado
 - 5  = Uniforme com probabilidade 0.05% de um atributo ser modificado


p - populacao

g - gerações

m - probabilidade de mutação

x - probabilidade de cruzamento

s - seleção

 - r - roleta
 - t - torneio
 - b - melhor


d - desenvolvimento
'''
parser.add_argument("-t", "--target"      , default = "asn" , help  = "a = (Single objective) Maximize mean accuracy;" 
                                                                    + " as = (Multi-objective) Maximize mean accuracy and minimize mean std deviation; "
                                                                    + " an = (Multi-objective) Maximize mean accuracy and minimize number of variables; "
                                                                    + " asn = (Multi-objective) Maximize mean accuracy, minimize mean std deviation and minimize number of variables;")
parser.add_argument("-s", "--selection"   ,default = "r"    , help = "Selection Method: r = roulette; t = tournament; b = best")
parser.add_argument("-c", "--crossover"   , default = 1     , help = "Crossover Method")
parser.add_argument("-p", "--population"  , default = 20    , help = "Number of individuals in a population")
parser.add_argument("-g", "--generations" , default = 50    , help = "Number of generations")
parser.add_argument("-m", "--mutationprob", default = 0.2   , help = "Mutation Probability")
parser.add_argument("-x", "--xprob"       , default = 0.5   , help = "Crossover Probability")
parser.add_argument("-d", "--definite", help = "Run definetively")
args = parser.parse_args()  

def getFitness(individual, X, y):
    if(individual.count(0) < len(individual)):
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        X = X.drop(X.columns[cols], axis = 1)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        max_feat = individual.count(1) 
        rfc = RandomForestClassifier(max_depth=12, 
                                     min_samples_split=6, 
                                     n_estimators = 100,
                                     criterion="gini",
                                     max_features=max_feat,
                                     min_samples_leaf=20)
        scores = []
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            rfc.fit(X_train, y_train)
            pred = rfc.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        scores = np.array(scores)
        if (args.target == "a"):
            return (scores.mean(),)
        elif (args.target == "as"):
            return (scores.mean(),scores.std())
        elif (args.target == "an"):
            return (scores.mean(),individual.count(1)/52)
        elif (args.target == "asn"):    
            return (scores.mean(),scores.std(),individual.count(1)/52)
        else:
            raise argparse.ArgumentTypeError('Invalid Targed Value')
    else:
        if (args.target == "a"):
            return (0,)
        elif (args.target == "as"):
            return (0,1)
        elif (args.target == "an"):
            return (0,1)
        elif (args.target == "asn"):    
            return (0,1,1)
        else:
            raise argparse.ArgumentTypeError('Invalid Targed Value')


def geneticAlgorithm(X, y):

    if (args.target == "a"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    elif (args.target == "as"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    elif (args.target == "an"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    elif (args.target == "asn"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    else:
        raise argparse.ArgumentTypeError('Invalid Targed Value')

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,creator.Individual, toolbox.attr_bool, n=len(X.columns))
    toolbox.register("population", tools.initRepeat, list,toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)


    if (args.crossover == 1):
        toolbox.register("mate", tools.cxOnePoint)
    elif (args.crossover == 2):
        toolbox.register("mate", tools.cxTwoPoint)
    elif(args.crossover > 2 and args.crossover < 10 ):
        toolbox.register("mate", tools.cxuniform, indpb=args.crossover/100)
    else:
        raise argparse.ArgumentTypeError('Value has to be between 1 and 9')
            
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)


    if(args.selection == "r"):
        toolbox.register("select", tools.selRoulette)
    elif(args.selection == "t"):
        toolbox.register("select", tools.selTournament)
    elif(args.selection == "b"):
        toolbox.register("select", tools.selBest)
    else:
        raise argparse.ArgumentTypeError('Invalid Argument Value')


    pop = toolbox.population(n=args.population)
    hof = tools.HallOfFame(args.population * args.generations)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=args.xprob, mutpb=args.mutationprob,
                                    ngen=args.generations, stats=stats, halloffame=hof,
                                    verbose=True)
    return hof, pop, log


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:
        if(individual.fitness.values[0] > maxAccurcy):
            maxAccurcy = individual.fitness.values[0]
            _individual = individual

    _individualHeader = [list(X.columns)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader

df = pd.read_csv("/home/marcos/Documentos/Desenvolvimento/Projeto/Datasets/TEP_AllCases_accumulated_winlen_50_Trainval_norm_group_marcos.csv")
dfTest = pd.read_csv("/home/marcos/Documentos/Desenvolvimento/Projeto/Datasets/TEP_AllCases_accumulated_winlen_50_Test_norm_group_marcos.csv")

if (not args.definite):
    print("OBS: Running a reduced version of the Data")
    df = df.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))
    dfTest = dfTest.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))

X = df.drop(['Fault_Class','simulationRun','window'], axis=1)
y = df['Fault_Class']

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

X_test = dfTest.drop(['Fault_Class','simulationRun','window'], axis=1)
y_test = dfTest['Fault_Class']

hof, pop, log = geneticAlgorithm(X, y)

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
plt.savefig('../Reports/'+args.target
                            +'_sel_'+str(args.selection)
                            +'_cross_'+str(args.crossover)
                            +'_p_'+str(args.population)
                            +'_g_'+str(args.generations)
                            +'_mp_'+str(args.mutationprob)
                            +'_xp_'+str(args.xprob)
                            +'/rfc_conf.png') 
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()
gen_performance = [item[0] for item in log.select("max")]
plt.plot(gen_performance)
plt.title("Acompanhamento da Performance")
plt.savefig("../Reports/"+args.target
                            +'_sel_'+str(args.selection)
                            +'_cross_'+str(args.crossover)
                            +'_p_'+str(args.population)
                            +'_g_'+str(args.generations)
                            +'_mp_'+str(args.mutationprob)
                            +'_xp_'+str(args.xprob)
                            +"/ga.png")