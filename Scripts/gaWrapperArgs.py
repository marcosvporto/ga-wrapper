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

p - populacao

g - gerações

m - probabilidade de mutação

x - probabilidade de cruzamento

s - seleção

 - r - roleta
 - t - torneio

'''
parser.add_argument("-t", "--target", default = "asn" ,help  = "a = (Single objective) Maximize mean accuracy;" 
                                            + " as = (Multi-objective) Maximize mean accuracy and minimize mean std deviation; "
                                            + " an = (Multi-objective) Maximize mean accuracy and minimize number of variables; "
                                            + " asn = (Multi-objective) Maximize mean accuracy, minimize mean std deviation and minimize number of variables;")
parser.add_argument("-c", "--crossover", default = 1 , help = "Crossover Method")
parser.add_argument("-p", "--population", default = 20 , help  = "Number of individuals in a population")
parser.add_argument("-g", "--generations", default = 50 , help = "Number of generations")
parser.add_argument("-m", "--mutationprob", default = 0.2 ,help = "Mutation Probability")
parser.add_argument("-x", "--xprob", default = 0.5 ,help = "Crossover Probability")
parser.add_argument("-s", "--selection",default = "r", help = "Selection Method: r = roulette; t = tournament")
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
        elif (args.target == "as")):
            return (scores.mean(),scores.std())
        elif (args.target == "an")):
            return (scores.mean(),individual.count(1)/52)
        else:    
            return (scores.mean(),scores.std(),individual.count(1)/52)
    else:
        if (args.target == "a"):
            return (0,)
        elif (args.target == "as")):
            return (0,1)
        elif (args.target == "an")):
            return (0,1)
        else:    
            return (0,1,1)
        

def geneticAlgorithm(X, y, n_population, n_generation):

    if (args.target == "a"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        elif (args.target == "as")):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        elif (args.target == "an")):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        else:    
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,creator.Individual, toolbox.attr_bool, n=len(X.columns))
    toolbox.register("population", tools.initRepeat, list,toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selRoulette)

    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                    ngen=n_generation, stats=stats, halloffame=hof,
                                    verbose=True)
    return hof, pop, log