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

'''
Função de Avaliação de um Indivíduo
===================================

Se um indivíduo não possui nenhuma variável então ele é punido. 
Pois não faz sentido um modelo com nenhuma variável.

Inicialmente seleciona-se as colunas ausentes no cromossomo. 
Em seguidas elas são removidas do dataset.
É feito uma separação dos dados para a validação cruzada:
 - 20% será para validação e 80% será para treino do modelo
 - Serão 5 divisões aleatórias e Estratificadas dos dados
  - Ou seja, Será mantida a proporção das classes de erros nas divisões aleatórias

 - O modelo é treinado com cada uma das 5 divisões dos dados e avaliado
 - A avaliação do indivíduo é a média do desempenho do modelo nas 5 divisões
  - Além do desvio padrão do desempenho
  - E o número de variáveis que o indivíduo usa

'''
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
        return (scores.mean(),scores.std(),individual.count(1)/52)
    else:
        return (0,1,1)

'''
    Algoritmo Genético
    ==================

    É utilizada a biblioteca DEAP para rodar o algoritmo genético
    Esta biblioteca oferece uma toolbox paa criarmos os recursos necessários para o GA
    Os recursos são:
     - Objetivo: Multi-obejtivo ou Mono-objetivo, Maximização ou Minimização
        - Vamos testar:
            - Mono-objetivo: Maximizar a acurácia média
            - Multi-objetivo: Maximizar a acurácia média, Minimizar o desvio-padrão da acurácia
            - Multi-objetivo: Maximizar a acurácia média, Minimizar o número de variáveis
            - Multi-objetivo: Maximizar a acurácia média, Minimizar o desvio-padrão da acurácia e Minimizar o número de variáveis

     - Formato do Cromossomo: Array de Booleanos
     - Crossover: Como geramos novos indivíduos a partir dos indivíduos antigos
        - Vamos testar:
            - Crossover de um ponto
     - Mutação:
        - Vamos testar:
            - Mutação de Flip Bit: Troca de 0 para 1 e vice-versa
     - Seleção:
        - Vamos testar:
            - Roleta
            - Torneio
    

'''

def geneticAlgorithm(X, y, n_population, n_generation):

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


'''
Melhor Indivíduo
'''
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
'''
Fracionamento para testes
Remover quando rodar as avaliações finais

'''
df = df.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))
dfTest = dfTest.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))


X = df.drop(['Fault_Class','simulationRun','window'], axis=1)
y = df['Fault_Class']

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

X_test = dfTest.drop(['Fault_Class','simulationRun','window'], axis=1)
y_test = dfTest['Fault_Class']

n_pop = 4
n_gen = 10
hof, pop, log = geneticAlgorithm(X, y, n_pop, n_gen)

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
# val_sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
# for train_index, val_index in val_sss.split(X, y):
#     X_train, X_val = X.loc[train_index], X.loc[val_index]
#     y_train, y_val = y.loc[train_index], y.loc[val_index]
#     train_indexes.append(train_index)
#     rfc.fit(X_train, y_train)
#     pred = rfc.predict(X_val)
#     scores.append(accuracy_score(y_val, pred))
# scores = np.array(scores)
# train_index = train_indexes[scores.argmax()]
# X_train = X.loc[train_index]
# y_train = y.loc[train_index]
print("Variables:",X.columns)
rfc.fit(X,y)
pred = rfc.predict(X_test)
print("Accuracy Score:",accuracy_score(y_test, pred))
cf_matrix = confusion_matrix(y_test, pred, labels = rfc.classes_)
fig, axes = plt.subplots(figsize=((max_feat,max_feat)))
disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=rfc.classes_)
disp.plot()
plt.savefig('../Reports/rfc_conf.png') 
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()
gen_performance = [item[0] for item in log.select("max")]
plt.plot(gen_performance)
plt.title("Acompanhamento da Performance")
plt.savefig("../Reports/ga.png")