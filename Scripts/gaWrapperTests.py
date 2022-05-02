from generateClassifier import generateClassifier 
import pandas as pd
from time import time
import os
import csv 


real = True

# '''
# t - target = Objetivo  
#  - a -> acuracia
#  - as -> acuracia media e desvio padrao medio
#  - an -> acuracia media e numero de variaveis
#  - asn -> acuarica media, desvio padrao medio e numero de variaveis

# c - crossover = cruzamento
#  - default = 1 ponto
#  - 2 = 2 pontos
#  - 3  = Uniforme com probabilidade 0.03% de um atributo ser modificado
#  - 4  = Uniforme com probabilidade 0.04% de um atributo ser modificado
#  - 5  = Uniforme com probabilidade 0.05% de um atributo ser modificado


# p - populacao

# g - geracoes

# m - probabilidade de mutacao

# x - probabilidade de cruzamento

# s - selecao

#  - r - roleta
#  - t - torneio
#  - b - melhor


# d - desenvolvimento
# '''

target_options = ["an"]
selection_options = ["t"]
crossover_options = [2]
population_options = [20]
generation_options = [20]
mutation_prob_options = [0.1]
crossover_prob_options = [1]
elitism_options = [True]

latest = ''
for to in target_options:
        for so in selection_options:
            for co in crossover_options:
                for  po in population_options :
                    for go in generation_options:
                        for  mpo in mutation_prob_options :
                            for cpo in crossover_prob_options:
                                for eo in  elitism_options:
                                    foldername = ('../Reports/target_'+str(to)
                                                    +'_sel_'+str(so)
                                                    +'_cross_'+str(co)
                                                    +'_p_'+str(po)
                                                    +'_g_'+str(go)
                                                    +'_mp_'+str(mpo)
                                                    +'_xp_'+str(cpo)
                                                    +'_elt_'+str(eo)
                                                    +('_real' if real else '_dev'))
                                    if not os.path.isdir(foldername):
                                        break
                                    else:
                                        latest = foldername




fields = ['accuracy','features','target', 'selection', 'crossover', 'population', 'generations', 'mutprob', 'crossprob','elitism','duration']
if not os.path.isfile('../Reports/'+('real' if real else 'dev')+'GenAlgWrapperTests.csv'):
    print("First Time Running Tests")
    inputHeader = True
else:
    print("Resuming Tests")
    inputHeader = False

df = pd.read_csv("../Datasets/TEP_AllCases_accumulated_winlen_50_Trainval_norm_20_percent.csv")
dfTest = pd.read_csv("../Datasets/TEP_AllCases_accumulated_winlen_50_Test_norm_20_percent.csv")

if (not real):
    print("OBS: Running a reduced version of the Data")
    df = df.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))
    dfTest = dfTest.groupby('Fault_Class').apply(lambda x:x.sample(frac=0.001))

df.reset_index(drop=True, inplace=True)
dfTest.reset_index(drop=True, inplace=True)
with open('../Reports/'+('real' if real else 'dev')+'GenAlgWrapperTests.csv', 'a', newline='') as f_output:
    csv_output = csv.DictWriter(f_output, fieldnames = fields, restval = 'NA')
    if inputHeader:
        csv_output.writeheader()
    for to in target_options:
        for so in selection_options:
            for co in crossover_options:
                for  po in population_options :
                    for go in generation_options:
                        for  mpo in mutation_prob_options :
                            for cpo in crossover_prob_options:
                                for eo in  elitism_options:
                                    foldername = ('../Reports/target_'+str(to)
                                                    +'_sel_'+str(so)
                                                    +'_cross_'+str(co)
                                                    +'_p_'+str(po)
                                                    +'_g_'+str(go)
                                                    +'_mp_'+str(mpo)
                                                    +'_xp_'+str(cpo)
                                                    +'_elt_'+str(eo)
                                                    +('_real' if real else '_dev'))
                                    if not os.path.isdir(foldername) or foldername == latest:
                                        if foldername == latest:
                                            print("Repeating Latest Test to avoid corrupted data")
                                        start = time()
                                        acc_score, n_features, header = generateClassifier( df=df, 
                                                                                            dfTest=dfTest,
                                                                                            real=real, 
                                                                                            target = to, 
                                                                                            selection=so, 
                                                                                            crossover=co, 
                                                                                            population=po, 
                                                                                            generations=go, 
                                                                                            mutationprob=mpo, 
                                                                                            xprob=cpo,
                                                                                            elitism = eo)
                                        end = time()
                                        #duration = "{:.2f}".format((end - start)/3600)
                                        duration = round((end - start)/3600,2)
                                        row = {
                                            'accuracy'      : acc_score,
                                            'features'      : n_features,
                                            'target'        : to , 
                                            'selection'     : so, 
                                            'crossover'     : co, 
                                            'population'    : po, 
                                            'generations'   : go, 
                                            'mutprob'       : mpo, 
                                            'crossprob'     : cpo,
                                            'elitism'       : eo,
                                            'duration'      : duration
                                        }
                                        csv_output.writerow(row)
                                    
                                    else:
                                        print("Skipping test with already tested hyperparameters")
    f_output.close()          
                     