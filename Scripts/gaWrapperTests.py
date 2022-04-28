from generateClassifier import generateClassifier 
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

target_options = ["an", "a"]
selection_options = ["r", "t"]
crossover_options = [1, 2]
population_options = [20,50]
generation_options = [50, 100]
mutation_prob_options = [0.05, 0.1]
crossover_prob_options = [0.5, 0.9]
elitism_options = [True, False]

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
if not os.path.isfile('../Reports/gaWrapperTests.csv'):
    print("First Time Running Tests")
    inputHeader = True
else:
    print("Resuming Tests")
    inputHeader = False

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
                                        acc_score, n_features, header = generateClassifier( real=real, 
                                                                                            target = to, 
                                                                                            selection=so, 
                                                                                            crossover=co, 
                                                                                            population=po, 
                                                                                            generations=go, 
                                                                                            mutationprob=mpo, 
                                                                                            xprob=cpo,
                                                                                            elitism = eo)
                                        end = time()
                                        duration = "{:.2f} h".format((end - start)/3600)
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
                     