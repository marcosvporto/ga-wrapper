from generateClassifier import generateClassifier 
import os
import csv 

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

target_options = ["an", "a"]
selection_options = ["r", "t"]
crossover_options = [1, 2]
population_options = [10,20]
generation_options = [20, 40]
mutation_prob_options = [0.05, 0.1]
crossover_prob_options = [0.5, 0.9]
elitism_options = [True, False]

fields = ['accuracy','features','target', 'selection', 'crossover', 'population', 'generations', 'mutprob', 'crossprob','elitism']
with open('../Reports/gaWrapperTests.csv', 'w', newline='') as f_output:
    csv_output = csv.DictWriter(f_output, fieldnames = fields, restval = 'NA')
    csv_output.writeheader()
    for to in target_options:
        for so in selection_options:
            for co in crossover_options:
                for  po in population_options :
                    for go in generation_options:
                        for  mpo in mutation_prob_options :
                            for cpo in crossover_prob_options:
                                for eo in  elitism_options:
                                    acc_score, n_features, header = generateClassifier( real=False, 
                                                                                        target = to, 
                                                                                        selection=so, 
                                                                                        crossover=co, 
                                                                                        population=po, 
                                                                                        generations=go, 
                                                                                        mutationprob=mpo, 
                                                                                        xprob=cpo,
                                                                                        elitism = eo)
                                    row = {
                                        'accuracy'      : acc_score,
                                        'features'      : n_features,
                                        'target'        : to , 
                                        'selection'     : so, 
                                        'crossover'     : co, 
                                        'population'    : po, 
                                        'generations'   :go, 
                                        'mutprob'       :mpo, 
                                        'crossprob'     :cpo,
                                        'elitism'       : eo
                                    }
                                    csv_output.writerow(row)           
    f_output.close()                 