from generateClassifier import generateClassifier 

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

target_options = ["a", "an"]
selection_options = ["r", "t"]
crossover_options = [ 1, 2]
population_options = [10,20]
generation_options = [20, 40]
mutation_prob_options = [0.01, 0.05]
crossover_prob_options = [0.5, 0.9]


for to in target_options:
    for so in selection_options:
        for co in crossover_options:
            for  po in population_options :
                for go in generation_options:
                    for  mpo in mutation_prob_options :
                        for cpo in crossover_prob_options:
                            generateClassifier(real=False, 
                                                target = to, 
                                                selection=so, 
                                                crossover=co, 
                                                population=po, 
                                                generations=go, 
                                                mutationprob=mpo, 
                                                xprob=cpo)