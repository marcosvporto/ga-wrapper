# Projeto de Graduação em Engenharia da Computação PUC-Rio 2022.1

## Algoritmo Genético para Seleção de Variáveis na CLassificação de Falhas em Processos Químicos de Larga Escala

### Modo de Usar
...
```
$ git clone https://github.com/marcosvporto/ga-wrapper.git
```

```
$ cd ga-wrapper/Scripts
```
```
$ nano gaWrapperTests.py
```
Edite os hiperparâmetros de teste como explicado nos comentários do arquivo e seguindo seguinte template
```
target_options = ["an"]
selection_options = ["t","r","b"]
crossover_options = [2]
population_options = [20,50]
generation_options = [40,80]
mutation_prob_options = [0.1]
crossover_prob_options = [1]
elitism_options = [True]
```
Para executar o script de testes em segundo plano e mandando os logs para um arquivo de saída:
```
$ python3 gaWrapperTests.py >> output.txt &
```
## Modulos

### gaWrapperTests.py

Este módulo pretende organizar a geração de testes automatizados. O utilizador do código deve alterar esse módulo de acordo com a forma que se deseja implementar os testes. Para economizar o tempo de processamento gasto para carregar as bases de dados na memória, esse módulo realiza o carregamento das bases e passa-as como parâmetro para o módulo de criação dos modelos (generateClassifier.py). Nesse módulo o usuário deve configurar quais são os hiperparâmetros que o mesmo deseja utilizar em seus testes e o módulo se encarrega de criar testes com todas as combinações possíveis de hiperparâmetros. É possível alterar o código para que este aceite várias opções de hiperparâmetros. 


### generateClassifier.py

Esse módulo é acionado pelo módulo de Criação dos Testes (gaWrapperTests.py) e tem como principal objetivo a criação do modelo do classificador, além de gerar relatórios sobre seu desempenho no teste. Esse módulo deve receber as bases de dados e os hiperparâmetros do Algoritmo Genético para que possa acionar o módulo de geração do GA. O módulo de geração do Algoritmo Genético então retorna o melhor indivíduo, ou seja, o subconjunto de variáveis que possui o melhor desempenho dentre todas os subconjuntos testados nas iterações do Algoritmo Genético. Esse indivíduo é então utilizado para gerar o modelo do classificador. O classificador gerado é então armazenado no diretório de modelos no formato .joblib para poder ser reutilizado sem a necessidade de retreinamento.    

### geneticAlgorithm.py

Este módulo realiza a criação do Algoritmo Genético utilizando a biblioteca DEAP, com base nos hiperparâmetros do Algoritmo Genético passados como parâmetro pelo módulo de Geração do Classificador (generateClassifier.py). A Figura X contém um exemplo de como a biblioteca DEAP deve ser utilizada para a criação de uma espécie de caixa de ferramentas que orquestrará as iterações do GA. 
![image](https://user-images.githubusercontent.com/39508000/177012417-1e9c3270-6ce3-4952-b2a1-41b7cc113a9e.png)

Na linha 5 do exemplo apresentado na figura, ocorre a definição da Função Objetivo do modelo, nesse caso, é uma função multiobjetivo. Assim, existem dois objetivos, é possível inferir que o primeiro é o de maximização do primeiro valor retornado pela função que identifica a acurácia obtida pelo algoritmo de classificação e o outro é minimizar o (maximizar o negativo do) segundo valor retornado pela função fitness. 

O Indivíduo é criado e registrado na caixa de ferramenta entre as linhas 6 e 14. Observe que nesse trecho já definido que o Indivíduo será representado por uma lista de booleanos (zeros e uns) gerados incialmente de forma aleatória e que seu tamanho será igual ao número de colunas da base de dados de entrada.  

O restante das demais entidades do Algoritmo Genético são criadas em seguida seguindo o mesmo padrão. A população inicial é gerada. Esse não é, entretanto, o código real do módulo, que foi adaptado para atender a diferentes hiperparâmetros. 

### libElitism.py

Este módulo serve apenas para sobrecarregar o método eaSimple() (observe na figura da seção anterior) presente na biblioteca DEAP que implementa as iterações do Algoritmo Genético para que seja implementada seguindo o conceito de elitismo. Dessa forma se garantirá que o melhor Indivíduo de cada geração esteja presente na geração seguinte. 

O método eaSimple() – que já é implementada pela biblioteca DEAP, ou a sua versão sobrecarregada eaSimpleWithElitsm(), é responsável pelas iterações do GA, esse método recebe como parâmetros a caixa de ferramentas criada no módulo de criação do Algoritmo Genético (geneticAlgorithm.py) além da população inicial e um objeto chamado Hall of Fame que armazena os melhores indivíduos entre todas as gerações, no caso do elitismo, só vai armazenas os x melhores indivíduos de cada geração. No caso deste projeto x = 2. O método retorna à população final além dos logs de cada geração. 

### getFitness.py

Este módulo pode ser considerado o mais importante deste projeto pois representa o cerne do algoritmo genético implementado e, portanto, a base onde esse projeto se sustenta. Ele é responsável por definir a função de aptidão na qual os indivíduos serão avaliados. No contexto do projeto a função avalia o quão relevante um determinado subconjunto de variáveis é para a classificação de falhas no Processo Tennessee Eastman. A função recebe como parâmetro a base de dados para treinamento e validação além do indivíduo que se deseja avaliar, ou seja, a lista de booleanos que representa o subconjunto das variáveis selecionadas. Então a base é transformada da seguinte forma: 

Considere um indivíduo I,sendo: 

I  = [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, …, 1] 
 

Considere o indivíduo como uma lista de 52 posições. 

| Indice | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | ... | 51 |
|--------|---|---|---|---|---|---|---|---|---|---|----|-----|----|
| Gene   | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1  | ... | 1  |

Nesse cenário as variáveis cujos índices são 0, 2,3,8,10 e 51 estarão presentes na base de dados transformada, enquanto as variáveis cujos índices são 1,4,5,6,7 e 9 serão descartadas.  

Após essa transformação na base, está é dividida entre bases de treino e validação. As bases então são divididas em 5 para a validação cruzada 5-Fold. Essas partes são utilizadas para treinar o modelo e depois avaliar a sua acurácia. Dependendo dos hiperparâmetros selecionados na definição do GA, a função fitness pode retornar a média das acurácias nas 5 partes, o desvio padrão e/ou o número de variáveis consideradas no indivíduo, normalizado. 

### bestIndividual.py

Esse módulo tem como objetivo selecionar o melhor indivíduo presente no Hall of Fame, e é utilizado somente para obter o melhor indivíduo da população final. A função recebe o Hall of Fame que é retorna pelo módulo do Algoritmo Genético (geneticAlgorithm.py). O que em linhas gerais é o mesmo que selecionar o melhor indivíduo da população final, e consequentemente de todas as gerações. 