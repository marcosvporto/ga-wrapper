from deap import base, algorithms, creator, tools
from getFitness import getFitness
import random
import numpy as np
import libElitism
def geneticAlgorithm(X, y, target, crossover, selection, population, xprob, mutationprob, generations,elitism,real, *args, **kwargs):

    if (target == "a"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    elif (target == "as"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    elif (target == "an"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    elif (target == "asn"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    else:
        raise TypeError('Invalid Targed Value')

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,creator.Individual, toolbox.attr_bool, n=len(X.columns))
    toolbox.register("population", tools.initRepeat, list,toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y, target=target, real=real)


    if (crossover == 1):
        toolbox.register("mate", tools.cxOnePoint)
    elif (crossover == 2):
        toolbox.register("mate", tools.cxTwoPoint)
    elif(crossover > 2 and crossover < 10 ):
        toolbox.register("mate", tools.cxuniform, indpb=crossover/100)
    else:
        raise TypeError('Value has to be between 1 and 9')
            
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)


    if(selection == "r"):
        toolbox.register("select", tools.selRoulette)
    elif(selection == "t"):
        toolbox.register("select", tools.selTournament, tournsize=population)
    elif(selection == "b"):
        toolbox.register("select", tools.selBest)
    else:
        raise TypeError('Invalid Argument Value')


    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(2 if elitism else generations * population)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    if elitism:
        pop, log = libElitism.eaSimpleWithElitism(pop, toolbox, cxpb=xprob, mutpb=mutationprob,
                                    ngen=generations, stats=stats, halloffame=hof,
                                    verbose=True)
    else:    
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=xprob, mutpb=mutationprob,
                                        ngen=generations, stats=stats, halloffame=hof,
                                        verbose=True)
    return hof, pop, log