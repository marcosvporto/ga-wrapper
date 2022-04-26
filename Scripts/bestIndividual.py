def bestIndividual(hof, X, y, *args, **kwargs):

    # Get the best individual
    maxAccurcy = 0.0
    for individual in hof:
        if(individual.fitness.values[0] > maxAccurcy):
            maxAccurcy = individual.fitness.values[0]
            _individual = individual

    _individualHeader = [list(X.columns)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader