from generateClassifier import generateClassifier 

acc_score, n_features, header = generateClassifier( real=True, 
                                                    target = "an", 
                                                    selection="r", 
                                                    crossover=1, 
                                                    population=20, 
                                                    generations=50, 
                                                    mutationprob=0.05, 
                                                    xprob=0.9,
                                                    elitism = True )
