from algorithm import initialize_population, initialize_environment


def runner():
    # initlaize pop
    # run loops
    # inside loop, do fitness, selection, crossover, mutation, and create a new generation
    # add generation to list? keep track of generation number
    # keep track of fitnesses. save the best ones each generation

    init_pop = initialize_population(10, 5)
    init_env = initialize_environment()
    print('pop', init_pop)
    # print('env', init_env)

    # for i in init_pop:
    #     init_env.play(pcont=i)  # error..
    init_env.state_to_log()
    print('solutions', init_env.get_solutions())


runner()
