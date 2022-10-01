

# def main_mu_plus_lambda(MU, LAMBDA, CXPB, MUTPB, NGEN):
def lambda_plus_mu(population, children):

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar=lambda x, y: np.all(x == y))
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                             cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)

    return pop, logbook, hof


def main_mu_comma_lambda(MU, LAMBDA, CXPB, MUTPB, NGEN):

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar=lambda x, y: np.all(x == y))
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)

    return pop, logbook, hof
