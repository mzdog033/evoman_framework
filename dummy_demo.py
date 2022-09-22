################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports basic stuff
from evoman.environment import Environment
import time
import numpy as np

# imports framework
import sys
import os
sys.path.insert(0, 'evoman')

# experiment name
experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10  # number of weights for multilayer with 10 hidden neurons

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
ini = time.time()  # sets time marker

# genetic algorithm params
run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons.
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_upper = 1  # upper limit
dom_lower = -1  # lower limit
npop = 100  # population size
gens = 30  # number of generations
mutation = 0.2  # mutation weight
last_best = 0  # last best fitness

# np.random.seed(420)


def simulation(env, x):
    # returns: Fitness, Player life, Enemy life, game run Time
    f, p, e, t = env.play(pcont=x)
    return f


env.play()
