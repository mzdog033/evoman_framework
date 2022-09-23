import os
from numpy.random import randint
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment
import sys
sys.path.insert(0, 'evoman')


def initialize_population(n_population: int, bit_length: int):
    # initiatilize population
    # CODE FROM SGA-SOLUTION
    pop = randint(0, 2, size=(n_population, bit_length))
    return pop


def initialize_environment():
    # initialize environment

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'individual_demo'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[2],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")
    return env

# specialist will run get algorithm.
# get algorithm will run all this stuff with proper parameters (such as ratios) and functions.
