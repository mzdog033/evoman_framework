import os
from numpy.random import uniform
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment
import sys
sys.path.insert(0, 'evoman')


def initialize_environment(enemies_list):
    # headless = True
    # if headless:
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'multimode_demo'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    env = Environment(experiment_name=experiment_name,
                      enemies=enemies_list,
                      playermode="ai",
                      multiplemode='yes',
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")
    return env
