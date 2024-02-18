##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np

# Imports from this project
import constants
import configuration
from graphics import PathToDraw


class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []

    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'
        if False:
            return 'demo'
        if False:
            return 'reset'
        if True:
            return 'step'

    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        random_action = np.random.uniform(
            [-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION], 2)
        return random_action

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        random_action = np.random.uniform(
            [-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION], 2)
        return random_action

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        # TODO: This allows you to process or store a transition that the robot has experienced in the environment
        # Currently, nothing happens
        pass

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # TODO: This allows you to process or store a demonstration that the robot has received
        # Currently, nothing happens
        pass

    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        # Currently, it just predicts the next state according to a simple linear model, although the actual environment dynamics is much more complex
        next_state = state + action
        return next_state
