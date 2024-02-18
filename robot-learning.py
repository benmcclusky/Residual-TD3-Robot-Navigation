#################################
# YOU SHOULD NOT EDIT THIS FILE #
#################################

# Import some external libraries
import time
import numpy as np
import pyglet

# Imports from this project
import constants
import configuration
from environment import Environment
from robot import Robot
from graphics import Graphics


# Set the numpy random seed
np.random.seed(configuration.RANDOM_SEED)
# Create an environment, which is the physical world the robot moves around in
environment = Environment()
state = environment.reset()
# Create a robot, which is the "brain" controlling the robot
robot = Robot(environment.goal_state)
# Create a graphics object, which controls what and where objects should be rendered on the window
graphics = Graphics(environment)
# Create a window on the screen
window = graphics.create_window()
# Set the current mode
mode = 'training'
# Keeping track of what has been bought during training
demos_bought = 0
resets_bought = 0
steps_bought = 0
# Keeping track of the testing
test_init_time = 0
test_best_distance = np.inf
# Keeping track of if any penalty was applied
penalty = False
# Start the training timer
train_init_time = time.time()


# Function to calculate how much money is left
def calculate_remaining_money():
    global train_init_time
    cpu_time_bought = time.time() - train_init_time
    money_spent = demos_bought * constants.COST_PER_DEMO + resets_bought * constants.COST_PER_RESET + steps_bought * constants.COST_PER_STEP + cpu_time_bought * constants.COST_PER_CPU_SECOND
    money_remaining = constants.STARTING_MONEY - money_spent
    return money_remaining


# Function that is called at a regular interval and is used to trigger the robot to plan or execute actions
def update(dt):
    # We are using these global variables
    global mode
    global demos_bought
    global resets_bought
    global steps_bought
    global state
    global train_init_time
    global test_init_time
    global test_best_distance
    global penalty
    # Determine if the robot is training or testing
    if mode == 'training':
        money_remaining = calculate_remaining_money()
        action_type = robot.get_next_action_type(state, money_remaining)
        # Calculate the money remaining again, which will have now reduced if the above function took lots of cpu time
        money_remaining = calculate_remaining_money()
        # If there is no money remaining, move to testing
        if money_remaining < 0:
            # First, check for penalty
            if money_remaining < -1.0:
                print('You have overspent by more than £1! A 10% penalty will be applied to the score.')
                penalty = True
            # Then reset and begin testing
            state = environment.reset()
            mode = 'testing'
            print('Training has finished, moving to testing.')
            test_init_time = time.time()
        elif action_type == 'reset':
            if money_remaining >= constants.COST_PER_RESET:
                state = environment.reset()
                resets_bought += 1
            else:
                print('Insufficient money to buy a reset.')
        elif action_type == 'demo':
            if money_remaining >= constants.COST_PER_DEMO:
                demonstration_states, demonstration_actions = environment.get_demonstration()
                robot.process_demonstration(demonstration_states, demonstration_actions, money_remaining)
                demos_bought += 1
            else:
                print('Insufficient money to buy a demo.')
        elif action_type == 'step':
            if money_remaining >= constants.COST_PER_STEP:
                action = robot.get_next_action_training(state, money_remaining)
                next_state = environment.step(action)
                robot.process_transition(state, action, next_state, money_remaining)
                state = next_state
                steps_bought += 1
        else:
            raise ValueError(f'Invalid value for action_type: {action_type}')
    elif mode == 'testing':
        action = robot.get_next_action_testing(state)
        next_state = environment.step(action)
        distance = np.linalg.norm(next_state - environment.goal_state)
        state = next_state
        test_time = time.time() - test_init_time
        if distance <= constants.TEST_DISTANCE_THRESHOLD:
            print(f'The robot reached the goal! Time: {test_time}.')
            pyglet.app.exit()
        if distance < test_best_distance:
            test_best_distance = distance
        if test_time >= constants.TEST_TIMEOUT:
            print(f'The robot did not reach the goal in time. Best distance: {test_best_distance}.')
            pyglet.app.exit()


# Function that is called at a regular interval to draw the environment and visualisation on the window
@window.event
def on_draw():
    if configuration.GRAPHICS_ON:
        graphics.draw(environment, robot)


# Set the rate at which the update() function is called
# (Note that the rate at which on_draw() is not set by this)
pyglet.clock.schedule_interval(update, 1/constants.UPDATE_RATE)

# Run the application, which will start calling update() and on_draw()
pyglet.app.run()
