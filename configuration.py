##########################
# YOU CAN EDIT THIS FILE #
##########################

# Imports from other libraries
import time

# Imports from this project
import constants

# The window width and height in pixels, for each of the "Live Environment", "Planning Visualisation", and "Model Visualisation" windows.
WINDOW_SIZE = 550

# GRAPHICS_ON determines whether a window will be created and graphics will be shown
# Set it to True when you are developing and debugging your algorithm
# Set it to False when you want to evaluate your full algorithm, so that it is not slowed down by the graphics
# In the coursework assessment, this will be set to False. However, note that the "money" is only deducted when code in robot.py is running, not when the rest of the program is running (e.g. displaying the graphics)
GRAPHICS_ON = True

# The action that is visualised in the "Model Visualisation" window
MODEL_VISUALISATION_ACTION = [0.5 * constants.ROBOT_MAX_ACTION, 0.5 * constants.ROBOT_MAX_ACTION]

# The random seed used for all random numbers, if you want to study specific environments
RANDOM_SEED = int(time.time())
