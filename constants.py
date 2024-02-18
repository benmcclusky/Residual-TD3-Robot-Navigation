#################################
# YOU SHOULD NOT EDIT THIS FILE #
#################################

# Size of the world, i.e. the coordinate frame in which the robot's state is defined
WORLD_SIZE = 100

# Attributes for the robot
ROBOT_RADIUS = 1.5
ROBOT_COLOUR = (50, 50, 255, 255)

# Attributes for the goal
GOAL_RADIUS = 2
GOAL_COLOUR = (100, 200, 100, 255)

# Attributes for the boundary
BOUNDARY_WIDTH = 0.5
BOUNDARY_COLOUR = (100, 100, 100, 255)

# Attributes for the environment background
ENVIRONMENT_COLOUR = (0, 0, 0, 255)
BACKGROUND_COLOUR_GL = (0, 0, 0, 1)

# The size of the initial region
INIT_REGION_SIZE = 25

# Colour of the robot's initial region
ROBOT_INIT_REGION_COLOUR = (150, 150, 255, 100)

# The number of times per second the state of the environment is updated
UPDATE_RATE = 10

# The maximum action magnitude the robot can execute in each action dimension
ROBOT_MAX_ACTION = 5

# For generating the demonstrations
DEMOS_CEM_NUM_ITERATIONS = 4
DEMOS_CEM_NUM_PATHS = 100
DEMOS_CEM_PATH_LENGTH = 200
DEMOS_CEM_NUM_ELITES = 10

# The amount of money available at the start
STARTING_MONEY = 100
COST_PER_STEP = 0.01
COST_PER_CPU_SECOND = 0.03
COST_PER_DEMO = 20
COST_PER_RESET = 5

# The distance to the goal the qualifies as the task being solved
TEST_DISTANCE_THRESHOLD = 5

# The number of steps at which testing finishes, in the case when the robot does not reach the goal
TEST_TIMEOUT = 100
