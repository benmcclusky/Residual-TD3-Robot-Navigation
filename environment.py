#################################
# YOU SHOULD NOT EDIT THIS FILE #
#################################

# Import some external libraries
import numpy as np
from perlin_noise import PerlinNoise

# Imports from this project
import constants
import configuration


class Environment:

    def __init__(self):
        self.robot_state = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_init_region = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_state = np.array([0.0, 0.0], dtype=np.float32)
        self.dynamics_speed = np.zeros([constants.WORLD_SIZE, constants.WORLD_SIZE], dtype=np.float32)
        self.dynamics_angle = np.zeros([constants.WORLD_SIZE, constants.WORLD_SIZE], dtype=np.float32)
        # Set random initial region and goal
        self.set_init_and_goal()
        # Set random dynamics
        self.set_dynamics()

    # Set the state of the goal, and the region where the robot initially spawns from
    def set_init_and_goal(self):
        r = np.random.choice([0, 1, 2, 3])
        if r == 0:
            init_left = 0
            init_right = constants.INIT_REGION_SIZE
            init_bottom = np.random.uniform(0, constants.WORLD_SIZE - constants.INIT_REGION_SIZE)
            init_top = init_bottom + constants.INIT_REGION_SIZE
        elif r == 1:
            init_left = np.random.uniform(0, constants.WORLD_SIZE - constants.INIT_REGION_SIZE)
            init_right = init_left + constants.INIT_REGION_SIZE
            init_bottom = constants.WORLD_SIZE - constants.INIT_REGION_SIZE
            init_top = constants.WORLD_SIZE
        elif r == 2:
            init_left = constants.WORLD_SIZE - constants.INIT_REGION_SIZE
            init_right = constants.WORLD_SIZE
            init_bottom = np.random.uniform(0, constants.WORLD_SIZE - constants.INIT_REGION_SIZE)
            init_top = init_bottom + constants.INIT_REGION_SIZE
        elif r == 3:
            init_left = np.random.uniform(0, constants.WORLD_SIZE - constants.INIT_REGION_SIZE)
            init_right = init_left + constants.INIT_REGION_SIZE
            init_bottom = 0
            init_top = constants.INIT_REGION_SIZE
        distance = 0
        init_mid = np.array([0.5 * (init_left + init_right), 0.5 * (init_bottom + init_top)])
        while distance < 90:
            random_goal = np.random.uniform(5, constants.WORLD_SIZE - 5, 2)
            distance = np.linalg.norm(random_goal - init_mid)
        self.goal_state = random_goal
        self.robot_init_region = np.array([init_left, init_right, init_bottom, init_top])

    # Set the random dynamics for this environment
    def set_dynamics(self):
        # SPEED
        # Get some random perlin noise functions
        noise_1 = PerlinNoise(octaves=5, seed=configuration.RANDOM_SEED)
        noise_2 = PerlinNoise(octaves=10, seed=configuration.RANDOM_SEED)
        noise_3 = PerlinNoise(octaves=20, seed=configuration.RANDOM_SEED)
        # Get a random stretch factor
        num_cells_across = constants.WORLD_SIZE
        cells = np.zeros([num_cells_across, num_cells_across], dtype=np.float32)
        # Loop over all the cells in the environment
        for col in range(num_cells_across):
            for row in range(num_cells_across):
                # Calculate the cell value as the weighted average of the perlin noise functions
                cell_value = noise_1([col / num_cells_across, row / num_cells_across])
                cell_value += 0.5 * noise_2([col / num_cells_across, row / num_cells_across])
                cell_value += 0.25 * noise_3([col / num_cells_across, row / num_cells_across])
                cells[col, row] = cell_value
        # Normalise the cells to between 0 and 1
        min_cell = np.min(cells)
        max_cell = np.max(cells)
        cells_normalised = (cells - min_cell) / (max_cell - min_cell)
        # Apply a sigmoid function to stretch out the values
        stretch_factor = 10
        cells_stretched = 1 / (1 + np.exp(-stretch_factor * (cells_normalised - 0.5)))
        self.dynamics_speed = cells_stretched
        # Set the angle
        noise = PerlinNoise(octaves=5, seed=configuration.RANDOM_SEED)
        num_cells_across = constants.WORLD_SIZE
        cells = np.zeros([num_cells_across, num_cells_across], dtype=np.float32)
        for col in range(num_cells_across):
            for row in range(num_cells_across):
                cell_value = noise([col / num_cells_across, row / num_cells_across])
                cells[col, row] = cell_value
        min_cell = np.min(cells)
        max_cell = np.max(cells)
        cells_normalised = (cells - min_cell) / (max_cell - min_cell)
        self.dynamics_angle = cells_normalised

    # The true environment dynamics
    def dynamics(self, state, action):
        # Check that the action is within the robot's maximum action limit
        action = np.clip(action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)
        # Calculate the magnitude and angle of this action
        action_magnitude = np.linalg.norm(action)
        action_angle = np.arctan2(action[1], action[0])
        # Get the rotation component of the dynamics
        cell_x = int(state[0])
        cell_y = int(state[1])
        rotation = self.dynamics_angle[cell_x, cell_y] * 2 * np.pi
        # Rotate the direction of the action by this rotation angle
        rotated_action_angle = action_angle + rotation
        # Get the speed component of the dynamics
        speed = self.dynamics_speed[cell_x, cell_y]
        # Compute the next state
        next_state_x = state[0] + speed * action_magnitude * np.cos(rotated_action_angle)
        next_state_y = state[1] + speed * action_magnitude * np.sin(rotated_action_angle)
        next_state = np.array([next_state_x, next_state_y])
        # Check for collision with the boundary
        next_state = np.clip(next_state, 0, constants.WORLD_SIZE - 1.0001)
        # Return this next state
        return next_state

    # Take one timestep in the environment
    def step(self, action):
        next_state = self.dynamics(self.robot_state, action)
        # Check to see if the robot has hit the boundary
        if 0 <= next_state[0] < constants.WORLD_SIZE and 0 <= next_state[1] < constants.WORLD_SIZE:
            self.robot_state = next_state
        return self.robot_state

    # Reset the environment back to its initial state
    def reset(self):
        self.robot_state = self.get_random_robot_init_state()
        return self.robot_state

    # Choose a random state within the initial region
    def get_random_robot_init_state(self):
        state = np.random.uniform([self.robot_init_region[0], self.robot_init_region[2]], [self.robot_init_region[1], self.robot_init_region[3]], 2)
        return state

    # Get a demonstration by using cross-entropy method planning
    def get_demonstration(self):
        # planning_actions is the full set of actions that are sampled
        planning_actions = np.zeros([constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_NUM_PATHS, constants.DEMOS_CEM_PATH_LENGTH, 2], dtype=np.float32)
        # planning_paths is the full set of paths (one path is a sequence of states) that are evaluated
        planning_paths = np.zeros([constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_NUM_PATHS, constants.DEMOS_CEM_PATH_LENGTH + 1, 2], dtype=np.float32)
        # planning_path_rewards is the full set of path rewards that are calculated
        planning_path_rewards = np.zeros([constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_NUM_PATHS])
        # planning_mean_actions is the full set of mean action sequences that are calculated at the end of each iteration (one sequence per iteration)
        planning_mean_actions = np.zeros([constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_PATH_LENGTH, 2], dtype=np.float32)
        # Loop over the iterations
        robot_current_state = self.get_random_robot_init_state()
        for iteration_num in range(constants.DEMOS_CEM_NUM_ITERATIONS):
            for path_num in range(constants.DEMOS_CEM_NUM_PATHS):
                planning_state = np.copy(robot_current_state)
                planning_paths[iteration_num, path_num, 0] = planning_state
                for step_num in range(constants.DEMOS_CEM_PATH_LENGTH):
                    if iteration_num == 0:
                        action = np.random.choice([-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION], 2)
                    else:
                        action = np.random.normal(best_paths_action_mean[step_num], best_paths_action_std_dev[step_num])
                    planning_actions[iteration_num, path_num, step_num] = action
                    next_state = self.dynamics(planning_state, action)
                    planning_paths[iteration_num, path_num, step_num + 1] = next_state
                    planning_state = next_state
                path_reward = self.compute_reward(planning_paths[iteration_num, path_num])
                planning_path_rewards[iteration_num, path_num] = path_reward
            sorted_path_rewards = planning_path_rewards[iteration_num].copy()
            sorted_path_costs = np.argsort(sorted_path_rewards)
            indices_best_paths = sorted_path_costs[-constants.DEMOS_CEM_NUM_ELITES:]
            best_paths_action_mean = np.mean(planning_actions[iteration_num, indices_best_paths], axis=0)
            best_paths_action_std_dev = np.std(planning_actions[iteration_num, indices_best_paths], axis=0)
            planning_mean_actions[iteration_num] = best_paths_action_mean
        # Calculate the index of the best path
        index_best_path = np.argmax(planning_path_rewards[-1])
        # Set the planned path (i.e. the best path) to be the path whose index is index_best_path (we remove the last state because this is not associated with an action)
        demonstration_states = planning_paths[-1, index_best_path, 0: constants.DEMOS_CEM_PATH_LENGTH]
        # Set the planned actions (i.e. the best action sequence) to be the action sequence whose index is index_best_path
        demonstration_actions = planning_actions[-1, index_best_path]
        # Return the demonstration states and actions
        return demonstration_states, demonstration_actions

    # The reward function (this is necessary in order to generate the demonstrations)
    def compute_reward(self, path):
        return -np.linalg.norm(path[-1] - self.goal_state)
