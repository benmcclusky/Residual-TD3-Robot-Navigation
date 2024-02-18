#################################
# YOU SHOULD NOT EDIT THIS FILE #
#################################

# Import some external libraries
import numpy as np
import pyglet
from scipy import ndimage

# Imports from this project
import constants
import configuration


# Function to convert from world space to window space
def world_to_window(world_pos):
    window_pos = int((world_pos / constants.WORLD_SIZE) * configuration.WINDOW_SIZE)
    return window_pos


# Class to store a line which will be drawn on the window
class PathToDraw:
    def __init__(self, path, colour, width):
        self.path = path
        # Clip the colour so that it is between (0, 0, 0) and (255, 255, 255)
        colour = [max(0, min(c, 255)) for c in colour]
        # Make sure the colour is ints not floats
        self.line_colour = [int(c) for c in colour]
        self.line_colour.append(255)
        self.line_width = int(width)


# The Graphics class is used to decide what and where to draw on the window
class Graphics:

    def __init__(self, environment):
        self.window = None
        self.middle_space = int(0.01 * configuration.WINDOW_SIZE)
        self.top_space = int(0.05 * configuration.WINDOW_SIZE)
        self.visualisation_iteration_num = 0
        self.num_visualisation_steps = 0
        # Create the environment's background image for the speed component of the dynamics
        transposed_image_speed = np.transpose(1 - environment.dynamics_speed)
        zoom_factor_speed = configuration.WINDOW_SIZE / constants.WORLD_SIZE
        scaled_image_speed = ndimage.zoom(transposed_image_speed, zoom_factor_speed, order=3)
        uint_image_speed = (scaled_image_speed * 255).astype(np.uint8)
        rgb_image_speed = np.stack([uint_image_speed] * 3, axis=-1)
        raw_image_data_speed = rgb_image_speed.flatten().tobytes()
        self.environment_dynamics_speed_image = pyglet.image.ImageData(configuration.WINDOW_SIZE, configuration.WINDOW_SIZE, 'RGB', raw_image_data_speed)
        # Create the environment's background image for the angle component of the dynamics
        transposed_image_angle = np.transpose(environment.dynamics_angle)
        zoom_factor_angle = configuration.WINDOW_SIZE / constants.WORLD_SIZE
        scaled_image_angle = ndimage.zoom(transposed_image_angle, zoom_factor_angle, order=3)
        uint_image_angle = (scaled_image_angle * 255).astype(np.uint8)
        rgb_image_angle = np.stack([uint_image_angle] * 3, axis=-1)
        raw_image_data_angle = rgb_image_angle.flatten().tobytes()
        self.environment_dynamics_angle_image = pyglet.image.ImageData(configuration.WINDOW_SIZE, configuration.WINDOW_SIZE, 'RGB', raw_image_data_angle)

    # Function to create a new pyglet window
    def create_window(self):
        # Create a new window
        window_width = 3 * configuration.WINDOW_SIZE + 2 * self.middle_space
        window_height = configuration.WINDOW_SIZE + self.top_space
        self.window = pyglet.window.Window(width=window_width, height=window_height)
        # Set the background colour
        pyglet.gl.glClearColor(*constants.BACKGROUND_COLOUR_GL)
        self.window.clear()
        # Return the window to the main script
        return self.window

    # Function to draw the environment and visualisations to the screen
    def draw(self, environment, robot):
        # Clear the entire window
        self.window.clear()
        # Draw the titles
        self.draw_titles()
        # LIVE ENVIRONMENT (left)
        x_left = 0
        # Draw the environment
        self.draw_environment_image()
        self.draw_goal(environment.goal_state, x_left=x_left)
        # Draw the robot
        self.draw_robot(environment.robot_state, x_left=x_left)
        self.draw_robot_init_region(environment.robot_init_region, x_left)
        # Draw the boundary
        self.draw_boundary(x_left=x_left)
        # PLANNING VISUALISATION (middle)
        x_left += configuration.WINDOW_SIZE + self.middle_space
        # Draw the environment
        self.draw_goal(environment.goal_state, x_left=x_left)
        # Draw the robot
        self.draw_robot(environment.robot_state, x_left=x_left)
        self.draw_robot_init_region(environment.robot_init_region, x_left)
        # Draw the user's own paths
        self.draw_planning_visualisation(robot.paths_to_draw, x_left=x_left)
        # Draw the boundary
        self.draw_boundary(x_left=x_left)
        # MODEL VISUALISATION (right)
        x_left += configuration.WINDOW_SIZE + self.middle_space
        # Draw the planning visualisations
        self.draw_model_visualisation(environment, robot, x_left=x_left)
        # Draw the boundary
        self.draw_boundary(x_left=x_left)

    # Function to draw the titles for the left-hand and right-hand sides
    def draw_titles(self):
        font_size = configuration.WINDOW_SIZE * 0.03
        text_x = int(0.5 * configuration.WINDOW_SIZE)
        text_y = configuration.WINDOW_SIZE
        label = pyglet.text.Label('Live Environment', font_name='Arial', font_size=font_size, x=text_x, y=text_y, anchor_x='center', anchor_y='bottom')
        label.draw()
        x_left = configuration.WINDOW_SIZE + self.middle_space
        text_x = int(x_left + 0.5 * configuration.WINDOW_SIZE)
        text_y = configuration.WINDOW_SIZE
        label = pyglet.text.Label('Your Paths', font_name='Arial', font_size=font_size, x=text_x, y=text_y, anchor_x='center', anchor_y='bottom')
        label.draw()
        x_left += configuration.WINDOW_SIZE + self.middle_space
        text_x = int(x_left + 0.5 * configuration.WINDOW_SIZE)
        text_y = configuration.WINDOW_SIZE
        label = pyglet.text.Label('Model Visualisation', font_name='Arial', font_size=font_size, x=text_x, y=text_y, anchor_x='center', anchor_y='bottom')
        label.draw()

    def draw_environment_image(self):
        self.environment_dynamics_speed_image.blit(0, 0)

    # Function to draw the robot's current position
    def draw_robot(self, robot_state, x_left):
        robot_x_window = x_left + world_to_window(robot_state[0])
        robot_y_window = world_to_window(robot_state[1])
        robot_radius = world_to_window(constants.ROBOT_RADIUS)
        robot_colour = constants.ROBOT_COLOUR
        robot_shape = pyglet.shapes.Circle(x=robot_x_window, y=robot_y_window, radius=robot_radius, color=robot_colour)
        robot_shape.draw()

    def draw_robot_init_region(self, robot_init_region, x_left):
        x = x_left + world_to_window(robot_init_region[0])
        y = world_to_window(robot_init_region[2])
        width = world_to_window(robot_init_region[1] - robot_init_region[0])
        height = world_to_window(robot_init_region[3] - robot_init_region[2])
        robot_init_region_shape = pyglet.shapes.Rectangle(x, y, width, height, constants.ROBOT_INIT_REGION_COLOUR)
        robot_init_region_shape.draw()

    # Function to draw the goal
    def draw_goal(self, goal_state, x_left):
        goal_x_window = x_left + world_to_window(goal_state[0])
        goal_y_window = world_to_window(goal_state[1])
        goal_radius = world_to_window(constants.GOAL_RADIUS)
        goal_colour = constants.GOAL_COLOUR
        goal_shape = pyglet.shapes.Circle(x=goal_x_window, y=goal_y_window, radius=goal_radius, color=goal_colour)
        goal_shape.draw()

    # Function to draw the environment's boundary
    def draw_boundary(self, x_left):
        # Left boundary
        x = x_left
        y = 0
        width = world_to_window(constants.BOUNDARY_WIDTH)
        height = world_to_window(100)
        left_boundary_shape = pyglet.shapes.Rectangle(x, y, width, height, constants.BOUNDARY_COLOUR)
        left_boundary_shape.draw()
        # Right boundary
        x = x_left + world_to_window(100 - constants.BOUNDARY_WIDTH)
        y = 0
        width = world_to_window(constants.BOUNDARY_WIDTH)
        height = world_to_window(100)
        left_boundary_shape = pyglet.shapes.Rectangle(x, y, width, height, constants.BOUNDARY_COLOUR)
        left_boundary_shape.draw()
        # Top boundary
        x = x_left
        y = world_to_window(100 - constants.BOUNDARY_WIDTH)
        width = world_to_window(100)
        height = world_to_window(constants.BOUNDARY_WIDTH)
        left_boundary_shape = pyglet.shapes.Rectangle(x, y, width, height, constants.BOUNDARY_COLOUR)
        left_boundary_shape.draw()
        # Bottom boundary
        x = x_left
        y = 0
        width = world_to_window(100)
        height = world_to_window(constants.BOUNDARY_WIDTH)
        left_boundary_shape = pyglet.shapes.Rectangle(x, y, width, height, constants.BOUNDARY_COLOUR)
        left_boundary_shape.draw()

    # Function to draw the planning visualisations
    def draw_planning_visualisation(self, paths_to_draw, x_left):
        shape_list = []
        batch = pyglet.graphics.Batch()
        for path_to_draw in paths_to_draw:
            path = path_to_draw.path
            prev_x = x_left + world_to_window(path[0, 0])
            prev_y = world_to_window(path[0, 1])
            num_steps = len(path)
            for step_num in range(1, num_steps):
                # Check the state is within the environment limit
                if 0 <= path[step_num, 0] <= constants.WORLD_SIZE and 0 <= path[step_num, 1] <= constants.WORLD_SIZE:
                    # Calculate the position
                    next_x = x_left + world_to_window(path[step_num, 0])
                    next_y = world_to_window(path[step_num, 1])
                    # Draw the line
                    line_shape = pyglet.shapes.Line(x=int(prev_x), y=int(prev_y), x2=int(next_x), y2=int(next_y), width=path_to_draw.line_width, color=path_to_draw.line_colour, batch=batch)
                    shape_list.append(line_shape)
                    # Set the new prev position
                    prev_x = next_x
                    prev_y = next_y
        # Draw all shapes
        batch.draw()

    def draw_model_visualisation(self, environment, robot, x_left):
        batch = pyglet.graphics.Batch()
        shape_list = []
        # Loop over all the cells
        num_cells_down = num_cells_across = 10
        for row in range(num_cells_down):
            world_y = ((row + 0.5) / num_cells_down) * constants.WORLD_SIZE
            window_y = world_to_window(world_y)
            for col in range(10):
                world_x = ((col + 0.5) / num_cells_across) * constants.WORLD_SIZE
                window_x = x_left + world_to_window(world_x)
                # Create a dot representing the robot's current state
                dot_shape = pyglet.shapes.Circle(x=window_x, y=window_y, radius=5, color=constants.ROBOT_COLOUR, batch=batch)
                shape_list.append(dot_shape)
                # Predict the next state using the environment's true dynamics model
                current_state = [world_x, world_y]
                action = [configuration.MODEL_VISUALISATION_ACTION[0], configuration.MODEL_VISUALISATION_ACTION[1]]
                predicted_next_state = environment.dynamics(current_state, action)
                prediction_window_x = x_left + world_to_window(predicted_next_state[0])
                prediction_window_y = world_to_window(predicted_next_state[1])
                # Create a line representing this next state
                line_shape = pyglet.shapes.Line(x=int(window_x), y=int(window_y), x2=int(prediction_window_x), y2=int(prediction_window_y), width=3, color=(100, 100, 100, 255), batch=batch)
                shape_list.append(line_shape)
                # Predict the next state using the robot's learned dynamics model
                current_state = np.array([world_x, world_y])
                action = np.array([configuration.MODEL_VISUALISATION_ACTION[0], configuration.MODEL_VISUALISATION_ACTION[1]])
                predicted_next_state = robot.dynamics_model(current_state, action)
                prediction_window_x = x_left + world_to_window(predicted_next_state[0])
                prediction_window_y = world_to_window(predicted_next_state[1])
                # Create a line representing this next state
                line_shape = pyglet.shapes.Line(x=int(window_x), y=int(window_y), x2=int(prediction_window_x), y2=int(prediction_window_y), width=1, color=(255, 255, 255, 255), batch=batch)
                shape_list.append(line_shape)
        # Draw the batch of shapes
        batch.draw()
