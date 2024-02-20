##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Imports from this project
import constants
import configuration
from graphics import PathToDraw

NUM_EXPLORATION = 5
NUM_DEMO = 2

TRAIN_AFTER_STEPS = 50

# Random Exploration
RANDOM_STEPS = 200
RANDOM_PATH_LENGTH = 20

# Cross-entropy method
CEM_NUM_PATHS = 30
CEM_PATH_LENGTH = 50
CEM_NUM_ELITES = 5
CEM_NUM_ITERATIONS = 5



class Policy_Network(nn.Module):
    def __init__(self):
        super(Policy_Network, self).__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=50)
        self.layer_2 = nn.Linear(in_features=50, out_features=50)
        self.layer_3 = nn.Linear(in_features=50, out_features=50)
        self.output_layer = nn.Linear(in_features=50, out_features=2)

    def forward(self, input):
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output
    

class Dynamics_Network(torch.nn.Module):
    def __init__(self):
        super(Dynamics_Network, self).__init__()
        self.layer_1 = torch.nn.Linear(
            in_features=4, out_features=50, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(
            in_features=50, out_features=50, dtype=torch.float32)
        self.layer_3 = torch.nn.Linear(
            in_features=50, out_features=50, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(
            in_features=50, out_features=2, dtype=torch.float32)
        
        # Apply custom weight initialization
        self.apply(self.init_weights)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output

    def init_weights(self, m):
        if type(m) == nn.Linear:
            # Initialize weights using a uniform distribution and then clip them
            torch.nn.init.uniform_(m.weight, -1, 1)
            # Clip the weights to the desired range
            with torch.no_grad(): # Ensure no gradients are computed for this operation
                m.weight.clamp_(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)
                
            # Initialize biases to zero
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Expand the buffer if not at capacity
        self.buffer[self.position] = (state, action, next_state)
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)


class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []

        # Behavioural Cloning
        self.last_action = "demo"
        self.policy_network = Policy_Network()
        self.optimizer = optim.Adam(
            self.policy_network.parameters(), lr=0.1)  # Initial learning rate
        self.criterion = nn.MSELoss()
        self.demonstration_states = []
        self.demonstration_actions = []

        self.exploring_flag = False
        self.demo_flag = False
        self.planning_flag = False

        self.num_episodes = 0

        # Replay buffer
        self.memory = ReplayBuffer(10000)  # Example size

        # CEM
        self.planned_actions = []
        self.planned_path = []
        self.planning_actions = []
        self.planning_paths = []
        self.planning_path_rewards = []
        self.planning_mean_actions = []
        self.plan_index = 0
        self.dynamics_model_network = Dynamics_Network()
        self.optimiser = torch.optim.Adam(
            self.dynamics_model_network.parameters(), lr=0.01)




    def get_next_action_type(self, state, money_remaining):
        # Initialize action_type with a default value
        action_type = 'step'

        if (self.num_episodes < NUM_EXPLORATION) and not self.exploring_flag:
            self.exploring_flag = True
            self.random_exploration()

        if NUM_EXPLORATION <= self.num_episodes < (NUM_EXPLORATION + NUM_DEMO) and not self.demo_flag:
            self.demo_flag = True
            action_type = 'demo'

        if (self.num_episodes >= (NUM_EXPLORATION + NUM_DEMO)) and not self.planning_flag:
            self.planning_flag = True
            self.cross_entropy_method_planning()

        if self.plan_index == len(self.planned_actions):
            self.plan_index = 0
            self.num_episodes += 1
            self.exploring_flag = False
            self.demo_flag = False
            self.planning_flag = False
            self.train_dynamics()
            self.train_policy()
            action_type = 'reset'


        # Debugging output
        print(f"Action Type: {action_type}, Money Remaining: {money_remaining}, Steps Taken: {self.plan_index}, Episode: {self.num_episodes}")

        return action_type

    def get_next_action_training(self, state, money_remaining):

        action = self.planned_actions[self.plan_index]  # Assume self.planned_actions is updated to hold only the next action or sequence
        self.plan_index += 1  # Reset plan_index as we will replan after this action

        action = self.add_noise(action)

        return action
    
    def add_noise(self, action):
        # Add some noise to the action if the amount of exploration is still small
        if self.num_episodes < 6:
            noise = np.random.normal(action, 0.5 * constants.ROBOT_MAX_ACTION)
        elif self.num_episodes < 20:
            noise = np.random.normal(action, 0.2 * constants.ROBOT_MAX_ACTION)

        action += noise

        return action


    def get_next_action_testing(self, state):
        return self.get_action_from_model(state)

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        self.memory.push(state, action, next_state)

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):

        self.demonstration_states.extend(demonstration_states)
        self.demonstration_actions.extend(demonstration_actions)

        # self.draw_path(self.demonstration_states, [0, 255, 0], 2)
        path = PathToDraw(demonstration_states, colour=[
                          255, 0, 0], width=2)  # Example color and width
        self.paths_to_draw.append(path)

        for i in range(len(demonstration_states) - 1):  # Assuming sequential data
            state = demonstration_states[i]
            action = demonstration_actions[i]
            # Next state is the subsequent state in the demo
            next_state = demonstration_states[i + 1]
            self.memory.push(state, action, next_state)


    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        
        with torch.no_grad():  # Ensure no gradients are computed
        # Flatten and combine: state and action from (2, 1) each to combined (4,)
            combined_input = np.hstack([state.flatten(), action.flatten()])

            # Convert to tensor and add batch dimension: from (4,) to (1, 4)
            combined_input_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0)

            # Neural network prediction: output shape from (1, 4) to (1, 2)
            predicted_next_state_tensor = self.dynamics_model_network(combined_input_tensor)

            # Convert to numpy and reshape: from (1, 2) back to (2, 1)
            predicted_next_state = predicted_next_state_tensor.numpy().reshape(-1, 1)

        return predicted_next_state

    
    # Function to calculate the reward for a path, in order to evaluate how good the path is
    def compute_reward(self, path):
        reward = -np.linalg.norm(path[-1] - self.goal_state)
        return reward

    def train_policy(self):

        num_epochs = 100
        minibatch_size = 100

        # Extract all states and actions from the memory for normalization factor calculation
        all_transitions = self.memory.buffer
        all_states = np.array([trans[0].reshape(-1) for trans in all_transitions if trans is not None], dtype=np.float32)
        all_actions = np.array([trans[1].reshape(-1) for trans in all_transitions if trans is not None], dtype=np.float32)

        # Calculate normalization factors
        input_normalisation_factor = np.max(np.abs(all_states)) if np.max(np.abs(all_states)) > 0 else 1
        output_normalisation_factor = np.max(np.abs(all_actions)) if np.max(np.abs(all_actions)) > 0 else 1

        for epoch in range(num_epochs):
            epoch_losses = []

            for _ in range(len(self.memory) // minibatch_size):
                transitions = self.memory.sample(minibatch_size)
                if transitions is None:
                    print("Not enough transitions to sample a minibatch.")
                    break

                state_batch = np.array([t[0].reshape(-1) for t in transitions], dtype=np.float32)
                action_batch = np.array([t[1].reshape(-1) for t in transitions], dtype=np.float32)
                next_state_batch = np.array([t[2].reshape(-1) for t in transitions], dtype=np.float32)

                # Normalize data
                states = torch.tensor(state_batch / input_normalisation_factor, dtype=torch.float32)
                actions = torch.tensor(action_batch / output_normalisation_factor, dtype=torch.float32)
                next_states = torch.tensor(next_state_batch / input_normalisation_factor, dtype=torch.float32)

                self.optimizer.zero_grad()
                outputs = self.policy_network(states)
                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            if len(epoch_losses) > 0:
                avg_loss = np.mean(epoch_losses)
                print(f"Policy Network: Epoch {epoch}, Average Loss: {avg_loss}")

        self.model_trained = True
        print("Policy Network: Training completed.")



    def get_action_from_model(self, state):
        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # Add batch dimension
            action = self.policy_network(state_tensor)
        return action.numpy().squeeze()  # Remove batch dimension and convert to numpy array
    

    def random_exploration(self):
        # Create some random actions and simulate their effects to generate a random path
        actions = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, [RANDOM_PATH_LENGTH, 2, 1])
        self.planned_actions = actions


    # Function to do cross-entropy planning
    def cross_entropy_method_planning(self, state):

        self.paths_to_draw.clear() 
        self.planning_actions = np.zeros([CEM_NUM_ITERATIONS,  CEM_NUM_PATHS,  CEM_PATH_LENGTH, 2, 1], dtype=np.float32)
        self.planning_paths = np.zeros([CEM_NUM_ITERATIONS,  CEM_NUM_PATHS,  CEM_PATH_LENGTH, 2, 1], dtype=np.float32)
        self.planning_path_rewards = np.zeros([CEM_NUM_ITERATIONS,  CEM_NUM_PATHS])
        self.planning_mean_actions = np.zeros([CEM_NUM_ITERATIONS,  CEM_PATH_LENGTH, 2, 1], dtype=np.float32)
        
        for iteration_num in range(CEM_NUM_ITERATIONS):

            for path_num in range(CEM_NUM_PATHS):
                planning_state = np.copy(state)

                for step_num in range(CEM_PATH_LENGTH):

                    if iteration_num == 0:
                        action = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, [2, 1])
                    else:
                        action = np.random.normal(best_paths_action_mean[step_num], best_paths_action_std_dev[step_num])

                    self.planning_actions[iteration_num,path_num, step_num] = action

                    next_state = self.dynamics_model(planning_state, action)

                    self.planning_paths[iteration_num, path_num, step_num] = next_state
                    planning_state = next_state 
                    path_reward = self.compute_reward(self.planning_paths[iteration_num, path_num])
                
                self.planning_path_rewards[iteration_num, path_num] = path_reward

                self.draw_path(self.planning_paths[iteration_num, path_num], [255, 255, 255], 1)

            sorted_path_rewards = self.planning_path_rewards[iteration_num].copy()
            sorted_path_costs = np.argsort(sorted_path_rewards)

            indices_best_paths = sorted_path_costs[- CEM_NUM_ELITES:]
            best_paths_action_mean = np.mean(self.planning_actions[iteration_num, indices_best_paths], axis=0)
            best_paths_action_std_dev = np.std(self.planning_actions[iteration_num, indices_best_paths], axis=0)
            self.planning_mean_actions[iteration_num] = best_paths_action_mean


        index_best_path = np.argmax(self.planning_path_rewards[-1])
        self.planned_path = self.planning_paths[-1, index_best_path]
        self.planned_actions = self.planning_actions[-1, index_best_path]
        self.draw_path(self.planned_path, [255, 0, 0], 2)


    def draw_path(self, path, colour, width):
        path_to_draw = PathToDraw(path, colour=colour, width=width)  
        self.paths_to_draw.append(path_to_draw)

    
    def train_dynamics(self):
        num_training_epochs = 0
        training_losses = []
        num_epochs = 100
        minibatch_size = 100
        loss_function = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            training_epoch_losses = []

            num_training_data = len(self.memory)
            num_minibatches = max(int(num_training_data / minibatch_size), 1)

            for _ in range(num_minibatches):
                transitions = self.memory.sample(minibatch_size)
                if transitions is None:
                    continue

                # Assuming each state, action, and next state is initially (2, 1) numpy array
                state_batch = np.array([t[0].flatten() for t in transitions])  # Flatten to ensure (2,) shape
                action_batch = np.array([t[1].flatten() for t in transitions])
                next_state_batch = np.array([t[2].flatten() for t in transitions])

                # Convert these numpy arrays to PyTorch tensors and reshape
                state_batch = torch.tensor(state_batch, dtype=torch.float32).view(minibatch_size, -1)
                action_batch = torch.tensor(action_batch, dtype=torch.float32).view(minibatch_size, -1)
                next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).view(minibatch_size, -1)

                combined_input_batch = torch.cat((state_batch, action_batch), dim=1)

                self.optimiser.zero_grad()

                predicted_next_state_batch = self.dynamics_model_network(combined_input_batch)

                training_loss = loss_function(predicted_next_state_batch, next_state_batch)
                training_loss.backward()
                self.optimiser.step()

                training_epoch_losses.append(training_loss.item())

            if training_epoch_losses:
                training_epoch_loss = np.mean(training_epoch_losses)
                training_losses.append(training_epoch_loss)
                num_training_epochs += 1
                print(f'Dynamics Network: Epoch {num_training_epochs}: Training Loss = {training_epoch_loss:.4f}')

        print("Dynamics Network: Training completed.")
