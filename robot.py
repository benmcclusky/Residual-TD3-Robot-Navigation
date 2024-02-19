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

NUM_DEMONSTRATIONS = 3
TRAIN_AFTER_STEPS = 50


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Adjust in_features based on your state dimension
        self.layer_1 = nn.Linear(in_features=2, out_features=50)
        self.layer_2 = nn.Linear(in_features=50, out_features=50)
        self.layer_3 = nn.Linear(in_features=50, out_features=50)
        # Adjust out_features based on your action dimension
        self.output_layer = nn.Linear(in_features=50, out_features=2)

    def forward(self, input):
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


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
        self.last_action = "demo"
        self.policy_network = Network()
        self.optimizer = optim.Adam(
            self.policy_network.parameters(), lr=0.1)  # Initial learning rate
        self.criterion = nn.MSELoss()
        self.demonstration_states = []
        self.demonstration_actions = []

        self.memory = ReplayBuffer(10000)  # Example size
        self.demo_counter = 0
        self.steps_taken = 0
        self.model_trained = False

    def get_next_action_type(self, state, money_remaining):
        # Initialize action_type with a default value
        action_type = 'step'

        if self.demo_counter < NUM_DEMONSTRATIONS:
            if self.demo_counter == 0:
                self.demo_counter += 1
                print(f"Demonstration: {self.demo_counter}")
                action_type = 'demo'

            elif self.last_action == 'demo':
                self.train_model()
                self.steps_taken = 0
                action_type = 'step'

            elif self.last_action == 'step':
                if self.steps_taken == TRAIN_AFTER_STEPS:
                    self.train_model()
                    self.steps_taken = 0
                    action_type = 'reset'

                else:
                    self.steps_taken += 1
                    action_type = 'step'

            elif self.last_action == 'reset':
                self.demo_counter += 1
                action_type = 'demo'

        else:
            if (self.steps_taken == TRAIN_AFTER_STEPS) and (self.steps_taken > 0) and (money_remaining > 7):
                self.train_model()
                self.steps_taken = 0
                action_type = 'reset'

            else:
                self.steps_taken += 1
                action_type = 'step'

        # Debugging output
        print(f"Action Type: {action_type}, Money Remaining: {money_remaining}, Last Action: {self.last_action}, Steps Taken: {self.steps_taken}, Demonstrations: {self.demo_counter}")

        self.last_action = action_type

        return action_type

    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'

        if self.model_trained:
            return self.get_action_from_model(state)
        else:
            # Fallback to random action if model is not yet trained
            return np.random.uniform([-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION], 2)

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        return self.get_action_from_model(state)

    # Function that processes a transition

    def process_transition(self, state, action, next_state, money_remaining):
        # TODO: This allows you to process or store a transition that the robot has experienced in the environment
        # Currently, nothing happens
        self.memory.push(state, action, next_state)

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # TODO: This allows you to process or store a demonstration that the robot has received

        self.demonstration_states.extend(demonstration_states)
        self.demonstration_actions.extend(demonstration_actions)

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
        # Currently, it just predicts the next state according to a simple linear model, although the actual environment dynamics is much more complex
        next_state = state + action
        return next_state

    def train_model(self):
        if len(self.memory) == 0:
            print("Memory buffer is empty. Cannot train model.")
            return

        num_epochs = 100
        minibatch_size = 100

        # Extract all states and actions from the memory for normalization factor calculation
        all_transitions = self.memory.buffer
        all_states = np.array(
            [trans[0] for trans in all_transitions if trans is not None], dtype=np.float32)
        all_actions = np.array(
            [trans[1] for trans in all_transitions if trans is not None], dtype=np.float32)

        # Calculate normalization factors
        input_normalisation_factor = np.max(
            np.abs(all_states)) if np.max(np.abs(all_states)) > 0 else 1
        output_normalisation_factor = np.max(
            np.abs(all_actions)) if np.max(np.abs(all_actions)) > 0 else 1

        for epoch in range(num_epochs):
            epoch_losses = []

            for _ in range(len(self.memory) // minibatch_size):
                transitions = self.memory.sample(minibatch_size)
                if transitions is None:
                    print("Not enough transitions to sample a minibatch.")
                    break

                batch = list(zip(*transitions))
                states = np.array(batch[0], dtype=np.float32)
                actions = np.array(batch[1], dtype=np.float32)
                next_states = np.array(batch[2], dtype=np.float32)

                # Normalize data
                states = torch.tensor(
                    states, dtype=torch.float32) / input_normalisation_factor
                actions = torch.tensor(
                    actions, dtype=torch.float32) / output_normalisation_factor
                next_states = torch.tensor(
                    next_states, dtype=torch.float32) / input_normalisation_factor

                self.optimizer.zero_grad()
                outputs = self.policy_network(states)
                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            if len(epoch_losses) > 0:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch}, Average Loss: {avg_loss}")

        self.model_trained = True
        print("Training completed.")

    def get_action_from_model(self, state):
        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(
                0)  # Add batch dimension
            action = self.policy_network(state_tensor)
        return action.numpy().squeeze()  # Remove batch dimension and convert to numpy array
