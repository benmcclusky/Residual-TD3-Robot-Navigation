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

NUM_DEMONSTRATIONS = 4


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


class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []

        self.policy_network = Network()
        self.optimizer = optim.Adam(
            self.policy_network.parameters(), lr=0.1)  # Initial learning rate
        self.criterion = nn.MSELoss()
        self.demonstration_states = []
        self.demonstration_actions = []
        self.num_demonstrations_collected = 0
        self.demo_done = False
        self.model_trained = False
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='Training Epochs', ylabel='Training Loss',
                    title='Loss Curve for Policy Training')

    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        if not self.demo_done:
            if self.num_demonstrations_collected < NUM_DEMONSTRATIONS:
                return 'demo'
            else:
                # Once 10 demonstrations are collected, proceed to train the model
                self.demo_done = True
                print("Demonstrations Complete")
                self.train_on_demonstrations()  # Train the model with collected demonstrations
                self.model_trained = True

        if self.model_trained:
            return 'step'
        return 'step'  # Default to demo if conditions above are not met

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
        pass

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # TODO: This allows you to process or store a demonstration that the robot has received

        self.demonstration_states.extend(demonstration_states)
        self.demonstration_actions.extend(demonstration_actions)
        # Increment the counter for each demonstration processed
        self.num_demonstrations_collected += 1
        print(f"Demonstration: {self.num_demonstrations_collected}")

        path = PathToDraw(demonstration_states, colour=[
                          255, 0, 0], width=2)  # Example color and width
        self.paths_to_draw.append(path)

    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        # Currently, it just predicts the next state according to a simple linear model, although the actual environment dynamics is much more complex
        next_state = state + action
        return next_state

    def train_on_demonstrations(self):
        states = np.array(self.demonstration_states, dtype=np.float32)
        actions = np.array(self.demonstration_actions, dtype=np.float32)
        network_input_data = torch.tensor(states, dtype=torch.float32)
        network_label_data = torch.tensor(actions, dtype=torch.float32)
        input_normalisation_factor = torch.max(network_input_data)
        output_normalisation_factor = torch.max(network_label_data)
        network_input_data /= input_normalisation_factor
        network_label_data /= output_normalisation_factor

        num_epochs = 100
        minibatch_size = 100
        num_minibatches = len(network_input_data) // minibatch_size
        training_losses = []

        for epoch in range(num_epochs):
            if epoch == 20:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.01
            elif epoch == 100:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.001

            permutation = torch.randperm(network_input_data.size()[0])
            epoch_losses = []

            for i in range(0, network_input_data.size()[0], minibatch_size):
                indices = permutation[i:i + minibatch_size]
                batch_x, batch_y = network_input_data[indices], network_label_data[indices]

                self.optimizer.zero_grad()
                outputs = self.policy_network(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)

            # Clear previous plots to avoid overlaying and set logarithmic scale
            self.ax.clear()
            self.ax.plot(training_losses, color='blue')
            self.ax.set(xlabel='Training Epochs', ylabel='Training Loss (log scale)',
                        title='Loss Curve for Policy Training')
            plt.yscale('log')  # Ensure the y-axis is logarithmic
            plt.draw()
            plt.pause(0.01)

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot

    def get_action_from_model(self, state):
        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(
                0)  # Add batch dimension
            action = self.policy_network(state_tensor)
        return action.numpy().squeeze()  # Remove batch dimension and convert to numpy array
