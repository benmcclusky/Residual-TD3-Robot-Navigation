##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.spatial import distance
import copy

# Imports from this project
import constants
import configuration
from graphics import PathToDraw

# Demo
NUM_DEMO = 2

# Residual Actor Critic 
PATH_LENGTH = 100

BASELINE_LR = 0.01
BASELINE_EPOCHS = 50
BASELINE_BATCH = 50

ACTOR_LR = 0.01
CRITIC_LR = 0.02

# Need change to TD
DDPG_EPOCHS = 100
DDPG_BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.001

INITIAL_NOISE = 0.2  # Initial noise scale
NOISE_DECAY = 0.75  # Decay rate for noise

STUCK_THRESHOLD = 0.2
STUCK_STEPS = 10


class Baseline_Policy_Network(nn.Module):
    def __init__(self, dropout_rate=0.5):  # Added a parameter to specify the dropout rate, with a default of 0.5
        super(Baseline_Policy_Network, self).__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=50)
        self.dropout_1 = nn.Dropout(dropout_rate)  # Dropout layer after the first layer
        self.layer_2 = nn.Linear(in_features=50, out_features=50)
        self.dropout_2 = nn.Dropout(dropout_rate)  # Dropout layer after the second layer
        self.layer_3 = nn.Linear(in_features=50, out_features=50)
        self.dropout_3 = nn.Dropout(dropout_rate)  # Dropout layer after the third layer
        self.output_layer = nn.Linear(in_features=50, out_features=2)

        # Apply custom weight initialization
        self.apply(self.init_weights)

    def forward(self, input):
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_1_output = self.dropout_1(layer_1_output)  # Apply dropout after activation
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_2_output = self.dropout_2(layer_2_output)  # Apply dropout after activation
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        layer_3_output = self.dropout_3(layer_3_output)  # Apply dropout after activation
        output = self.output_layer(layer_3_output)
        return output
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization for the weights
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            # Zero initialization for the biases
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

class Residual_Actor_Network(nn.Module):
    def __init__(self):
        super(Residual_Actor_Network, self).__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=200)
        self.layer_2 = nn.Linear(in_features=200, out_features=200)
        self.layer_3 = nn.Linear(in_features=200, out_features=200)
        self.output_layer = nn.Linear(in_features=200, out_features=2)

        # Apply custom weight initialization
        self.apply(self.init_weights)

    def forward(self, input):
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization for the weights
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            # Zero initialization for the biases
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Residual_Critic_Network(nn.Module):
    def __init__(self):
        super(Residual_Critic_Network, self).__init__()
        self.layer_1 = nn.Linear(in_features=4, out_features=200)
        self.layer_2 = nn.Linear(in_features=200, out_features=200)
        self.layer_3 = nn.Linear(in_features=200, out_features=200)
        self.output_layer = nn.Linear(in_features=200, out_features=1)

        # Apply custom weight initialization
        self.apply(self.init_weights)

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization for the weights
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            # Zero initialization for the biases
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        samples = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in samples])
        
        # Convert lists to numpy arrays
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)



class Robot:
    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []

        # Behavioural Cloning
        self.baseline_network = Baseline_Policy_Network()
        self.optimizer = optim.Adam(self.baseline_network.parameters(), lr=BASELINE_LR)
        self.criterion = nn.MSELoss()
        self.demonstration_states = []
        self.demonstration_actions = []

        self.demo_flag = False
        self.num_episodes = 0
        self.current_noise_scale = INITIAL_NOISE

        # Replay buffer
        self.memory = ReplayBuffer(10000)

        # Planning
        self.plan_index = 0

        # Normalisation
        self.state_mean = np.array([50.0, 50.0])
        self.state_std = np.array([29, 29])
        self.action_mean = np.array([0.0, 0.0])
        self.action_std = np.array([2.5, 2.5])

        # TD3 Actor / Critic
        self.residual_actor_network = Residual_Actor_Network()
        # Initialize twin critic networks and their targets
        self.residual_critic_network_1 = Residual_Critic_Network()
        self.residual_critic_network_2 = Residual_Critic_Network()
        self.target_critic_network_1 = copy.deepcopy(self.residual_critic_network_1)
        self.target_critic_network_2 = copy.deepcopy(self.residual_critic_network_2)
        
        self.actor_optimizer = optim.Adam(self.residual_actor_network.parameters(), lr=ACTOR_LR)
        self.critic_optimizer_1 = optim.Adam(self.residual_critic_network_1.parameters(), lr=CRITIC_LR)
        self.critic_optimizer_2 = optim.Adam(self.residual_critic_network_2.parameters(), lr=CRITIC_LR)

        self.target_actor = copy.deepcopy(self.residual_actor_network)

        # TD3 specific hyperparameters
        self.policy_update_delay = 2  # Delay policy (actor) update frequency
        self.target_policy_noise = 0.2  # Noise added to target policy
        self.noise_clip = 0.5  # Clipping value for target policy noise
        self.max_action = constants.ROBOT_MAX_ACTION  # Maximum action value, adjust as per your environment

        # Initialize lists to track losses
        self.baseline_losses = []
        self.actor_losses = []
        self.critic_losses = []

        self.previous_states = []  # To track previous states
        self.state_threshold = STUCK_THRESHOLD  # Set a threshold value for state change
        self.state_change_steps = STUCK_STEPS  # The number of steps to check for state change



  
    def get_next_action_type(self, state, money_remaining):
        # Initialize action_type with a default value
        action_type = 'step'

        if (self.num_episodes <= NUM_DEMO) and not self.demo_flag:
            self.num_episodes += 1
            action_type = 'demo'

        if (self.num_episodes > NUM_DEMO) and not self.demo_flag:
            self.demo_flag = True
            self.num_episodes += 1
            self.train_policy(num_epochs = BASELINE_EPOCHS, minibatch_size = BASELINE_BATCH)
            action_type = 'reset'

        if self.plan_index == (PATH_LENGTH - 1):
            self.plan_index = 0
            self.num_episodes += 1
            self.current_noise_scale *= NOISE_DECAY
            self.td3_update(num_epochs = DDPG_EPOCHS, batch_size=DDPG_BATCH_SIZE, gamma = GAMMA, tau = TAU)
            action_type = 'reset'

        else:
            self.plan_index += 1 

        if self.check_if_stuck(state):
            self.plan_index = 0
            self.td3_update(num_epochs = DDPG_EPOCHS, batch_size=DDPG_BATCH_SIZE, gamma = GAMMA, tau = TAU)
            action_type = 'reset'

        # Debugging output
        print(f"Episode: {self.num_episodes} Action: {action_type}, Money: {money_remaining}, Steps: {self.plan_index}")

        return action_type


    def check_if_stuck(self, state):

        stuck = False

        # Check if state has not changed significantly for a certain number of steps
        if len(self.previous_states) >= self.state_change_steps:
            if all(np.linalg.norm(np.array(state) - np.array(prev_state)) < self.state_threshold for prev_state in self.previous_states[-self.state_change_steps:]):
                stuck = True
                self.previous_states.clear()  # Clear the state history after reset
            else:
                self.previous_states.pop(0)  # Remove the oldest state if not resetting

        # Add the current state to the tracking list
        self.previous_states.append(state)

        return stuck



    def get_next_action_training(self, state, money_remaining):
        baseline_action = self.get_action_from_model(state)
        residual_action = self.residual_action(baseline_action)
        corrected_action = baseline_action + residual_action
        noise = self.generate_noise(corrected_action)
        noisy_action = corrected_action + noise
        final_action = np.clip(noisy_action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)

        self.draw_action(state, baseline_action, [0,0,255])
        self.draw_action(state, residual_action, [255,0,255])

        # print(f'Baseline: {baseline_action}, Residual: {residual_action}, Noise: {noise}, Final {final_action}')

        return final_action
    
    # Used for testing, currently only getting baseline
    def get_action_from_model(self, state):
         # Normalize the state
        normalized_state = self.normalize(state, self.state_mean, self.state_std)

        self.baseline_network.eval()
        with torch.no_grad():
            # Convert normalized state to tensor and add a batch dimension
            state_tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0)
            
            # Get the action from the policy network
            action = self.baseline_network(state_tensor)
            
            # Convert to numpy array and remove batch dimension
            action_np = action.numpy().squeeze()

        return action_np
    
    def draw_action(self, state, action, colour):
        expected_state = self.dynamics_model(state, action)
        path = np.array([state, expected_state])
        self.draw_path(path, colour, 1)
        
    
    
    def residual_action(self, state):
        # Assuming state is a numpy array and needs to be unsqueezed to simulate batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        self.residual_actor_network.eval()  # Set the network to evaluation mode
        with torch.no_grad():  # Ensure no gradients are computed
            residual_action = self.residual_actor_network(state_tensor)

        residual_action_np = residual_action.squeeze(0).numpy()

        return residual_action_np

    
    def generate_noise(self, action):
        # Convert action to NumPy array if it's a tensor
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        
        # Generate noise
        noise = np.random.normal(0, self.current_noise_scale * constants.ROBOT_MAX_ACTION, size=action.shape)
        
        return noise


    def get_next_action_testing(self, state):
        baseline_action = self.get_action_from_model(state)
        residual_action = self.residual_action(baseline_action)
        corrected_action = baseline_action + residual_action
        final_action = np.clip(corrected_action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)

        print(f'Baseline: {baseline_action}, Residual: {residual_action}, Final {final_action}')

        return final_action

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        reward = self.compute_reward([next_state])
        done = self.plan_index == (PATH_LENGTH - 1)
        self.memory.push(state, action, reward, next_state, done)


    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):

        self.demonstration_states.extend(demonstration_states)
        self.demonstration_actions.extend(demonstration_actions)

        self.draw_path(demonstration_states, [0, 255, 0], 2)

        self.augment_demonstration_data(demonstration_states, demonstration_actions)

        for i in range(len(demonstration_states) - 1):  # Assuming sequential data
            state = demonstration_states[i]
            action = demonstration_actions[i]
            next_state = demonstration_states[i + 1]
            reward = self.compute_reward([next_state])
            done = i == len(demonstration_states) - 2
            self.memory.push(state, action, reward, next_state, done)


    def dynamics_model(self, state, action):
        # does nothing
        return state + action


    # Function to calculate the reward for a path, in order to evaluate how good the path is
    def compute_reward(self, path):
        # Original reward based on distance to the goal state
        goal_distance_reward = -np.linalg.norm(path[-1] - self.goal_state)

        # If there are no demonstration states, fallback to original reward
        if not self.demonstration_states:
            return goal_distance_reward

        # Convert current state and demonstration states to numpy arrays for distance calculation
        current_state = np.array([path[-1]])  # Convert last state of the path to 2D array for cdist
        demonstration_states_np = np.array(self.demonstration_states)

        # Compute distances from current state to all demonstration states
        distances = distance.cdist(current_state, demonstration_states_np, 'euclidean')

        # Find the minimum distance to a demonstration state
        min_distance_to_demo = np.min(distances)

        # Secondary objective: Reward for being close to any demonstration state
        # You might want to normalize or scale this distance to fit your reward scheme
        demo_proximity_reward = -min_distance_to_demo

        # Combine the two rewards
        # You can adjust the weighting of goal_distance_reward vs. demo_proximity_reward
        # depending on their relative importance
        reward = goal_distance_reward + demo_proximity_reward

        return reward

    def normalize(self, data, mean, std):
        return (data - mean) / std

    def train_policy(self, num_epochs, minibatch_size):

        # Convert demonstration data to numpy arrays for easier handling
        state_array = np.array(self.demonstration_states, dtype=np.float32)
        action_array = np.array(self.demonstration_actions, dtype=np.float32)

        if len(state_array) < 1:
            return
        
        # Normalize demonstration states and actions
        state_array_normalized = self.normalize(state_array, self.state_mean, self.state_std)
        # action_array_normalized = self.normalize(action_array, self.action_mean, self.action_std)

        for epoch in range(num_epochs):
            epoch_losses = []
            num_samples = len(state_array_normalized)
            num_batches = max(num_samples // minibatch_size, 1)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * minibatch_size
                end_idx = min((batch_idx + 1) * minibatch_size, num_samples)

                states = torch.tensor(state_array_normalized[start_idx:end_idx], dtype=torch.float32)
                actions = torch.tensor(action_array[start_idx:end_idx], dtype=torch.float32)

                self.optimizer.zero_grad()
                outputs = self.baseline_network(states)
                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            if len(epoch_losses) > 0:
                avg_loss = np.mean(epoch_losses)
                self.baseline_losses.append(avg_loss)
                print(f"Policy Network: Epoch {epoch + 1}, Average Loss: {avg_loss}")

        print("Policy Network: Training completed.")
        

    def draw_path(self, path, colour, width):
        path_to_draw = PathToDraw(path, colour=colour, width=width)  
        self.paths_to_draw.append(path_to_draw)
    
    def td3_update(self, num_epochs, batch_size, gamma, tau):
        for epoch in range(num_epochs):
            critic_loss_1, critic_loss_2 = self.train_critic(batch_size, gamma)
            # Delayed policy update
            if epoch % self.policy_update_delay == 0:
                actor_loss = self.train_actor(batch_size)
                self.actor_losses.append(actor_loss)
                # Soft update the target networks
                self.soft_update(self.target_actor, self.residual_actor_network, tau)
                self.soft_update(self.target_critic_network_1, self.residual_critic_network_1, tau)
                self.soft_update(self.target_critic_network_2, self.residual_critic_network_2, tau)
            else:
                actor_loss = None  # No actor update this epoch

            self.critic_losses.append((critic_loss_1 + critic_loss_2) / 2)  # Average critic loss for tracking

            # Optionally print the losses
            if actor_loss is not None:
                print(f'Epoch {epoch + 1}/{num_epochs}, Critic Loss: {(critic_loss_1 + critic_loss_2) / 2:.4f}, Actor Loss: {actor_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}, Critic Loss: {(critic_loss_1 + critic_loss_2) / 2:.4f}')

        # Call the method to plot losses after each update
        self.plot_losses()


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    
    def train_critic(self, batch_size, gamma):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(1 - dones).unsqueeze(1)  # 1 for not done, 0 for done

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)

            next_Q_values_1 = self.target_critic_network_1(next_states, next_actions)
            next_Q_values_2 = self.target_critic_network_2(next_states, next_actions)
            Q_targets = rewards + gamma * torch.min(next_Q_values_1, next_Q_values_2) * dones

        current_Q_values_1 = self.residual_critic_network_1(states, actions)
        current_Q_values_2 = self.residual_critic_network_2(states, actions)

        critic_loss_1 = self.criterion(current_Q_values_1, Q_targets)
        critic_loss_2 = self.criterion(current_Q_values_2, Q_targets)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        return critic_loss_1.item(), critic_loss_2.item()


    def train_actor(self, batch_size):
        states, _, _, _, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)

        actions = self.residual_actor_network(states)
        actor_loss = -self.residual_critic_network_1(states, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    
    def plot_losses(self):
 
        # Plot for baseline losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.baseline_losses, label='Baseline Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')  # Set the y-axis to logarithmic scale
        plt.title('Baseline Training Losses')
        plt.savefig('baseline_losses.png')
        plt.close()  # Close the figure to free memory

        # Plot for actor losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.actor_losses, label='Actor Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Set the y-axis to logarithmic scale
        plt.legend()
        plt.title('Actor Training Losses')
        plt.savefig('actor_losses.png')
        plt.close()

        # Plot for critic losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.critic_losses, label='Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Set the y-axis to logarithmic scale
        plt.legend()
        plt.title('Critic Training Losses')
        plt.savefig('critic_losses.png')
        plt.close()

    def augment_demonstration_data(self, demonstration_states, demonstration_actions, noise_level=2.5, interpolation_steps=5, num_augmentations=3):

        for augmentation in range(num_augmentations):
            augmented_states = []
            augmented_actions = []

            # Apply noise and interpolation
            for i in range(len(demonstration_states) - 1):
                current_state = demonstration_states[i]
                next_state = demonstration_states[i + 1]
                action = demonstration_actions[i]

                for step in range(1, interpolation_steps + 1):
                    fraction = step / float(interpolation_steps + 1)
                    synthetic_state = current_state + fraction * (next_state - current_state)

                    noisy_state = synthetic_state + np.random.normal(0, noise_level, synthetic_state.shape)
                    noisy_action = action + np.random.normal(0, noise_level, action.shape)

                    augmented_states.append(noisy_state)
                    augmented_actions.append(noisy_action)

                noisy_current_state = current_state + np.random.normal(0, noise_level, current_state.shape)
                noisy_action = action + np.random.normal(0, noise_level, action.shape)

                augmented_states.append(noisy_current_state)
                augmented_actions.append(noisy_action)

            # Optionally, add the last state with some noise
            last_state = demonstration_states[-1] + np.random.normal(0, noise_level, demonstration_states[-1].shape)
            augmented_states.append(last_state)

            # Repeat the last known action with noise
            augmented_actions.append(demonstration_actions[-1] + np.random.normal(0, noise_level, demonstration_actions[-1].shape))
            
            self.draw_path(np.array(augmented_states), [0, 0, 255], 2)

            self.demonstration_states.extend(augmented_states)
            self.demonstration_actions.extend(augmented_actions)








