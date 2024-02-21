##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

# Imports from this project
import constants
import configuration
from graphics import PathToDraw

# Demo
NUM_DEMO = 3

# Residual Actor Critic 
PATH_LENGTH = 50


class Baseline_Policy_Network(nn.Module):
    def __init__(self):
        super(Baseline_Policy_Network, self).__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=50)
        self.layer_2 = nn.Linear(in_features=50, out_features=50)
        self.layer_3 = nn.Linear(in_features=50, out_features=50)
        self.output_layer = nn.Linear(in_features=50, out_features=2)

        # Apply custom weight initialization
        self.apply(self.init_weights)

    def forward(self, input):
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        # Scale output with tanh to fit the action range, then scale to action bounds
        output = torch.tanh(output) * constants.ROBOT_MAX_ACTION
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
        self.layer_1 = nn.Linear(in_features=2, out_features=50)
        self.layer_2 = nn.Linear(in_features=50, out_features=50)
        self.layer_3 = nn.Linear(in_features=50, out_features=50)
        self.output_layer = nn.Linear(in_features=50, out_features=2)

        # Apply custom weight initialization
        self.apply(self.init_weights)

    def forward(self, input):
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        # Scale output with tanh to fit the action range, then scale to action bounds
        output = torch.tanh(output) * constants.ROBOT_MAX_ACTION
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
        self.layer_1 = nn.Linear(in_features=4, out_features=50)
        self.layer_2 = nn.Linear(in_features=50, out_features=50)
        self.layer_3 = nn.Linear(in_features=50, out_features=50)
        self.output_layer = nn.Linear(in_features=50, out_features=1)

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
        self.optimizer = optim.Adam(self.baseline_network.parameters(), lr=0.001)  # Initial learning rate
        self.criterion = nn.MSELoss()
        self.demonstration_states = []
        self.demonstration_actions = []

        self.demo_flag = False
        self.num_episodes = 0

        # Replay buffer
        self.memory = ReplayBuffer(10000)  # Example size

        # Planning
        self.plan_index = 0

        # Normalisation
        self.state_mean = np.array([50.0, 50.0])  
        self.state_std = np.array([29, 29]) 
        self.action_mean = np.array([0.0, 0.0])  
        self.action_std = np.array([2.5, 2.5])  

        # Residual Actor / Critic
        self.residual_actor_network = Residual_Actor_Network()
        self.residual_critic_network = Residual_Critic_Network()
        self.actor_optimizer = optim.Adam(self.residual_actor_network.parameters(), lr=0.001) 
        self.critic_optimizer = optim.Adam(self.residual_critic_network.parameters(), lr=0.001)  

        self.target_actor = copy.deepcopy(self.residual_actor_network)
        self.target_critic = copy.deepcopy(self.residual_critic_network)
        
  
    def get_next_action_type(self, state, money_remaining):
        # Initialize action_type with a default value
        action_type = 'step'

        if (self.num_episodes < NUM_DEMO) and not self.demo_flag:
            self.demo_flag = True
            self.num_episodes += 1
            self.train_policy()
            action_type = 'demo'

        if self.plan_index == (PATH_LENGTH - 1):
            self.plan_index = 0
            self.num_episodes += 1
            self.demo_flag = False
            self.train_policy()
            self.ddpg_update(batch_size=16)
            action_type = 'reset'

        else:
            self.plan_index += 1 

        # Debugging output
        print(f"Action Type: {action_type}, Money Remaining: {money_remaining}, Steps Taken: {self.plan_index}, Episode: {self.num_episodes}")

        return action_type

    def get_next_action_training(self, state, money_remaining):
        baseline_action = self.get_action_from_model(state)
        corrected_action = self.residual_action(baseline_action)
        noisy_action = self.add_noise(corrected_action)
        final_action = np.clip(noisy_action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)

        print(f'Baseline: {baseline_action}, Residual: {corrected_action}, Noisey: {noisy_action}, Final {final_action}')

        return final_action
    
    def residual_action(self, state):
        # Assuming state is a numpy array and needs to be unsqueezed to simulate batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        self.residual_actor_network.eval()  # Set the network to evaluation mode
        with torch.no_grad():  # Ensure no gradients are computed
            residual_action = self.residual_actor_network(state_tensor)

        residual_action_np = residual_action.squeeze(0).numpy()

        return residual_action_np

    
    def add_noise(self, action):
        # Convert action to NumPy array if it's a tensor
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        
        # Determine the scale of the noise based on the episode number
        noise_scale = 0.2 if self.num_episodes < 6 else 0.05
        
        # Generate noise
        noise = np.random.normal(0, noise_scale * constants.ROBOT_MAX_ACTION, size=action.shape)
        
        # Add noise to the action
        noisy_action = action + noise
        
        return noisy_action



    def get_next_action_testing(self, state):
        baseline_action = self.get_action_from_model(state)
        corrected_action = self.residual_action(baseline_action)
        noisy_action = self.add_noise(corrected_action)
        return noisy_action

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
        reward = -np.linalg.norm(path[-1] - self.goal_state)
        return reward

    def train_policy(self):
        num_epochs = 100
        minibatch_size = 100  

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
                print(f"Policy Network: Epoch {epoch + 1}, Average Loss: {avg_loss}")

        print("Policy Network: Training completed.")
        

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
    

    def draw_path(self, path, colour, width):
        path_to_draw = PathToDraw(path, colour=colour, width=width)  
        self.paths_to_draw.append(path_to_draw)
    
    # DDPG Algorithm

    def ddpg_update(self, batch_size, gamma=0.99, tau=0.001):
        critic_loss = self.train_critic(self.residual_critic_network, self.target_actor, self.target_critic, self.critic_optimizer, self.memory, batch_size, gamma)
        
        actor_loss = self.train_actor(self.residual_actor_network, self.residual_critic_network, self.actor_optimizer, self.memory, batch_size)
        
        # Soft update the target networks
        self.soft_update(self.target_critic, self.residual_critic_network, tau)
        self.soft_update(self.target_actor, self.residual_actor_network, tau)
        
        return critic_loss, actor_loss


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def train_critic(self, critic, target_actor, target_critic, optimizer, replay_buffer, batch_size, gamma):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(1 - dones).unsqueeze(1)  # 1 for not done, 0 for done
        
        # Compute the target Q value
        with torch.no_grad():
            next_actions = target_actor(next_states)
            next_Q_values = target_critic(next_states, next_actions)
            Q_targets = rewards + (gamma * next_Q_values * dones)
        
        # Get current Q estimate
        current_Q_values = critic(states, actions)
        
        # Compute critic loss
        critic_loss = self.criterion(current_Q_values, Q_targets)
        
        # Step optimizer
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        return critic_loss.item()

    def train_actor(self, actor, critic, optimizer, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        
        # Compute actor loss
        actions = actor(states)
        actor_loss = -critic(states, actions).mean()
        
        # Step optimizer
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
        
        return actor_loss.item()
    
    def normalize(self, data, mean, std):
        return (data - mean) / std

    def unnormalize(self, data, mean, std):
        return (data * std) + mean







