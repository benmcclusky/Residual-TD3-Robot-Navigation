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
NUM_DEMO = 1

# Residual Actor Critic 
PATH_LENGTH = 50

INITIAL_NOISE = 0.3  # Initial noise scale
NOISE_DECAY = 0.75  # Decay rate for noise

STUCK_THRESHOLD = 1
STUCK_STEPS = 10

# TD3 specific hyperparameters
ACTOR_LR = 0.00001
CRITIC_LR = 0.00001
POLICY_UPDATE_DELAY = 2 # Delay policy (actor) update frequency
TARGET_POLICY_NOISE = 0.2 # Noise added to target policy
NOISE_CLIP = 0.5  # Clipping value for target policy noise
TD3_EPOCHS = 100
TD3_BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.001

    

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



class TD3:
    def __init__(self, actor_network, critic_network_1, critic_network_2, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU,
                 policy_noise=TARGET_POLICY_NOISE, noise_clip=NOISE_CLIP, policy_update_delay=POLICY_UPDATE_DELAY, 
                 num_epochs = TD3_EPOCHS, batch_size = TD3_BATCH_SIZE ):

        # Actor and Critic networks
        self.actor_network = actor_network
        self.critic_network_1 = critic_network_1
        self.critic_network_2 = critic_network_2

        # Target networks
        self.target_actor = copy.deepcopy(self.actor_network)
        self.target_critic_network_1 = copy.deepcopy(self.critic_network_1)
        self.target_critic_network_2 = copy.deepcopy(self.critic_network_2)

        # Optimizers
        self.actor_optimizer = optim.Adam(actor_network.parameters(), lr=actor_lr)
        self.critic_optimizer_1 = optim.Adam(critic_network_1.parameters(), lr=critic_lr)
        self.critic_optimizer_2 = optim.Adam(critic_network_1.parameters(), lr=critic_lr)

        # Loss Function
        self.criterion = nn.MSELoss()

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_delay = policy_update_delay
        self.max_action = constants.ROBOT_MAX_ACTION
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Tracking losses
        self.actor_losses = []
        self.critic_losses = []

    def td3_update(self, replay_buffer):

        for epoch in range(self.num_epochs):
            critic_loss_1, critic_loss_2 = self.train_critic(self.batch_size, self.gamma, replay_buffer)
            # Delayed policy update
            if epoch % self.policy_update_delay == 0:
                actor_loss = self.train_actor(self.batch_size, replay_buffer)
                self.actor_losses.append(actor_loss)
                # Soft update the target networks
                self.soft_update(self.target_actor, self.actor_network, self.tau)
                self.soft_update(self.target_critic_network_1, self.critic_network_1, self.tau)
                self.soft_update(self.target_critic_network_2, self.critic_network_2, self.tau)
            else:
                actor_loss = None  # No actor update this epoch

            self.critic_losses.append((critic_loss_1 + critic_loss_2) / 2)  # Average critic loss for tracking

            # Optionally print the losses
            if actor_loss is not None:
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Critic Loss: {(critic_loss_1 + critic_loss_2) / 2:.4f}, Actor Loss: {actor_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Critic Loss: {(critic_loss_1 + critic_loss_2) / 2:.4f}')

        # Call the method to plot losses after each update
        self.plot_losses()
        self.plot_critic_values()


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    
    def train_critic(self, batch_size, gamma, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(1 - dones).unsqueeze(1)  # 1 for not done, 0 for done

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)

            next_Q_values_1 = self.target_critic_network_1(next_states, next_actions)
            next_Q_values_2 = self.target_critic_network_2(next_states, next_actions)
            Q_targets = rewards + gamma * torch.min(next_Q_values_1, next_Q_values_2) * dones

        current_Q_values_1 = self.critic_network_1(states, actions)
        current_Q_values_2 = self.critic_network_2(states, actions)

        critic_loss_1 = self.criterion(current_Q_values_1, Q_targets)
        critic_loss_2 = self.criterion(current_Q_values_2, Q_targets)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        return critic_loss_1.item(), critic_loss_2.item()


    def train_actor(self, batch_size, replay_buffer):
        states, _, _, _, _ = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)

        actions = self.actor_network(states)
        actor_loss = -self.critic_network_1(states, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    
    def plot_losses(self):
 
        # Plot for actor losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.actor_losses, label='Actor Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.yscale('log')  # Set the y-axis to logarithmic scale
        plt.legend()
        plt.title('Actor Training Losses')
        plt.savefig('actor_losses.png')
        plt.close()

        # Plot for critic losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.critic_losses, label='Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.yscale('log')  # Set the y-axis to logarithmic scale
        plt.legend()
        plt.title('Critic Training Losses')
        plt.savefig('critic_losses.png')
        plt.close()

    
    def plot_critic_values(self):
        # Create a 100x100 grid of state values, for each dimension
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        grid_shape = X.shape
        
        # Flatten the grid to iterate over it
        states = np.vstack([X.ravel(), Y.ravel()]).T

        # Convert states to tensor
        states_tensor = torch.FloatTensor(states)
        
        # Use the actor network to generate actions for each state, if applicable
        with torch.no_grad():
            actions = self.actor_network(states_tensor)
        
        # Evaluate the critic for each state-action pair
        critic_values = self.critic_network_1(states_tensor, actions).detach().numpy()

        # Reshape the critic values to match the grid shape
        Z = critic_values.reshape(grid_shape)
        
        # Create a contour plot of critic values
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(X, Y, Z, levels=100, cmap='viridis')  # Adjust levels and colormap as needed
        plt.colorbar(cp)
        plt.title('Critic Value Function')
        plt.xlabel('State Dimension 1')
        plt.ylabel('State Dimension 2')
        plt.savefig('critic_value_function.png')





class Robot:
    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []

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

        self.td3_agent = TD3(actor_network=Residual_Actor_Network(),
                critic_network_1=Residual_Critic_Network(),
                critic_network_2=Residual_Critic_Network(),)

        # Initialize lists to track losses
        self.baseline_losses = []
        self.actor_losses = []
        self.critic_losses = []

        self.previous_states = []  # To track previous states
        self.state_threshold = STUCK_THRESHOLD  # Set a threshold value for state change
        self.state_change_steps = STUCK_STEPS  # The number of steps to check for state change

        self.path_length = PATH_LENGTH
        self.best_reward = -200


  
    def get_next_action_type(self, state, money_remaining):
        # Initialize action_type with a default value
        action_type = 'step'

        if (self.num_episodes <= NUM_DEMO) and not self.demo_flag:
            self.num_episodes += 1
            action_type = 'demo'

        if (self.num_episodes > NUM_DEMO) and not self.demo_flag:
            self.demo_flag = True
            self.num_episodes += 1
            action_type = 'reset'

        if self.plan_index == (self.path_length - 1):
            self.plan_index = 0
            self.num_episodes += 1
            self.current_noise_scale *= NOISE_DECAY
            # Update policy via TD3
            self.td3_agent.td3_update(self.memory)
            self.path_length += 20
            action_type = 'reset'

        else:
            self.plan_index += 1 

        if self.check_if_stuck(state):
            # print('Stuck')

            if money_remaining >= constants.COST_PER_RESET:
                self.plan_index = 0
                # Update policy via TD3
                self.td3_agent.td3_update(self.memory)
                action_type = 'reset'
            

        # # Debugging output
        # print(f"Episode: {self.num_episodes} Action: {action_type}, Money: {money_remaining}, Steps: {self.plan_index}, Best Reward: {self.best_reward}")

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
        # baseline_action = self.get_action_from_model(state)
        baseline_action = state - self.goal_state
        residual_action = self.residual_action(baseline_action)
        corrected_action = baseline_action + residual_action
        noise = self.generate_noise(corrected_action)
        noisy_action = corrected_action + noise
        final_action = np.clip(noisy_action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)
        # print(f'Baseline: {baseline_action}, Residual: {residual_action}, Noise: {noise}, Final {final_action}')

        print(f"Reward: {self.compute_reward([state])}")
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

    
    
    def residual_action(self, state):
        # Assuming state is a numpy array and needs to be unsqueezed to simulate batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        self.td3_agent.actor_network.eval()  # Set the network to evaluation mode
        with torch.no_grad():  # Ensure no gradients are computed
            residual_action = self.td3_agent.actor_network(state_tensor)

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
        # baseline_action = self.get_action_from_model(state)
        baseline_action = state - self.goal_state
        residual_action = self.residual_action(baseline_action)
        corrected_action = baseline_action + residual_action
        final_action = np.clip(corrected_action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)

        print(f"Reward: {self.compute_reward([state])}")

        # print(f'Baseline: {baseline_action}, Residual: {residual_action}, Final {final_action}')

        return final_action

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        
        reward = self.compute_reward([next_state])

        if self.check_if_stuck(state):
            reward = reward - 50

        done = self.plan_index == (self.path_length - 1)
        self.memory.push(state, action, reward, next_state, done)

        if reward > self.best_reward:
            self.best_reward = reward
        


    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):

        self.demonstration_states.extend(demonstration_states)
        self.demonstration_actions.extend(demonstration_actions)

        self.augment_demonstration_data(demonstration_states, demonstration_actions)

        self.draw_path(demonstration_states, [0, 255, 0], 2)

        for i in range(len(demonstration_states) - 1):  # Assuming sequential data
            state = demonstration_states[i]
            action = demonstration_actions[i]
            next_state = demonstration_states[i + 1]
            reward = self.compute_reward([next_state])
            done = i == len(demonstration_states) - 2
            self.memory.push(state, action, reward, next_state, done)


    def dynamics_model(self, state, action):
        return state + action


    # Function to calculate the reward for a path, in order to evaluate how good the path is
    def compute_reward(self, path):
        # Original reward based on distance to the goal state
        goal_distance_reward = -np.linalg.norm(path[-1] - self.goal_state)

        if goal_distance_reward >= -constants.TEST_DISTANCE_THRESHOLD:
            reward = 50
            return reward

        # If there are no demonstration states, fallback to original reward
        if not self.demonstration_states:
            return goal_distance_reward

        # Convert demonstration states to numpy arrays for distance calculation
        demonstration_states_np = np.array(self.demonstration_states)
        min_distances = []

        # Calculate the minimum distance to a demonstration state for each step in the path
        for step in path:
            current_state = np.array(step).reshape(1, -1)  # Reshape each step for cdist
            distances = distance.cdist(current_state, demonstration_states_np, 'euclidean')
            min_distance_to_demo = np.min(distances)
            min_distances.append(min_distance_to_demo)

        # Average the minimum distances for the path
        avg_min_distance = np.mean(min_distances)

        # Use the average minimum distance as the demo_proximity_reward
        demo_proximity_reward = -avg_min_distance

        if not self.demo_flag:
            demo_proximity_reward = 0

        # Combine the two rewards
        reward = goal_distance_reward + (10 * demo_proximity_reward)

        return reward


    def draw_path(self, path, colour, width):
        path_to_draw = PathToDraw(path, colour=colour, width=width)  
        self.paths_to_draw.append(path_to_draw)


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
