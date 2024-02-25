##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import distance
import copy

# Imports from this project
import constants
import configuration
from graphics import PathToDraw

# Demo
NUM_DEMO = 1 # Number of demonstrations
NUM_AUGMENTS = 3 # The number of times demonstration data is augmented
AUG_NOISE = 2.5 # STD for augmentation
AUG_INTERPOLATION = 5 # Interpolation steps for augmentatioon

# Path Length
PATH_LENGTH = 50 # Starting Path Length
PATH_INCREASE = 20 # Increase every episode

# Exploration Noise
INITIAL_NOISE = 1  # Initial noise scale
NOISE_DECAY = 0.75  # Decay rate for noise

# Replay Buffer
BUFFER_SIZE = 10000 # Max size of replay buffer

# Reward Shaping
STUCK_THRESHOLD = 2 # Distance threshold for determining if stuck
STUCK_STEPS = 5 # Review number of steps for determining if stuck
STUCK_PENALTY = 50 # Reward penalty for being stuck
GOAL_REWARD = 50 # Reward for reaching goal
DEMO_PROXIMITY_FACTOR = 10 # Weighting for proximity to nearest demonstration states

# TD3 hyperparameters
ACTOR_LR = 0.00001 # Learning rate for actor
CRITIC_LR = 0.00001 # Learning rate for critic
POLICY_UPDATE_DELAY = 2 # Delay policy (actor) update frequency
TARGET_POLICY_NOISE = 0.2 # Noise added to target policy
NOISE_CLIP = 0.5  # Clipping value for target policy noise
TD3_EPOCHS = 100 # Number of epochs to train TD3
TD3_BATCH_SIZE = 100 # Batch size for TD3
GAMMA = 0.99 # Discount factor for future rewards in bellman equation
TAU = 0.001 # Target network update rate



class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling transitions collected during training.
    The buffer stores tuples of (state, action, reward, next_state, done) for each experience.
    
    Attributes:
        capacity (int): The maximum number of transitions the buffer can hold.
        buffer (list): A list to store the transitions.
        position (int): The current position in the buffer to insert the next transition.
    """
    def __init__(self, capacity):
        """
        Initializes the ReplayBuffer.
        
        Args:
            capacity (int): The maximum size of the buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer. If the buffer is full, it replaces the oldest transition.
        
        Args:
            state: The current state.
            action: The action taken in the current state.
            reward: The reward received after taking the action.
            next_state: The next state reached after taking the action.
            done: A boolean indicating whether the episode has ended.
        """
        # Expand the buffer if not yet at capacity
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # Insert the new transition
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # Update the position with wrap-around
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from the buffer.
        
        Args:
            batch_size (int): The size of the batch to sample.
        
        Returns:
            tuple of np.arrays: Tuple of states, actions, rewards, next_states, dones.
        """
        if len(self.buffer) < batch_size:
            # Return None if there aren't enough samples
            return None
        samples = np.random.choice(len(self.buffer), batch_size, replace=False)
        # Unpack the sampled transitions
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in samples])
        # Convert to numpy arrays and return
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        """
        Returns the current size of the buffer.
        
        Returns:
            int: The number of transitions in the buffer.
        """
        return len(self.buffer)



class Residual_Actor_Network(nn.Module):
    """
    A neural network architecture for the actor in a TD3 setup, using residual connections.
    This network predicts the action to take given the current state of the environment.

    The network consists of three fully connected layers with ReLU activations, followed by an output layer.
    It uses He initialization (also known as Kaiming initialization) for the weights to ensure appropriate scaling.

    Attributes:
        layer_1: The first fully connected layer.
        layer_2: The second fully connected layer.
        layer_3: The third fully connected layer.
        output_layer: The output layer producing the action values.
    """
    def __init__(self):
        super(Residual_Actor_Network, self).__init__()
        # Define the network layers
        self.layer_1 = nn.Linear(in_features=2, out_features=200)
        self.layer_2 = nn.Linear(in_features=200, out_features=200)
        self.layer_3 = nn.Linear(in_features=200, out_features=200)
        self.output_layer = nn.Linear(in_features=200, out_features=2)

        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, input):
        # Forward pass through the network
        layer_1_output = F.relu(self.layer_1(input))
        layer_2_output = F.relu(self.layer_2(layer_1_output))
        layer_3_output = F.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output

    def init_weights(self, m):
        # Custom weight initialization
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)  # Initialize biases to zero


class Residual_Critic_Network(nn.Module):
    """
    A neural network architecture for the critic in a TD3 setup, using residual connections.
    This network estimates the value of taking a given action in a given state of the environment.

    Similar to the actor network, it features three fully connected layers with ReLU activations
    and an output layer. Weights are initialized using He initialization for optimal performance.

    Attributes:
        layer_1: The first fully connected layer that combines state and action inputs.
        layer_2: The second fully connected layer.
        layer_3: The third fully connected layer.
        output_layer: The output layer producing the value estimate.
    """
    def __init__(self):
        super(Residual_Critic_Network, self).__init__()
        # Define the network layers
        self.layer_1 = nn.Linear(in_features=4, out_features=200)  # State and action combined
        self.layer_2 = nn.Linear(in_features=200, out_features=200)
        self.layer_3 = nn.Linear(in_features=200, out_features=200)
        self.output_layer = nn.Linear(in_features=200, out_features=1)

        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, state, action):
        # Combine state and action for the input
        input = torch.cat([state, action], dim=1)
        layer_1_output = F.relu(self.layer_1(input))
        layer_2_output = F.relu(self.layer_2(layer_1_output))
        layer_3_output = F.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output

    def init_weights(self, m):
        # Custom weight initialization
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)  # Initialize biases to zero


class TD3:

    """
    The TD3 (Twin Delayed Deep Deterministic Policy Gradient) class implements a reinforcement learning strategy designed to
    enhance the stability and efficiency of the DDPG algorithm. It utilizes twin critic networks to mitigate overestimation bias
    and employs delayed policy updates alongside target policy smoothing for improved learning outcomes. This class encapsulates
    the actor and critic networks, their target counterparts, and the training logic, including hyperparameters like discount
    rate, soft update rate, and noise parameters for action exploration. It's equipped to handle the training process by interacting
    with environments through experience replay, aiming to optimize policy and value functions towards achieving higher performance
    in decision-making tasks.

    """

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
        self.critic_optimizer_2 = optim.Adam(critic_network_2.parameters(), lr=critic_lr)

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
        """Perform a TD3 update for a certain number of epochs using the provided replay buffer.

        This method updates both the critic and actor networks using samples from the replay buffer.
        The critic network is updated at each epoch, while the actor network is updated with a delay
        to stabilize training. Target networks are softly updated after each actor update.

        Args:
            replay_buffer: The replay buffer from which training samples are drawn.

        """
        actor_losses = []
        critic_losses = []

        for epoch in range(self.num_epochs):
            # Step 1: Train critic networks and calculate losses
            critic_loss_1, critic_loss_2 = self.train_critic(replay_buffer)
            critic_losses.append((critic_loss_1 + critic_loss_2) / 2)

            # Step 2: Delayed policy updates for the actor network
            if epoch % self.policy_update_delay == 0:
                actor_loss = self.train_actor(replay_buffer)
                actor_losses.append(actor_loss)

                # Step 3: Soft update target networks
                self.soft_update(self.target_actor, self.actor_network, self.tau)
                self.soft_update(self.target_critic_network_1, self.critic_network_1, self.tau)
                self.soft_update(self.target_critic_network_2, self.critic_network_2, self.tau)

        # Optionally, log or print average losses for monitoring
        # average_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        # average_critic_loss = sum(critic_losses) / len(critic_losses)
        # print(f"Average Actor Loss: {average_actor_loss:.4f}, Average Critic Loss: {average_critic_loss:.4f}")


    def soft_update(self, target, source, tau):
        """Softly update target network parameters with those from the source network.

        This method implements the soft update strategy used in TD3 to gradually blend the parameters
        of the target network with those of the source network. This approach helps to maintain the
        stability of the learning process by ensuring that the target networks evolve smoothly over time.

        Args:
            target: The target network whose parameters are to be softly updated.
            source: The source network providing the new parameters.
            tau: The interpolation parameter indicating how much of the source parameters to blend
                into the target parameters. A value of 0 means no update, while 1 means full replacement.
        """
        # Iterate over the parameter pairs from both networks
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            # Perform the soft update for each parameter
            updated_param = target_param.data * (1.0 - tau) + source_param.data * tau
            target_param.data.copy_(updated_param)

    def train_critic(self, replay_buffer):
        """Train the critic networks using a batch of experiences from the replay buffer.

        This method updates the critic networks based on the temporal difference error, using the
        Bellman equation. It applies target smoothing techniques and noise to the target actions
        to stabilize training.

        Args:
            replay_buffer: The replay buffer from which to sample experiences.

        Returns:
            float, float: The loss values for the two critic networks after the update.
        """
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)

        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(1 - dones).unsqueeze(1)  # Convert 'dones' to a suitable format

        # Compute the target Q values
        with torch.no_grad():
            # Add noise to the next actions, clipped to the noise limits
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)

            # Calculate the next Q values from the target networks
            next_Q_values_1 = self.target_critic_network_1(next_states, next_actions)
            next_Q_values_2 = self.target_critic_network_2(next_states, next_actions)
            # The target Q value is the minimum of the two target Q values, adjusted by reward and discount factor
            Q_targets = rewards + self.gamma * torch.min(next_Q_values_1, next_Q_values_2) * dones

        # Compute the current Q values from the critic networks
        current_Q_values_1 = self.critic_network_1(states, actions)
        current_Q_values_2 = self.critic_network_2(states, actions)

        # Calculate the loss for both critic networks
        critic_loss_1 = self.criterion(current_Q_values_1, Q_targets)
        critic_loss_2 = self.criterion(current_Q_values_2, Q_targets)

        # Update the first critic network
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        # Update the second critic network
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Return the loss values for monitoring
        return critic_loss_1.item(), critic_loss_2.item()
    

    def train_actor(self, replay_buffer):
        """Train the actor network using policy gradient.

        This method updates the actor network to maximize the expected return as estimated by one of the critic networks.
        It uses a batch of states from the replay buffer to compute the policy gradient and update the actor network parameters.

        Args:
            replay_buffer: The replay buffer from which to sample experiences.

        Returns:
            float: The loss value for the actor network after the update.
        """
        # Sample a batch of states from the replay buffer
        states, _, _, _, _ = replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)

        # Generate actions for the current states using the actor network
        actions = self.actor_network(states)

        # Calculate the actor loss as the negative mean of the critic network's Q-value estimates for these actions
        # The negative sign is used to perform gradient ascent to maximize the critic's Q-value estimates
        actor_loss = -self.critic_network_1(states, actions).mean()

        # Perform backpropagation and an optimization step to update the actor network
        self.actor_optimizer.zero_grad()  # Reset gradients to zero for a clean update
        actor_loss.backward()  # Compute the gradient of the loss with respect to the actor network parameters
        self.actor_optimizer.step()  # Apply the gradients to update the actor network parameters

        # Return the actor loss value for monitoring
        return actor_loss.item()







class Robot:

    """
    The Robot class uses a residual reinforcement learning approach for robot navigation utilising TD3 as the prefered reinforcement learning algorithm. The class supports taking actions in the simulated enviroment both within the training and testing phase, calling for, storing, augmenting and visualising demonstrations, storing transition experiences in a replay buffer, computing rewards using a custom reward shaping method. 
    
    """

    def __init__(self, goal_state):

        self.goal_state = goal_state  # The desired goal state the robot aims to reach.
        self.paths_to_draw = []  # List of paths to draw for visualization.

        self.demonstration_states = []  # States from demonstrations.
        self.demonstration_actions = []  # Actions from demonstrations.

        self.num_episodes = 0  # Counter for the number of episodes completed.
        self.current_noise_scale = INITIAL_NOISE  # Scale of noise for action exploration.
        self.path_length = PATH_LENGTH  # Initial path length for planning.
        self.plan_index = 0  # Current index in the planned path.
        self.previous_states = []  # To track previous states for detecting stuck situations.

        # Replay buffer for storing and retrieving experiences.
        self.memory = ReplayBuffer(BUFFER_SIZE)

        # The TD3 learning agent with specific actor and critic networks.
        self.td3_agent = TD3(actor_network=Residual_Actor_Network(),
                             critic_network_1=Residual_Critic_Network(),
                             critic_network_2=Residual_Critic_Network())

        # Flags for indicating the state of the robot.
        self.goal_reached = False  # Flag for goal achievement.
        self.demo_flag = False  # Flag for demonstration mode.
        self.stuck_flag = False  # Flag for detecting if the robot is stuck.
       


  
    def get_next_action_type(self, state, money_remaining):
        """Determine the next action type based on the agent's current state and training progression.

        The method selects between continuing with the next step, following a demonstration, or resetting the state,
        based on the number of completed episodes, demonstration flags, and whether specific conditions like reaching
        the goal or getting stuck have occurred.

        Args:
            state: The current state of the agent.
            money_remaining: The amount of money remaining. This parameter is included for future extensibility and
                            is not used in the current implementation.

        Returns:
            str: The type of the next action ('step', 'demo', 'reset') depending on the agent's state and training progress.

        Note:
            - The action 'demo' is chosen if the agent is within the demonstration episode count and the demo flag is not set.
            - The action 'reset' is selected when transitioning out of the demonstration phase, or when the agent has reached
            the end of a path, achieved its goal, or is stuck.
            - Otherwise, the agent continues with the next 'step'.
        """
        # Default action type is to proceed with the next step
        action_type = 'step'

        # If within the number of demo episodes and not in demo mode, switch to demonstration action
        if (self.num_episodes <= NUM_DEMO) and not self.demo_flag:
            self.num_episodes += 1
            action_type = 'demo'

        # Transition out of demo mode once the demo episode count is exceeded
        if (self.num_episodes > NUM_DEMO) and not self.demo_flag:
            self.demo_flag = True
            self.num_episodes += 1
            action_type = 'reset'

        # Reset and update learning policy if at the end of a path, goal reached, or stuck
        if self.plan_index == (self.path_length - 1) or self.goal_reached or self.stuck_flag:
            self.reset()  # Reset various state flags and adjust for the new episode
            self.td3_agent.td3_update(self.memory)  # Update the agent's policy based on accumulated experience
            action_type = 'reset'
        else:
            # Otherwise, simply proceed to the next step in the current path
            self.plan_index += 1

        return action_type


    def reset(self):
        """Reset the agent's state for a new episode.

        This method increments the episode counter and resets the agent's plan index, goal achievement flag,
        and stuck flag. It also adjusts the current noise scale and path length for the next episode, 
        following predefined decay and increase rates.

        """

        self.num_episodes += 1          # Increment the number of episodes
        self.plan_index = 0             # Reset the plan index to the start
        self.goal_reached = False       # Reset the flag indicating whether the goal was reached
        self.stuck_flag = False         # Reset the flag indicating whether the agent is stuck
        self.current_noise_scale *= NOISE_DECAY  # Apply decay to the current noise scale
        self.path_length += PATH_INCREASE        # Increment the path length


    def check_if_stuck(self, state):
        """Determine if the agent is stuck based on its state history.

        Args:
            state: The current state of the agent, expected to be a list or array of numerical values.

        Returns:
            bool: True if the agent is considered stuck, False otherwise.
        """
        
        stuck = False

        # Ensure there's enough history to check for being stuck
        if len(self.previous_states) >= STUCK_STEPS:
            # Calculate differences between the current state and previous states
            differences = [np.linalg.norm(np.array(state) - np.array(prev_state)) for prev_state in self.previous_states[-STUCK_STEPS:]]

            # Check if all differences are below the stuck threshold
            if all(diff < STUCK_THRESHOLD for diff in differences):
                stuck = True
                # Clear the state history as we're resetting the check
                self.previous_states.clear()
            else:
                # Remove the oldest state if not stuck to maintain the history size
                self.previous_states.pop(0)

        # Add the current state to the history for future checks
        self.previous_states.append(state)

        return stuck


    def get_next_action_training(self, state, money_remaining):
        """Calculate the next action during training considering the current state and money remaining.

        This function computes the action based on the difference between the current state and the goal state,
        adjusts it with a residual action, adds noise for exploration, and clips it to the maximum action limits.

        Args:
            state: The current state of the agent.
            money_remaining: The remaining money, not used directly in action calculation but available for extension.

        Returns:
            np.ndarray: The calculated action to be taken, clipped to the maximum bounds.
        """
        
        # Calculate the baseline action as the difference from the goal state
        baseline_action = state - self.goal_state
        
        # Adjust the baseline action with a residual component
        residual_action = self.residual_action(baseline_action)
        corrected_action = baseline_action + residual_action
        
        # Introduce noise to the corrected action for exploration
        noise = self.generate_noise(corrected_action)
        noisy_action = corrected_action + noise
        
        # Clip the noisy action within the defined maximum action bounds
        final_action = np.clip(noisy_action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)
        
        return final_action


    def get_next_action_testing(self, state):
        """Calculate the next action during testing considering the current state.

        This function computes the action based on the difference between the current state and the goal state,
        adjusts it with a residual action and clips it to the maximum action limits.

        Args:
            state: The current state of the agent.

        Returns:
            np.ndarray: The calculated action to be taken, clipped to the maximum bounds.
        """
        
        # Calculate the baseline action as the difference from the goal state
        baseline_action = state - self.goal_state
        
        # Adjust the baseline action with a residual component
        residual_action = self.residual_action(baseline_action)
        corrected_action = baseline_action + residual_action
        
        # Clip the noisy action within the defined maximum action bounds
        final_action = np.clip(corrected_action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)
        
        return final_action


    def residual_action(self, state):
        """Compute the residual action for a given state using the TD3 agent's actor network.

        This method processes the state through the TD3 agent's actor network to produce a residual action.
        It ensures the network is in evaluation mode and that no gradients are computed during the forward pass.

        Args:
            state: The current state of the agent, expected to be a list or array of numerical values.

        Returns:
            numpy.ndarray: The residual action derived from the actor network, converted back into a NumPy array.
        """
        
        # Convert state to a PyTorch tensor and add a batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Ensure the actor network is in evaluation mode
        self.td3_agent.actor_network.eval()
        
        with torch.no_grad():  # Disable gradient computation for inference
            # Forward pass through the actor network to compute the residual action
            residual_action = self.td3_agent.actor_network(state_tensor)
        
        # Remove the batch dimension and convert the tensor back to a NumPy array
        residual_action_np = residual_action.squeeze(0).numpy()
        
        return residual_action_np

    
    def generate_noise(self, action):
        """Generate noise to add to the action for exploration.

        This method generates Gaussian noise based on the current noise scale and the action's shape.
        The noise is scaled by the maximum action constant to ensure it's proportional to the action magnitude.

        Args:
            action: The action to which the noise will be added, as a NumPy array.

        Returns:
            np.ndarray: The generated noise, with the same shape as the action.
        """
        # Generate Gaussian noise scaled by the current noise scale and maximum action limit
        noise = np.random.normal(0, self.current_noise_scale * constants.ROBOT_MAX_ACTION, size=action.shape)
        
        return noise


    def process_transition(self, state, action, next_state, money_remaining):
        """Process a transition and store it in memory.

        This function computes the reward for moving to the next state, checks if the agent is stuck,
        applies a penalty if so, determines if the current plan is done, and stores the transition
        in the agent's memory.

        Args:
            state: The current state of the agent.
            action: The action taken by the agent from the current state.
            next_state: The state resulting from the action taken.
            money_remaining: The amount of money remaining, not directly used but available for extension.

        Note:
            - The reward is calculated based on the next state.
            - A penalty is applied to the reward if the agent is determined to be stuck.
            - The 'done' flag indicates if the current plan or episode has concluded.
        """
        # Compute the reward based on the next state
        reward = self.compute_reward([next_state])

        # Check if the agent is stuck and apply a penalty to the reward if so
        if self.check_if_stuck(state):
            self.stuck_flag = True
            reward -= STUCK_PENALTY  

        # Determine if the plan is done based on the current plan index and path length
        done = self.plan_index == (self.path_length - 1)

        # Store the transition in the agent's memory
        self.memory.push(state, action, reward, next_state, done)



    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        """Process a demonstration for training.

        This function incorporates demonstration data into the agent's learning process by:
        - Extending the internal lists of states and actions with the demonstration data.
        - Augmenting the demonstration data for more robust learning.
        - Visually representing the demonstration path.
        - Processing each transition in the demonstration for memory storage.

        Args:
            demonstration_states: A list of states from the demonstration.
            demonstration_actions: A list of actions taken in the demonstration.
            money_remaining: The amount of money remaining, not used directly but available for extension.
        """
        # Extend the internal lists with the demonstration data
        self.demonstration_states.extend(demonstration_states)
        self.demonstration_actions.extend(demonstration_actions)

        # Optionally augment the demonstration data for more robust learning
        self.augment_demonstration_data(demonstration_states, demonstration_actions)

        # Draw the path taken in the demonstration
        self.draw_path(demonstration_states, colour=[0, 255, 0], width=2)

        # Process each transition in the demonstration
        for i in range(len(demonstration_states) - 1):
            state = demonstration_states[i]
            action = demonstration_actions[i]
            next_state = demonstration_states[i + 1]

            # Compute reward for the transition
            reward = self.compute_reward([next_state])

            # Determine if this is the last transition in the demonstration
            done = i == (len(demonstration_states) - 2)

            # Store the transition in memory
            self.memory.push(state, action, reward, next_state, done)


    # Not used
    def dynamics_model(self, state, action):
        return state + action


   
    def compute_reward(self, path):
        """Calculate the reward for a given path to evaluate its effectiveness.

        The reward is calculated based on two factors:
        - The distance to the goal state, with a higher reward for being closer.
        - The proximity to demonstration states, if available, to encourage similarity to demonstrated paths.

        Args:
            path: A list of states representing the path taken.

        Returns:
            float: The calculated reward for the path.
        """
        # Calculate the distance to the goal state from the last state in the path
        goal_distance_reward = -np.linalg.norm(path[-1] - self.goal_state)

        # If the path ends within a certain threshold of the goal, return a predefined goal reward
        if goal_distance_reward >= -constants.TEST_DISTANCE_THRESHOLD:
            self.goal_reached = True
            return GOAL_REWARD  

        # Default to goal distance reward if there are no demonstration states
        if not self.demonstration_states:
            return goal_distance_reward

        # Calculate the minimum distance to a demonstration state for each step in the path
        min_distances = [np.min(distance.cdist([step], self.demonstration_states, 'euclidean')) for step in path]

        # Average the minimum distances to get the demonstration proximity reward
        avg_min_distance = np.mean(min_distances)
        demo_proximity_reward = -avg_min_distance if self.demo_flag else 0  # Only apply if demo_flag is True

        # Combine the goal distance reward with the demonstration proximity reward
        reward = goal_distance_reward + (DEMO_PROXIMITY_FACTOR * demo_proximity_reward)  

        return reward


    # Method to visaulise paths 
    def draw_path(self, path, colour = [255, 255, 255], width = 2):
        path_to_draw = PathToDraw(path, colour=colour, width=width)  
        self.paths_to_draw.append(path_to_draw)


    def augment_demonstration_data(self, demonstration_states, demonstration_actions, noise_level=AUG_NOISE, interpolation_steps=AUG_INTERPOLATION, num_augmentations=NUM_AUGMENTS):
        """Augment demonstration data by applying noise and interpolation.

        This method creates synthetic demonstration data by interpolating between consecutive states and actions,
        applying Gaussian noise, and then extending the original demonstration data with this synthetic data. The
        process aims to enrich the training dataset, providing the model with a more diverse set of situations.

        Args:
            demonstration_states: List of original demonstration states.
            demonstration_actions: List of original demonstration actions.
            noise_level: The standard deviation of the Gaussian noise to be applied. Defaults to 2.5.
            interpolation_steps: The number of synthetic steps to create between each original step. Defaults to 5.
            num_augmentations: The number of times the augmentation process is applied. Defaults to 3.

        Note:
            - This method not only interpolates and applies noise to states and actions but also visually represents
            the augmented paths and updates the internal lists of demonstration states and actions.
        """
        for _ in range(num_augmentations):
            augmented_states = []
            augmented_actions = []

            # Iterate through demonstration states and actions to apply noise and interpolation
            for i in range(len(demonstration_states) - 1):
                current_state, next_state = demonstration_states[i], demonstration_states[i + 1]
                action = demonstration_actions[i]

                # Interpolate between current and next state, then apply noise
                for step in range(1, interpolation_steps + 1):
                    fraction = step / float(interpolation_steps + 1)
                    synthetic_state = current_state + fraction * (next_state - current_state)
                    noisy_state = synthetic_state + np.random.normal(0, noise_level, synthetic_state.shape)
                    noisy_action = action + np.random.normal(0, noise_level, action.shape)

                    augmented_states.append(noisy_state)
                    augmented_actions.append(noisy_action)

                # Apply noise directly to the current state and action
                noisy_current_state = current_state + np.random.normal(0, noise_level, current_state.shape)
                augmented_states.append(noisy_current_state)
                augmented_actions.append(action + np.random.normal(0, noise_level, action.shape))

            # Add noise to the last state and its action
            last_state_noisy = demonstration_states[-1] + np.random.normal(0, noise_level, demonstration_states[-1].shape)
            last_action_noisy = demonstration_actions[-1] + np.random.normal(0, noise_level, demonstration_actions[-1].shape)
            augmented_states.append(last_state_noisy)
            augmented_actions.append(last_action_noisy)

            # Visual representation of the augmented path
            self.draw_path(np.array(augmented_states), colour=[0, 0, 255], width=2)

            # Extend the demonstration data with the augmented data
            self.demonstration_states.extend(augmented_states)
            self.demonstration_actions.extend(augmented_actions)