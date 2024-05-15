# Residual-TD3-Robot-Navigation

### Simulator
Continuous 2D action space robot navigation simulator with seeded start and end points
Unknown non-linear enviroment dynamics
Capability to request demonstrations 
Cost associated with enviroment reset, demonstration and taking a step

### Algorithm 
Uses imitation learning to create a baseline policy based on demonstrations 
Then uses residual learning over the baseline policy to adjust for non-linear dynamics and different starting positions. 
Uses Twin Delayed Deep Deterministic Policy Gradient (TD3) as the chosen RL algorithm.
