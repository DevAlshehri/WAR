Project: AI War Simulation - V1 Commander Model
This document outlines the architecture, algorithms, and learned strategies of the first version of our reinforcement learning agent, trained for 10 million timesteps with 1 x A40 9 vCPU 50 GB RAM

1. Core Algorithm
Algorithm: Proximal Policy Optimization (PPO)

Library: stable-baselines3 (a PyTorch-based implementation)

Description: PPO is a state-of-the-art, actor-critic reinforcement learning algorithm. It's known for its stability and performance across a wide range of tasks, making it an ideal choice for this complex simulation.

2. AI Architecture: The "Central Commander"
Instead of giving every soldier its own brain (Multi-Agent RL), we trained a single, centralized "Army Commander" AI. This single neural network observes the overall state of the battlefield and issues one high-level strategic command that is then executed by every soldier in its army.

3. The AI's "Senses": Observation Space
The AI did not have a visual map of the battlefield. Instead, it made decisions based on a simplified, 8-number summary of the current situation:

Blue Army Strength: The percentage of blue soldiers remaining.

Red Army Strength: The percentage of red soldiers remaining.

Blue Army Average Health: The average health of all living blue soldiers.

Red Army Average Health: The average health of all living red soldiers.

Blue Army Center of Mass (X-coordinate): The normalized horizontal center of the blue army.

Blue Army Center of Mass (Y-coordinate): The normalized vertical center of the blue army.

Red Army Center of Mass (X-coordinate): The normalized horizontal center of the red army.

Red Army Center of Mass (Y-coordinate): The normalized vertical center of the red army.

4. The AI's "Commands": Action Space
The AI Commander could choose one of three discrete actions at any given time:

Action 0: ATTACK: A general order for all soldiers to move towards the enemy's calculated center of mass.

Action 1: HOLD POSITION: An order for all soldiers to stop moving.

Action 2: SPREAD OUT: An order for soldiers to move away from their own army's center of mass (intended to break up clusters).

5. The AI's "Motivation": Reward Function
The AI was trained using a sophisticated reward shaping strategy to encourage intelligent behavior:

Primary Goal (Damage Delta): The AI received a positive reward for dealing damage to the enemy and a negative reward for taking damage.

Secondary Goal (Survival Bonus): It received a small, continuous bonus for every soldier that remained alive. This taught the AI the value of preserving its forces.

Tertiary Goal (Efficiency Penalty): It received a tiny, continuous penalty for every second that passed. This encouraged the AI to win efficiently and avoid stalemates.

Terminal Goal (Victory Bonus): It received a large lump-sum reward for winning the battle and a large penalty for losing.

6. Learned Strategies and Limitations
Learned Behavior: The AI successfully learned to balance aggression and survival. It understood that it needed to attack to get the damage reward but also learned to value its troops' lives to get the survival bonus. Its primary strategy was to consolidate its forces and attack the enemy's core.

Critical Limitation (Lack of Spatial Awareness): The model's key weakness was its inability to "see" the battlefield layout. Because it only knew the center of the armies, it could not perceive troop density, choke points, or traffic jams. This led to the observed behavior where the AI would command its troops to "ATTACK" even if they were stuck behind an obstacle, as it was strategically blind to the physical blockage. This limitation is the primary motivation for developing the next-generation "High Ground" model.
