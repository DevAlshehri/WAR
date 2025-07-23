import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import random
import math
import numba
import re

# Stable Baselines3 for Reinforcement Learning
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# ==============================================================================
# 1. HYPERPARAMETERS AND CONFIGURATION (Tweak these)
# ==============================================================================

# --- Training Hyperparameters ---
TOTAL_TIMESTEPS = 10_000_000
NUM_ENVIRONMENTS = 16
LEARNING_RATE = 0.0003
N_STEPS = 4096
BATCH_SIZE = 1024
POLICY_KWARGS = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]))

# --- Simulation Configuration ---
TROOP_COUNT = 500
NUM_OBSTACLES = 10
MAX_STEPS_PER_EPISODE = 2500

# --- File Paths ---
MODEL_SAVE_PATH = "./models"
LOG_PATH = "./logs"
CHECKPOINT_FREQ = 50000

# ==============================================================================
# 2. THE PYGAME SIMULATION (Now with Numba for performance)
# ==============================================================================

# --- Soldier Properties ---
SOLDIER_RADIUS = 4
SOLDIER_SPEED = 60
SOLDIER_HEALTH = 100
SOLDIER_DAMAGE = 10
SOLDIER_ATTACK_RANGE_SQ = 40**2
SOLDIER_ATTACK_COOLDOWN = 1.0

@numba.jit(nopython=True)
def find_target_numba(self_pos, enemy_positions, enemy_aliveness):
    closest_enemy_idx = -1
    min_dist_sq = 1e9
    for i in range(len(enemy_positions)):
        if enemy_aliveness[i]:
            dx = self_pos[0] - enemy_positions[i][0]
            dy = self_pos[1] - enemy_positions[i][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_enemy_idx = i
    return closest_enemy_idx, min_dist_sq

class WarSimEnv(gym.Env):
    def __init__(self):
        super(WarSimEnv, self).__init__()
        self.screen_width = 640
        self.screen_height = 480
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

    def _get_obs(self):
        blue_alive_indices = np.where((self.teams == 0) & (self.aliveness == 1))[0]
        red_alive_indices = np.where((self.teams == 1) & (self.aliveness == 1))[0]
        
        blue_count = len(blue_alive_indices)
        red_count = len(red_alive_indices)

        obs = np.zeros(8, dtype=np.float32)
        obs[0] = blue_count / TROOP_COUNT
        obs[1] = red_count / TROOP_COUNT

        if blue_count > 0:
            obs[2] = np.sum(self.healths[blue_alive_indices]) / (blue_count * SOLDIER_HEALTH)
            blue_com = np.mean(self.positions[blue_alive_indices], axis=0)
            obs[4] = blue_com[0] / self.screen_width
            obs[5] = blue_com[1] / self.screen_height
        
        if red_count > 0:
            obs[3] = np.sum(self.healths[red_alive_indices]) / (red_count * SOLDIER_HEALTH)
            red_com = np.mean(self.positions[red_alive_indices], axis=0)
            obs[6] = red_com[0] / self.screen_width
            obs[7] = red_com[1] / self.screen_height
            
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.obstacles = np.array([
            [random.uniform(self.screen_width / 4, self.screen_width * 3 / 4 - 80), random.uniform(0, self.screen_height - 80), random.randint(30, 80), random.randint(30, 80)]
            for _ in range(NUM_OBSTACLES)
        ], dtype=np.float32)

        total_soldiers = TROOP_COUNT * 2
        self.positions = np.zeros((total_soldiers, 2), dtype=np.float32)
        self.teams = np.zeros(total_soldiers, dtype=np.int8)
        self.healths = np.full(total_soldiers, SOLDIER_HEALTH, dtype=np.float32)
        self.aliveness = np.ones(total_soldiers, dtype=np.bool_)
        self.attack_cooldowns = np.zeros(total_soldiers, dtype=np.float32)
        self.targets = np.full(total_soldiers, -1, dtype=np.int32)

        for i in range(TROOP_COUNT):
            self.positions[i] = [random.uniform(0, self.screen_width / 4), random.uniform(0, self.screen_height)]
            self.teams[i] = 0
            
            red_idx = i + TROOP_COUNT
            self.positions[red_idx] = [random.uniform(self.screen_width * 3/4, self.screen_width), random.uniform(0, self.screen_height)]
            self.teams[red_idx] = 1

        self.prev_blue_health = np.sum(self.healths[self.teams == 0])
        self.prev_red_health = np.sum(self.healths[self.teams == 1])

        return self._get_obs(), {}

    def step(self, action):
        self.positions, self.healths, self.aliveness, self.attack_cooldowns, self.targets = self.update_simulation(
            self.positions, self.healths, self.aliveness, self.attack_cooldowns, self.targets, self.teams, self.obstacles, action
        )
        self.current_step += 1
        
        current_blue_health = np.sum(self.healths[self.teams == 0])
        current_red_health = np.sum(self.healths[self.teams == 1])
        damage_dealt = (self.prev_red_health - current_red_health) * 0.1
        damage_taken = (self.prev_blue_health - current_blue_health) * 0.1
        blue_alive_count = np.sum(self.aliveness[self.teams == 0])
        survival_bonus = blue_alive_count * 0.001
        reward = (damage_dealt - damage_taken) + survival_bonus - 0.01
        
        self.prev_blue_health = current_blue_health
        self.prev_red_health = current_red_health
        
        observation = self._get_obs()
        
        blue_alive = blue_alive_count > 0
        red_alive = np.sum(self.aliveness[self.teams == 1]) > 0
        
        terminated = not blue_alive or not red_alive
        truncated = self.current_step >= MAX_STEPS_PER_EPISODE
        
        if terminated:
            if blue_alive: reward += 500
            else: reward -= 500

        return observation, reward, terminated, truncated, {}

    @staticmethod
    @numba.jit(nopython=True)
    def update_simulation(positions, healths, aliveness, attack_cooldowns, targets, teams, obstacles, blue_command):
        dt = 0.1
        num_soldiers = len(positions)
        for i in range(num_soldiers):
            if attack_cooldowns[i] > 0:
                attack_cooldowns[i] -= dt

        for i in range(num_soldiers):
            if not aliveness[i]: continue

            if targets[i] != -1 and not aliveness[targets[i]]:
                targets[i] = -1
            if targets[i] == -1:
                enemy_indices = np.where((teams != teams[i]) & (aliveness == 1))[0]
                if len(enemy_indices) > 0:
                    target_idx, _ = find_target_numba(positions[i], positions[enemy_indices], aliveness[enemy_indices])
                    targets[i] = enemy_indices[target_idx]

            move_direction = np.zeros(2, dtype=np.float32)
            target_idx = targets[i]
            if target_idx != -1:
                dx = positions[target_idx][0] - positions[i][0]
                dy = positions[target_idx][1] - positions[i][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < SOLDIER_ATTACK_RANGE_SQ:
                    if attack_cooldowns[i] <= 0:
                        healths[target_idx] -= SOLDIER_DAMAGE
                        if healths[target_idx] <= 0: aliveness[target_idx] = False
                        attack_cooldowns[i] = SOLDIER_ATTACK_COOLDOWN
                else:
                    dist = np.sqrt(dist_sq)
                    move_direction[0] = dx / dist
                    move_direction[1] = dy / dist
            else:
                command = blue_command if teams[i] == 0 else 0
                if command == 0:
                    enemy_indices = np.where((teams != teams[i]) & (aliveness == 1))[0]
                    if len(enemy_indices) > 0:
                        enemy_com_x = np.mean(positions[enemy_indices, 0])
                        enemy_com_y = np.mean(positions[enemy_indices, 1])
                        dx = enemy_com_x - positions[i, 0]
                        dy = enemy_com_y - positions[i, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1:
                            move_direction[0] = dx / dist
                            move_direction[1] = dy / dist

            if move_direction[0] != 0 or move_direction[1] != 0:
                next_x = positions[i, 0] + move_direction[0] * SOLDIER_SPEED * dt
                next_y = positions[i, 1] + move_direction[1] * SOLDIER_SPEED * dt
                collided = False
                for j in range(len(obstacles)):
                    obs = obstacles[j]
                    if obs[0] < next_x < obs[0] + obs[2] and obs[1] < next_y < obs[1] + obs[3]:
                        collided = True
                        break
                if not collided:
                    positions[i, 0] = next_x
                    positions[i, 1] = next_y
        return positions, healths, aliveness, attack_cooldowns, targets

# ==============================================================================
# 4. MAIN TRAINING EXECUTION WITH RESUME LOGIC
# ==============================================================================

def get_latest_checkpoint(path):
    """Finds the latest model checkpoint in a directory."""
    if not os.path.isdir(path):
        return None
    files = os.listdir(path)
    checkpoints = [f for f in files if f.startswith("warsim_model_") and f.endswith(".zip")]
    if not checkpoints:
        return None
    
    # Extract step numbers and find the max
    steps = [int(re.search(r"(\d+)_steps", f).group(1)) for f in checkpoints]
    latest_step = max(steps)
    return os.path.join(path, f"warsim_model_{latest_step}_steps.zip")

if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    
    print("--- Initializing Vectorized Environments ---")
    env = SubprocVecEnv([lambda: WarSimEnv() for i in range(NUM_ENVIRONMENTS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVIRONMENTS, 1),
        save_path=MODEL_SAVE_PATH,
        name_prefix="warsim_model"
    )

    latest_checkpoint = get_latest_checkpoint(MODEL_SAVE_PATH)

    if latest_checkpoint:
        print(f"--- Resuming training from {latest_checkpoint} ---")
        model = PPO.load(
            latest_checkpoint,
            env=env,
            device="cuda",
            custom_objects={"learning_rate": LEARNING_RATE, "clip_range": 0.2}
        )
        # The number of timesteps completed is in the filename
        completed_steps = int(re.search(r"(\d+)_steps", latest_checkpoint).group(1))
        remaining_timesteps = TOTAL_TIMESTEPS - completed_steps
        if remaining_timesteps <= 0:
            print("--- Training is already complete. ---")
            exit()
        print(f"--- Training for an additional {remaining_timesteps} timesteps. ---")
    else:
        print("--- Starting a new training run. ---")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=LOG_PATH,
            policy_kwargs=POLICY_KWARGS,
            device="cuda"
        )
        remaining_timesteps = TOTAL_TIMESTEPS

    print(f"Policy Network Architecture: {model.policy_kwargs['net_arch']}")
    
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False # IMPORTANT: Do not reset the step counter
    )

    final_model_path = os.path.join(MODEL_SAVE_PATH, "warsim_model_final")
    model.save(final_model_path)
    print(f"--- Training Complete. Final model saved to {final_model_path} ---")
