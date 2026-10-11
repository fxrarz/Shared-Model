import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate RL agent stability across multi-seed checkpoints.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-Play-v0", help="Name of the task.")
parser.add_argument("--arch", type=str, choices=["DAC", "UAC"], default="DAC", help="Architecture to evaluate.")
parser.add_argument("--push_force", type=float, default=120.0, help="Magnitude of external push force (N).")
parser.add_argument("--push_interval", type=int, default=150, help="Simulation steps between pushes (~3s at 50Hz).")
parser.add_argument("--push_duration", type=int, default=25, help="Simulation steps force is held for (~0.5s).")
parser.add_argument("--push_body_name", type=str, default=None, help="Target body name to apply force.")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of evaluation episodes per checkpoint.")
parser.add_argument("--runs_dir", type=str, default="runs/", help="Directory containing checkpoint folders.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for environment physics.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import re
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed

set_seed(args_cli.seed)

# Network Definitions
class PolicyDAC(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class ValueDAC(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class SharedUAC(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum")
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU()
        )
        self.mean_layer = nn.Linear(100, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.value_layer = nn.Linear(100, 1)
        self._shared_output = None

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            mean = 2 * torch.tanh(self.mean_layer(self._shared_output))
            return mean, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is not None and self._shared_output.shape[0] == inputs["states"].shape[0]:
                shared_output = self._shared_output
            else:
                shared_output = self.net(inputs["states"])
            self._shared_output = None
            return self.value_layer(shared_output), {}

def main():
    env_cfg = parse_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 10.0
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    device = env.device

    memory = RandomMemory(memory_size=32, num_envs=1, device=device)
    
    if args_cli.arch == "DAC":
        models = {
            "policy": PolicyDAC(env.observation_space, env.action_space, device),
            "value": ValueDAC(env.observation_space, env.action_space, device)
        }
    else:
        uac_model = SharedUAC(env.observation_space, env.action_space, device)
        models = {"policy": uac_model, "value": uac_model}

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 32
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    agent = PPO(models=models, memory=memory, cfg=cfg,
                observation_space=env.observation_space, action_space=env.action_space, device=device)

    robot = env.unwrapped.scene["robot"]
    push_body_ids = [0] if args_cli.push_body_name is None else robot.find_bodies(args_cli.push_body_name)[0]

    def apply_push(force_vec):
        forces = torch.zeros(1, len(push_body_ids), 3, device=device)
        torques = torch.zeros_like(forces)
        forces[:, :, :] = force_vec
        robot.set_external_force_and_torque(forces, torques, body_ids=push_body_ids)

    def clear_push():
        forces = torch.zeros(1, len(push_body_ids), 3, device=device)
        torques = torch.zeros_like(forces)
        robot.set_external_force_and_torque(forces, torques, body_ids=push_body_ids)

    # Search for all seed folders matching arch and terrain
    terrain_key = "flat" if "Flat" in args_cli.task else "stair"
    search_pattern = os.path.join(args_cli.runs_dir, f"*{terrain_key}*_{args_cli.arch}*")
    folder_list = glob.glob(search_pattern)

    if not folder_list:
        print(f"No checkpoint folders matching pattern: {search_pattern}")
        env.close()
        return

    seed_rewards, seed_lengths, seed_fall_rates = [], [], []

    for ckpt_folder in sorted(folder_list):
        ckpt_path = os.path.join(ckpt_folder, "checkpoints", "best_agent.pt")
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"\nEvaluating Checkpoint: {ckpt_path}")
        try:
            agent.load(ckpt_path)
        except Exception as e:
            print(f"Skipping '{ckpt_path}': {e}")
            continue

        rewards, episode_lengths, fall_count = [], [], 0
        step_count = 0
        push_active = False
        push_end_step = -1

        obs, _ = env.reset()
        current_episode_reward = 0.0

        while len(rewards) < args_cli.num_episodes:
            # Trigger Push
            if not push_active and step_count > 0 and step_count % args_cli.push_interval == 0:
                angle = torch.rand(1).item() * 2 * math.pi
                force_vec = torch.tensor(
                    [args_cli.push_force * math.cos(angle), args_cli.push_force * math.sin(angle), 0.0],
                    device=device
                )
                apply_push(force_vec)
                push_active = True
                push_end_step = step_count + args_cli.push_duration

            # Clear Push
            if push_active and step_count >= push_end_step:
                clear_push()
                push_active = False

            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, reward, terminated, truncated, info = env.step(actions)

                current_episode_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                step_count += 1

                if terminated or truncated:
                    log_data = info.get("log", {})
                    fell = float(log_data.get('Episode_Termination/base_contact', 0.0)) > 0
                    if fell:
                        fall_count += 1

                    rewards.append(current_episode_reward)
                    episode_lengths.append(step_count)

                    clear_push()
                    push_active = False
                    step_count = 0
                    current_episode_reward = 0.0
                    obs, _ = env.reset()

        mean_rew = np.mean(rewards)
        mean_len = np.mean(episode_lengths)
        fall_rate = fall_count / len(rewards)

        seed_rewards.append(mean_rew)
        seed_lengths.append(mean_len)
        seed_fall_rates.append(fall_rate)

        print(f" Seed Results -> Reward: {mean_rew:.2f} | Length: {mean_len:.1f} | Fall Rate: {fall_rate:.2%}")

    env.close()

    # Aggregate Statistics across Seeds (Mean ± 95% CI)
    n_seeds = len(seed_rewards)
    if n_seeds > 0:
        def get_ci(arr):
            if n_seeds < 2:
                return 0.0
            return 1.96 * (np.std(arr, ddof=1) / np.sqrt(n_seeds))

        print(f"\n{'='*70}\nAGGREGATED BENCHMARK SUMMARY ({args_cli.arch} on {args_cli.task})\n{'='*70}")
        print(f"Evaluated Seeds: {n_seeds} | Push Force: {args_cli.push_force} N\n")
        print(f"Mean Reward         : {np.mean(seed_rewards):.2f} ± {get_ci(seed_rewards):.2f}")
        print(f"Mean Episode Length : {np.mean(seed_lengths):.2f} ± {get_ci(seed_lengths):.2f} steps")
        print(f"Mean Fall Rate       : {np.mean(seed_fall_rates)*100:.2f}% ± {get_ci(seed_fall_rates)*100:.2f}%")
        print(f"{'='*70}\n")

        print(f"\n{'='*70}\nRAW DATA ({args_cli.arch} on {args_cli.task})\n{'='*70}")
        print(f"Evaluated Seeds: {n_seeds} | Push Force: {args_cli.push_force} N\n")
        print(f"Reward         : {seed_rewards}")
        print(f"Length : {seed_lengths}")
        print(f"Fall Rate       : {seed_fall_rates}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
    simulation_app.close()
