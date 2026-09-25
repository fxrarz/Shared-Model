# $ python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 100.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 110.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 120.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 130.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 140.0 --push_interval 10

# $ python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 100.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 110.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 120.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 130.0 --push_interval 10
# $ python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 140.0 --push_interval 10


import argparse
import sys
from isaaclab.app import AppLauncher

# 1. Parse CLI arguments first
parser = argparse.ArgumentParser(description="Evaluate RL agent stability under force perturbations.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Stairway-H1-Play-v0", help="Name of the task.")
parser.add_argument("--push_force", type=float, default=100.0, help="Magnitude of external push force (N).")
parser.add_argument("--push_interval", type=int, default=10, help="Simulation steps between pushes.")
parser.add_argument("--push_duration", type=int, default=10, help="Simulation steps force is held for.")
parser.add_argument("--push_body_name", type=str, default=None, help="Target body name to apply force (default: root body).")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of evaluation episodes per checkpoint.")
parser.add_argument("--runs_dir", type=str, default="runs/", help="Directory containing checkpoint folders.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

# Add Isaac Lab visualizer / headless flags
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clean sys.argv so Hydra/Omniverse parsers don't crash on custom arguments
sys.argv = [sys.argv[0]] + hydra_args

# Launch SimulationApp BEFORE importing torch, gym, or omniverse components
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports MUST happen AFTER AppLauncher initialization
import os
import re
import glob
import math
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

# Set seed
set_seed(args_cli.seed)


def parse_folder_name(name):
    match = re.match(r"(.*?)_nogui_(.*?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_seed_(\d+)(.*)", name)
    if match:
        terrain, arch, timestamp, seed, archived = match.groups()
        return (terrain, arch, int(seed), timestamp, archived)
    return (name, "", 0, "", "")


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 400),
                                 nn.ELU(),
                                 nn.Linear(400, 200),
                                 nn.ELU(),
                                 nn.Linear(200, 100),
                                 nn.ELU(),
                                 nn.Linear(100, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    def compute(self, inputs, role):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 400),
                                 nn.ELU(),
                                 nn.Linear(400, 200),
                                 nn.ELU(),
                                 nn.Linear(200, 100),
                                 nn.ELU(),
                                 nn.Linear(100, 1))
    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def main():
    # Load env using standard Isaac Lab pipeline
    env_cfg = parse_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 10.0
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    num_envs = 1
    device = env.device

    action_space = env.action_space
    observation_space = env.observation_space

    # Instantiate SKRL Memory & Agent
    memory = RandomMemory(memory_size=32, num_envs=num_envs, device=device)
    models = {
        "policy": Policy(observation_space, action_space, device),
        "value": Value(observation_space, action_space, device)
    }

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 32
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 8
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 4.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * 0.01
    cfg["time_limit_bootstrap"] = False
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=observation_space,
                action_space=action_space,
                device=device)

    # Physics body handles
    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]
    
    if args_cli.push_body_name is not None:
        body_ids, body_names = robot.find_bodies(args_cli.push_body_name)
        if len(body_ids) == 0:
            raise ValueError(f"No bodies matched '{args_cli.push_body_name}'. Available: {robot.body_names}")
        push_body_ids = body_ids
        print(f"Targeting push body: {body_names}")
    else:
        push_body_ids = [0]
        print(f"Targeting root body: {robot.body_names[0]}")

    def apply_push(force_vec):
        num_bodies = len(push_body_ids)
        forces = torch.zeros(unwrapped_env.num_envs, num_bodies, 3, device=device)
        torques = torch.zeros_like(forces)
        forces[:, :, :] = force_vec
        robot.set_external_force_and_torque(forces, torques, body_ids=push_body_ids)

    def clear_push():
        num_bodies = len(push_body_ids)
        forces = torch.zeros(unwrapped_env.num_envs, num_bodies, 3, device=device)
        torques = torch.zeros_like(forces)
        robot.set_external_force_and_torque(forces, torques, body_ids=push_body_ids)

    # Locate Checkpoints
    if args_cli.task == "Isaac-Velocity-Flat-H1-Play-v0":
        folder_list = glob.glob(os.path.join(args_cli.runs_dir, "c_flat_*_DAC*"))
    else:
        folder_list = glob.glob(os.path.join(args_cli.runs_dir, "c_stair*_DAC*"))
    sorted_categorically = sorted(folder_list, key=parse_folder_name)
    
    summary = {}

    for ckpt_folder in sorted_categorically:
        ckpt_path = f"{ckpt_folder}/checkpoints/best_agent.pt"
        print(f"\n{'='*60}\nLoading Checkpoint: {ckpt_path}\n{'='*60}")
        try:
            agent.load(ckpt_path)
        except Exception as e:
            print(f"Skipping '{ckpt_path}': {e}")
            continue

        rewards = []
        episode_lengths = []
        fall_count = 0
        fall_flags = []
        step_count = 0
        push_active = False
        push_end_step = -1

        obs, _ = env.reset()
        current_episode_reward = 0.0

        while len(rewards) < args_cli.num_episodes:
            # Periodically apply external force perturbation
            if not push_active and step_count > 0 and step_count % args_cli.push_interval == 0:
                angle = torch.rand(1).item() * 2 * math.pi
                force_vec = torch.tensor(
                    [args_cli.push_force * math.cos(angle),
                     args_cli.push_force * math.sin(angle),
                     0.0],
                    device=device,
                )
                apply_push(force_vec)
                push_active = True
                push_end_step = step_count + args_cli.push_duration

            if push_active and step_count >= push_end_step:
                clear_push()
                push_active = False

            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, reward, terminated, truncated, info = env.step(actions)

                step_reward = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                current_episode_reward += step_reward
                step_count += 1

                done = terminated or truncated

                if done:
                    log_data = info.get("log", {})
                    fell = float(log_data.get('Episode_Termination/base_contact', 0.0)) > 0
                    fall_flags.append(fell)
                    if fell:
                        fall_count += 1

                    rewards.append(current_episode_reward)
                    episode_lengths.append(step_count)
                    print(f"[{ckpt_path}] Episode {len(rewards)}/{args_cli.num_episodes} | "
                          f"Reward: {current_episode_reward:.2f} | Length: {step_count} steps | Fell: {fell}")

                    clear_push()
                    push_active = False
                    step_count = 0  
                    current_episode_reward = 0.0
                    obs, _ = env.reset()

        summary[ckpt_path] = {
            "avg_reward": sum(rewards) / len(rewards),
            "avg_length": sum(episode_lengths) / len(episode_lengths),
            "fall_count": fall_count,
            "fall_rate": fall_count / len(rewards),
        }
        print(f"--> Summary: Avg Reward = {summary[ckpt_path]['avg_reward']:.4f}, "
              f"Avg Length = {summary[ckpt_path]['avg_length']:.1f} steps, "
              f"Fall Rate = {summary[ckpt_path]['fall_rate']:.2f}")
              
        break

    print(f"\n{'='*60}\nEVALUATION SUMMARY ({args_cli.task})\n{'='*60}")
    print(f"Push Force: {args_cli.push_force} N | Interval: {args_cli.push_interval} steps | Duration: {args_cli.push_duration} steps\n")
    for ckpt, metrics in summary.items():
        print(f"{ckpt}\n -> Avg Reward: {metrics['avg_reward']:.4f} | Avg Length: {metrics['avg_length']:.1f} steps | Fall Rate: {metrics['fall_rate']:.2f} ({metrics['fall_count']}/{args_cli.num_episodes})\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
