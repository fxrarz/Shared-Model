# Usage examples:
# python3 loss_scaling.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --value_loss_scale 1.0
# python3 loss_scaling.py --task Isaac-Stairway-H1-v0 --headless --seed 42 --value_loss_scale 4.0

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with skrl for UAC Loss Ablation.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for environment and training.")
parser.add_argument("--value_loss_scale", type=float, default=4.0, help="Value loss scale coefficient (c1)")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import random
from datetime import datetime
from packaging import version
import torch
import torch.nn as nn
import skrl
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils import set_seed
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.agents.torch.ppo import PPO

# Check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ELU(),
            nn.Linear(400, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU()
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
            # Scaled tanh action mapping matching benchmark script
            mean = 2 * torch.tanh(self.mean_layer(self._shared_output))
            return mean, self.log_std_parameter, {}
            
        elif role == "value":
            # Safe caching check: verify tensor exists and batch size matches
            if self._shared_output is not None and self._shared_output.shape[0] == inputs["states"].shape[0]:
                shared_output = self._shared_output
            else:
                shared_output = self.net(inputs["states"])
            self._shared_output = None
            return self.value_layer(shared_output), {}

task = args_cli.task
task_name = "flat_terrain" if "Flat" in task else "stair_terrain"
gui = "nogui" if args_cli.headless else "gui"
v_scale = args_cli.value_loss_scale
log_dir = f"ablation_1_{task_name}_{gui}_UAC_vscale_{v_scale}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_seed_{args_cli.seed}"

@hydra_task_config(task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    # Set seed correctly from CLI args
    set_seed(args_cli.seed)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    device = env.device

    ppo_agent_cfg = {
        "class": agent_cfg["agent"]["class"],
        "rollouts": 32,
        "learning_epochs": 8,
        "mini_batches": 8,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "learning_rate_scheduler": KLAdaptiveRL,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.0,
        "value_loss_scale": args_cli.value_loss_scale,
        "kl_threshold": 0,
        "rewards_shaper": lambda rewards, *args, **kwargs: rewards * 0.01,
        "time_limit_bootstrap": False,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "mixed_precision": False,
        "experiment": {
            "directory": "runs/ablation",
            "experiment_name": log_dir,
            "write_interval": "auto",
            "checkpoint_interval": "auto",
            "wandb": True,
            "wandb_kwargs": {
                "project": "Shared-Arch-v1",
                "name": log_dir,
            }
        }
    }

    agent_cfg["agent"] = ppo_agent_cfg
    print(f"Running Ablation Experiment: {log_dir}")

    models = {}
    models["policy"] = Shared(env.observation_space, env.action_space, device)
    models["value"] = models["policy"]

    memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=device)
    agent = PPO(models=models, memory=memory, cfg=ppo_agent_cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)

    trainer = SequentialTrainer(cfg=agent_cfg['trainer'], env=env, agents=agent)
    trainer.train()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
