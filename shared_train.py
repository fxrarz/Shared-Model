# /home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --headless --seed 42

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

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
set_seed(42)

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml, load_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 400),
                                 nn.ELU(),
                                 nn.Linear(400, 200),
                                 nn.ELU(),
                                 nn.Linear(200, 100),
                                 nn.ELU())
        self.mean_layer = nn.Linear(100, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.value_layer = nn.Linear(100, 1)
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


task = "Isaac-Velocity-Flat-H1-v0"
@hydra_task_config(task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # same as: `wrap_env(env, wrapper="auto")`
    device = env.device
    
    ppo_agent_cfg = {
		"class": agent_cfg["agent"]["class"],  # agent name
		"rollouts": 32,  # memory_size
		"learning_epochs": 8,
		"mini_batches": 8,  # 32 * 1024 / 4096
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
		"value_loss_scale": 4.0,
		"kl_threshold": 0,
		"rewards_shaper": lambda rewards, *args, **kwargs: rewards * 0.01,
		"time_limit_bootstrap": False,
		"state_preprocessor": RunningStandardScaler,
		"state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
		"value_preprocessor": RunningStandardScaler,
		"value_preprocessor_kwargs": {"size": 1, "device": device},
		"mixed_precision": False,
		"experiment": agent_cfg["agent"]["experiment"]
	}
	
    log_dir = "UAC_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_seed_{args_cli.seed}"
    agent_cfg["agent"] = ppo_agent_cfg
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    print(f"Running {log_dir}")

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
