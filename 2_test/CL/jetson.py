
from datetime import datetime
import torch
import torch.nn as nn
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils import set_seed
from skrl.agents.torch.ppo import PPO
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

set_seed(42)

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


device = 'cuda:0'
log_dir = "runs/"
seed = 42
task_name = "flat_terrain"
action_space = 19
observation_space = 69
gui = "nogui"
log_dir = f"{task_name}_DAC_{gui}_seed_{seed}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


models = {
    "policy": Policy(69, 19, device),
    "value": Value(69, 19, device)
}


agent_cfg = {}
agent_cfg["agent"] = {}
agent_cfg["agent"]["class"] = {}
agent_cfg["agent"]["class"] = {'seed': 42, 'models': {'separate': False, 'policy': {'class': 'GaussianMixin', 'clip_actions': False, 'clip_log_std': True, 'min_log_std': -20.0, 'max_log_std': 2.0, 'initial_log_std': 0.0, 'network': [{'name': 'net', 'input': 'STATES', 'layers': [128, 128, 128], 'activations': 'elu'}], 'output': 'ACTIONS'}, 'value': {'class': 'DeterministicMixin', 'clip_actions': False, 'network': [{'name': 'net', 'input': 'STATES', 'layers': [128, 128, 128], 'activations': 'elu'}], 'output': 'ONE'}}, 'memory': {'class': 'RandomMemory', 'memory_size': -1}, 'agent': {'class': 'PPO', 'rollouts': 24, 'learning_epochs': 5, 'mini_batches': 4, 'discount_factor': 0.99, 'lambda': 0.95, 'learning_rate': 0.001, 'learning_rate_scheduler': 'KLAdaptiveLR', 'learning_rate_scheduler_kwargs': {'kl_threshold': 0.01}, 'state_preprocessor': None, 'state_preprocessor_kwargs': None, 'value_preprocessor': None, 'value_preprocessor_kwargs': None, 'random_timesteps': 0, 'learning_starts': 0, 'grad_norm_clip': 1.0, 'ratio_clip': 0.2, 'value_clip': 0.2, 'clip_predicted_values': True, 'entropy_loss_scale': 0.01, 'value_loss_scale': 1.0, 'kl_threshold': 0.0, 'rewards_shaper_scale': 1.0, 'time_limit_bootstrap': False, 'experiment': {'directory': 'h1_flat', 'experiment_name': '', 'write_interval': 'auto', 'checkpoint_interval': 'auto'}}, 'trainer': {'class': 'SequentialTrainer', 'timesteps': 24000, 'environment_info': 'log', 'close_environment_at_exit': False}}


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
    "value_loss_scale": 4.0,
    "kl_threshold": 0,
    "rewards_shaper": lambda rewards, *args, **kwargs: rewards * 0.01,
    "time_limit_bootstrap": False,
    "state_preprocessor": RunningStandardScaler,
    "state_preprocessor_kwargs": {"size": observation_space, "device": device},
    "value_preprocessor": RunningStandardScaler,
    "value_preprocessor_kwargs": {"size": 1, "device": device},
    "mixed_precision": False,
    "experiment": {
        "directory": "",
        "experiment_name": log_dir,
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "wandb": False,
        "wandb_kwargs": {
            "project": "Shared-Arch-v1",
            "name": log_dir,
        }
    }
}


memory = RandomMemory(memory_size=32, num_envs=1, device=device)
agent = PPO(models=models, memory=memory, cfg=ppo_agent_cfg, observation_space=observation_space, action_space=action_space, device=device)

obs = torch.tensor([[ 8.0038e-02,  3.7939e-02,  1.8159e-03, -1.3222e-01, -2.2559e-02,
          1.3382e-01, -2.4859e-02, -3.4354e-03, -9.6073e-01,  5.2944e-01,
          0.0000e+00,  3.2297e-02,  3.6842e-03,  1.1852e-03, -8.5261e-03,
          5.3355e-03, -4.5503e-03,  2.2918e-03, -9.6369e-03,  5.5492e-03,
         -5.1787e-04,  8.2180e-03, -7.7221e-03,  6.2474e-03, -4.0504e-03,
         -9.4803e-04,  1.6240e-03, -4.9606e-03, -7.8228e-03, -8.7285e-03,
         -9.8892e-03, -1.3291e+00,  2.0916e-01, -7.4265e-01,  7.3148e-01,
         -7.1611e-01,  1.3771e+00, -6.7941e-02, -1.1931e+00, -1.1310e-01,
          3.8384e-01, -3.6111e-01,  1.1526e+00,  2.3516e-01,  7.0527e-01,
          4.2649e-01, -5.9539e-01, -9.2568e-01, -3.7427e-01,  3.9634e-01,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]], device='cuda:0')

timesteps = 5
rewards = []
next_obs = obs
terminated = 1
truncated = 0
info = []

for timestep in range(timesteps):
    # Optional pre-interaction hook (e.g. for random exploration warmup)
    agent.pre_interaction(timestep=timestep, timesteps=50000)
    
    # 2. Compute action
    # agent.act returns a tuple: (actions, log_prob, outputs)
    actions, _, _ = agent.act(obs, timestep=timestep, timesteps=timesteps)
    
    # 3. Step the environment
#    next_obs, rewards, terminated, truncated, info = env.step(actions)
#    dones = terminated | truncated # or standard gym formatting
    
    # 4. Record the transition into agent's memory
    agent.record_transition(
        states=obs,
        actions=actions,
        rewards=rewards,
        next_states=next_obs,
        terminated=terminated,
        truncated=truncated,
        infos=info,
        timestep=timestep,
        timesteps=timesteps
    )
    
    # 5. Perform learning updates (gradient descent, etc.)
    agent.post_interaction(timestep=timestep, timesteps=50000)
    
    # Move to next observation
    obs = next_obs

