from dotenv import load_dotenv
import os
load_dotenv() 

from datetime import datetime
import torch
import torch.nn as nn
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils import set_seed
from skrl.agents.torch.ppo import PPO
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

import json
import socket
import struct

def send_msg(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)

def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf

def recv_msg(sock):
    (length,) = struct.unpack("!I", recvall(sock, 4))
    return json.loads(recvall(sock, length).decode("utf-8"))

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Create a TCP/IP socket
host = "10.42.0.1"
port = 5000
client_socket.connect((host, port)) # Connect to the server


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
            mean = 2 * torch.tanh(self.mean_layer(self._shared_output))
            return mean, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is not None and self._shared_output.shape[0] == inputs["states"].shape[0]:
                shared_output = self._shared_output
            else:
                shared_output = self.net(inputs["states"])
            self._shared_output = None  # Safely reset cache
            return self.value_layer(shared_output), {}

# start
send_msg(client_socket, {"stage": 'init'}) # send
response = recv_msg(client_socket) # rec

action_space = response['action_space']
arch = response['arch']
device = response['device']
max_steps = response['max_steps']
num_envs = response['num_envs']
num_episodes = response['num_episodes']
observation_space = response['observation_space']
seed = response['seed']
task = response['task']
task_name = response['task_name']

set_seed(seed)
log_dir = f"cl_{task_name}_{arch}_nogui_seed_{seed}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if arch == "dac":
    models = {
        "policy": Policy(observation_space, action_space, device),
        "value": Value(observation_space, action_space, device)
    }

if arch == "uac":
    models = {}
    models["policy"] = Shared(observation_space, action_space, device)
    models["value"] = models["policy"]

ppo_agent_cfg = {
    "class": "PPO",
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
        "wandb": True,
        "wandb_kwargs": {
            "project": "Shared-Arch-v1",
            "name": log_dir,
        }
    }
}

memory = RandomMemory(memory_size=32, num_envs=num_envs, device=device)
agent = PPO(models=models, memory=memory, cfg=ppo_agent_cfg, observation_space=observation_space, action_space=action_space, device=device)

if task_name == "slope_terrain":
    path = f"./checkpoints/{arch}_flat_terrain/best_agent.pt"
if task_name == "box_terrain":
    path = f"./checkpoints/{arch}_box_terrain/best_agent.pt"
agent.load(path)


total_timesteps = num_episodes * max_steps
rollouts_target = ppo_agent_cfg["rollouts"] # e.g., 32 steps per update block

agent.init(trainer_cfg={"timesteps": total_timesteps})
print("total_timesteps:", total_timesteps)
print("write_interval:", agent.write_interval, "checkpoint_interval:", agent.checkpoint_interval)

for episode in range(num_episodes):
    # reset
    send_msg(client_socket, {"state": 'reset'}) # send
    response = recv_msg(client_socket) # rec
    obs = response['obs']
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    episode_reward = 0
    for step in range(max_steps):
        current_timestep = episode * max_steps + step
        if current_timestep%100 == 0:
            print(current_timestep)
        # act
        with torch.no_grad():
            outputs = agent.act(obs_tensor, timestep=current_timestep, timesteps=total_timesteps)
            action_tensor = outputs[0]#.detach()
        action_list = action_tensor.cpu().tolist()
        # step
        send_msg(client_socket, {"state": "step", "action": action_list}) # send
        response = recv_msg(client_socket) # rec
        next_obs = response["next_obs"]
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
        rewards_tensor    = torch.tensor(response["rewards"],    dtype=torch.float32, device=device).view(-1, 1)
        terminated_tensor = torch.tensor(response["terminated"], dtype=torch.bool,    device=device).view(-1, 1)
        truncated_tensor  = torch.tensor(response["truncated"],  dtype=torch.bool,    device=device).view(-1, 1)
        infos = response["infos"]
        # update
        with torch.no_grad():
            agent.record_transition(
                states=obs_tensor.detach(), 
                actions=action_tensor.detach(), 
                rewards=rewards_tensor.detach(), 
                next_states=next_obs_tensor.detach(), 
                terminated=terminated_tensor.detach(), 
                truncated=truncated_tensor.detach(), 
                infos=infos,
                timestep=current_timestep,
                timesteps=total_timesteps
            )
        agent.post_interaction(
            timestep=current_timestep, 
            timesteps=total_timesteps
        )
        # Advance observation tensor
        obs_tensor = next_obs_tensor#.detach().clone()
        if terminated_tensor.any() or truncated_tensor.any():
            print("episode ended at step", step, "tracked rewards:", len(agent._track_rewards))
            #print("break triggered")
            break

# Close the socket
send_msg(client_socket, {"state": "stop"}) # send
response = recv_msg(client_socket) # rec
client_socket.close()

path = f"./checkpoints/CL/multi_{arch}_{task_name}.pt"
agent.save(path)
