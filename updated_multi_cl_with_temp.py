from dotenv import load_dotenv
import os
load_dotenv()

import sys
import time
JTOP_PATH = "/usr/local/lib/python3.6/dist-packages"
if JTOP_PATH not in sys.path:
    sys.path.insert(0, JTOP_PATH)

from datetime import datetime
import torch
import torch.nn as nn
import wandb
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


# ---------------------------------------------------------------- jtop logger
class JetsonLogger:
    """Reads jtop stats (temps, power, GPU, RAM...) and logs them to the active wandb run."""

    def __init__(self, interval_s=5.0):
        self.interval_s = interval_s
        self.last = 0.0
        self.jetson = None
        try:
            from jtop import jtop
            self.jetson = jtop()
            self.jetson.start()          # keeps one connection open for the whole run
        except Exception as e:
            print("[JetsonLogger] jtop unavailable, disabled:", e)
            self.jetson = None
            return
        if wandb.run is not None:
            wandb.define_metric("jetson/step")
            wandb.define_metric("jetson/*", step_metric="jetson/step")

    @staticmethod
    def _name(key):
        return "jetson/" + key.lower().replace(" ", "_")

    def log(self, timestep):
        if self.jetson is None or wandb.run is None:
            return
        now = time.time()
        if now - self.last < self.interval_s:
            return
        self.last = now
        if not self.jetson.ok():
            return
        data = {"jetson/step": timestep}
        for key, val in self.jetson.stats.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue                  # skips datetime, timedelta and 'OFF'/'ON' strings
            if key.startswith("Power"):
                data[self._name(key) + "_W"] = val / 1000.0   # jtop reports mW
            else:
                data[self._name(key)] = val
        wandb.log(data)

    def close(self):
        if self.jetson is not None:
            try:
                self.jetson.close()
            except Exception:
                pass
# ---------------------------------------------------------------------------


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "10.42.0.1"
port = 5000
client_socket.connect((host, port))


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
            self._shared_output = None
            return self.value_layer(shared_output), {}

# start
send_msg(client_socket, {"stage": 'init'})
response = recv_msg(client_socket)

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
log_dir = f"jetson_cl_{task_name}_{arch}_nogui_seed_{seed}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
        "write_interval": 100,          # explicit, so points appear often
        "checkpoint_interval": 1000,
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
    path = f"./checkpoints/{arch}_stair_terrain/best_agent.pt"
agent.load(path)

total_timesteps = num_episodes * max_steps

agent.init(trainer_cfg={"timesteps": total_timesteps})

# >>> NEW: start jetson logger AFTER agent.init (wandb run exists now)
jlog = JetsonLogger(interval_s=5.0)

# reset
send_msg(client_socket, {"state": 'reset'})
obs_tensor = torch.tensor(recv_msg(client_socket)["obs"], dtype=torch.float32, device=device)

for timestep in range(total_timesteps):
    # act
    with torch.no_grad():
        action_tensor = agent.act(obs_tensor, timestep=timestep, timesteps=total_timesteps)[0]

    # step
    send_msg(client_socket, {"state": "step", "action": action_tensor.cpu().tolist()})
    response = recv_msg(client_socket)

    next_obs_tensor   = torch.tensor(response["next_obs"],   dtype=torch.float32, device=device)
    rewards_tensor    = torch.tensor(response["rewards"],    dtype=torch.float32, device=device).view(-1, 1)
    terminated_tensor = torch.tensor(response["terminated"], dtype=torch.bool,    device=device).view(-1, 1)
    truncated_tensor  = torch.tensor(response["truncated"],  dtype=torch.bool,    device=device).view(-1, 1)

    # store
    with torch.no_grad():
        agent.record_transition(
            states=obs_tensor, actions=action_tensor, rewards=rewards_tensor,
            next_states=next_obs_tensor, terminated=terminated_tensor,
            truncated=truncated_tensor, infos={},
            timestep=timestep, timesteps=total_timesteps,
        )
    agent.post_interaction(timestep=timestep, timesteps=total_timesteps)

    # >>> NEW: log temperature / power (rate-limited to once per interval_s)
    jlog.log(timestep)

    obs_tensor = next_obs_tensor


# >>> NEW: close jtop before shutting down
jlog.close()

# Close the socket
send_msg(client_socket, {"state": "stop"})
response = recv_msg(client_socket)
client_socket.close()

path = f"./checkpoints/CL/multi_{arch}_{task_name}.pt"
agent.save(path)
