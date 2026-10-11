# python3 2_test/CL/multi_agent_sim.py --task Isaac-Velocity-Flat-H1-v0 --headless --num_envs 1
# python3 2_test/CL/multi_agent_sim.py --task Isaac-Slope-H1-v0 --headless --num_envs 1 --num_episodes 25 --max_steps 1000 --arch dac
import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
#parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Slope-H1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--arch", type=str, default="dac", help="Architecture of the policy")
parser.add_argument("--num_episodes", type=int, default=2, help="Number of Episodes")
parser.add_argument("--max_steps", type=int, default=100, help="Length of Episodes")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from skrl.utils import set_seed
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import torch

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
    
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", 5000))
server_socket.listen(1)

seed = args_cli.seed
task = args_cli.task
arch = args_cli.arch
num_episodes = args_cli.num_episodes
max_steps = args_cli.max_steps
num_envs = args_cli.num_envs

if args_cli.task == "Isaac-Slope-H1-v0":
    task_name = "slope_terrain"
if args_cli.task == "Isaac-Box-H1-v0":
    task_name = "box_terrain"

def recursive_to_cpu(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item() if val.numel() == 1 else val.detach().cpu().tolist()
    elif isinstance(val, dict):
        return {k: recursive_to_cpu(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [recursive_to_cpu(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(recursive_to_cpu(v) for v in val)
    return val

@hydra_task_config(task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    set_seed(seed)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    device = env.device
    
    client_socket, client_address = server_socket.accept()
    print(f"Connected by {client_address}")

    while 1:
        data = recv_msg(client_socket)
        if data:
            if data.get('stage') == 'init':
                print("state init")
                response_data = {"action_space": env.action_space.shape[0], "arch": arch, "device": str(device), "max_steps": max_steps, "num_envs": num_envs, "num_episodes": num_episodes, "observation_space": env.observation_space.shape[0], "seed": seed, "task": task, "task_name": task_name}
                send_msg(client_socket, response_data)

            if data.get('state') == 'reset':
                print("state reset")
                obs, _ = env.reset()
                response_data = {"obs": obs.cpu().tolist()}
                send_msg(client_socket, response_data)

            if data.get('state') == 'step':
                print("state step")
                action_list = data['action']
                action_tensor = torch.tensor(action_list, dtype=torch.float32, device=device) 
                next_obs, rewards, terminated, truncated, infos = env.step(action_tensor)
                
                response_data = {
                    "next_obs": next_obs.detach().cpu().tolist(), 
                        "rewards": recursive_to_cpu(rewards), 
                        "terminated": recursive_to_cpu(terminated), 
                        "truncated": recursive_to_cpu(truncated), 
                        "infos": recursive_to_cpu(infos)
                }
                send_msg(client_socket, response_data)

            if data.get('state') == 'stop':
                print("state stop")
                response_data = {"status": "end"}
                send_msg(client_socket, response_data)
                env.close()
                break
                
if __name__ == "__main__":
    main()
    simulation_app.close()
