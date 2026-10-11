# python3 2_test/CL/GPU.py --task Isaac-Velocity-Flat-H1-v0 --headless --num_envs 1
# python3 2_test/CL/GPU.py --task Isaac-Slope-H1-v0 --headless --num_envs 1 --num_episodes 25 --max_steps 1000 --arch dac
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
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", 5000))
server_socket.listen(1)


#total_rewards = []
seed = args_cli.seed
task = args_cli.task
arch = args_cli.arch
num_episodes = args_cli.num_episodes
max_steps = args_cli.max_steps
num_envs = args_cli.num_envs

#if args_cli.task == "Isaac-Velocity-Flat-H1-v0":
#    task_name = "flat_terrain" 
#if args_cli.task == "Isaac-Stairway-H1-v0":
#    task_name = "stair_terrain"
if args_cli.task == "Isaac-Slope-H1-v0":
    task_name = "slope_terrain"
if args_cli.task == "Isaac-Box-H1-v0":
    task_name = "box_terrain"

#task_name = "flat_terrain" if args_cli.task == "Isaac-Velocity-Flat-H1-v0" else "stair_terrain"

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
        raw_data = client_socket.recv(4096)
#        print("outside raw_data")
        if raw_data:
#            print("inside raw_data")
            data = json.loads(raw_data.decode("utf-8"))
            if data.get('stage') == 'init':
                print("stage init")
                response_data = {"action_space": env.action_space.shape[0], "arch": arch, "device": str(device), "max_steps": max_steps, "num_envs": num_envs, "num_episodes": num_episodes, "observation_space": env.observation_space.shape[0], "seed": seed, "task": task, "task_name": task_name}
                client_socket.sendall(json.dumps(response_data).encode("utf-8"))

            if data.get('state') == 'reset':
                print("state reset")
                obs, _ = env.reset()
                response_data = {"obs": obs.cpu().tolist()}
                #print(response_data)
                client_socket.sendall(json.dumps(response_data).encode("utf-8"))

            if data.get('state') == 'step':
#                print("state step")
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

#                response_data = {"next_obs": next_obs, 'rewards': rewards, 'terminated': terminated, 'truncated': truncated, 'infos': infos}
                #print(response_data)
                client_socket.sendall(json.dumps(response_data).encode("utf-8"))

            if data.get('state') == 'stop':
                print("state stop")
                response_data = {"status": "end"}
                client_socket.sendall(json.dumps(response_data).encode("utf-8"))
                env.close()
                break
                
                                 
'''
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
            average_reward = sum(total_rewards) / num_episodes
            print(f"Average reward over {num_episodes} episodes: {average_reward}")
            env.close()
            print("Simulation completed.")
'''

'''
    for episode_id in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for i in range(max_steps):
            obs = obs.squeeze().tolist()
            response = requests.post(url, json={"obs":obs}).json()
            actions = response["actions"]
            actions = torch.tensor(actions, device=device)
            obs, reward, terminated, truncated, info = env.step(actions)
            print(reward)
            episode_reward += reward
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"Episode {episode_id}: {episode_reward}")
        total_rewards.append(episode_reward)
'''

if __name__ == "__main__":
    main()
    simulation_app.close()
