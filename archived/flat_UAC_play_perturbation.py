import sys
import os
import re
import glob
import math
import pickle
import torch
import torch.nn as nn
import gymnasium

# --- Hardcoded Perturbation & Evaluation Settings ---
PUSH_FORCE = 100.0        # Magnitude of external push force (N)
PUSH_INTERVAL = 10        # Simulation steps between pushes (frequency)
PUSH_DURATION = 10        # Simulation steps the push force is held for (time interval)
PUSH_BODY_NAME = None     # None defaults to root body (e.g., 'torso_link' or 'pelvis')
NUM_EPISODES = 1         # Number of episodes per checkpoint

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed

# Seed for reproducibility
set_seed(42)

def parse_folder_name(name):
    match = re.match(r"(.*?)_nogui_(.*?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_seed_(\d+)(.*)", name)
    if match:
        terrain, arch, timestamp, seed, archived = match.groups()
        return (terrain, arch, int(seed), timestamp, archived)
    return (name, "", 0, "", "")

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


def main():
    # Load and wrap the Isaac Lab environment safely
    env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-H1-Play-v0", num_envs=1)
    env = wrap_env(env)

    num_envs = 1
    device = torch.device("cuda:0")

    action_space = gymnasium.spaces.box.Box(low=float("-inf"), high=float("inf"), shape=(19,))
    observation_space = gymnasium.spaces.box.Box(low=float("-inf"), high=float("inf"), shape=(69,))

    # Instantiate memory & models
    memory = RandomMemory(memory_size=32, num_envs=num_envs, device=device)
    models = {}
    models["policy"] = Shared(env.observation_space, env.action_space, device)
    models["value"] = models["policy"]

    # Configure PPO agent
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

    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]
    
    if PUSH_BODY_NAME is not None:
        body_ids, body_names = robot.find_bodies(PUSH_BODY_NAME)
        if len(body_ids) == 0:
            raise ValueError(f"No bodies matched '{PUSH_BODY_NAME}'. Available: {robot.body_names}")
        push_body_ids = body_ids
        print(f"Pushing target bodies: {body_names}")
    else:
        push_body_ids = [0]
        print(f"Pushing default root body: {robot.body_names[0]}")

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

    # Find checkpoints
    folder_list = glob.glob(os.path.join("runs/", "*UAC*"))
    sorted_categorically = sorted(folder_list, key=parse_folder_name)
    
    summary = {}

    for ckpt_folder in sorted_categorically:
        ckpt_path = f"{ckpt_folder}/checkpoints/best_agent.pt"
        print(f"\n{'='*50}\nLoading Checkpoint: {ckpt_path}\n{'='*50}")
        try:
            agent.load(ckpt_path)
        except Exception as e:
            print(f"Failed to load checkpoint '{ckpt_path}': {e}")
            continue

        rewards = []
        episode_lengths = []
        reward_split = {
            'Episode_Reward/track_lin_vel_xy_exp': [], 'Episode_Reward/track_ang_vel_z_exp': [], 
            'Episode_Reward/ang_vel_xy_l2': [], 'Episode_Reward/dof_torques_l2': [], 
            'Episode_Reward/dof_acc_l2': [], 'Episode_Reward/action_rate_l2': [], 
            'Episode_Reward/feet_air_time': [], 'Episode_Reward/flat_orientation_l2': [], 
            'Episode_Reward/dof_pos_limits': [], 'Episode_Reward/termination_penalty': [], 
            'Episode_Reward/feet_slide': [], 'Episode_Reward/joint_deviation_hip': [], 
            'Episode_Reward/joint_deviation_arms': [], 'Episode_Reward/joint_deviation_torso': [], 
            'Metrics/base_velocity/error_vel_xy': [], 'Metrics/base_velocity/error_vel_yaw': [], 
            'Episode_Termination/time_out': [], 'Episode_Termination/base_contact': []
        }
        
        fall_count = 0
        fall_flags = []
        step_count = 0
        push_active = False
        push_end_step = -1

        obs, _ = env.reset()
        current_episode_reward = 0.0

        while len(rewards) < NUM_EPISODES:
            if not push_active and step_count > 0 and step_count % PUSH_INTERVAL == 0:
                angle = torch.rand(1).item() * 2 * math.pi
                force_vec = torch.tensor(
                    [PUSH_FORCE * math.cos(angle),
                     PUSH_FORCE * math.sin(angle),
                     0.0],
                    device=device,
                )
                apply_push(force_vec)
                push_active = True
                push_end_step = step_count + PUSH_DURATION

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

                # Episode completes strictly via env termination or truncation
                done = terminated or truncated
                
                log_data = info.get("log", {})
                for k in reward_split.keys():
                    if k in log_data:
                        reward_split[k].append(log_data[k])

                if done:
                    fell = float(log_data.get('Episode_Termination/base_contact', 0.0)) > 0
                    fall_flags.append(fell)
                    if fell:
                        fall_count += 1

                    rewards.append(current_episode_reward)
                    episode_lengths.append(step_count)
                    print(f"[{ckpt_path}] Episode {len(rewards)}/{NUM_EPISODES} finished | Reward: {current_episode_reward:.2f} | Length: {step_count} steps | Fell: {fell}")

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
            "fall_flags": fall_flags
        }
        print(f"--> Summary for {ckpt_path}: Avg Reward = {summary[ckpt_path]['avg_reward']:.4f}, Avg Length = {summary[ckpt_path]['avg_length']:.1f} steps, Fall Rate = {summary[ckpt_path]['fall_rate']:.2f}")

    print(f"\n{'='*50}\nPERTURBATION EVALUATION SUMMARY\n{'='*50}")
    print(f"Push Force: {PUSH_FORCE} N | Interval (Freq): {PUSH_INTERVAL} steps | Duration (Time): {PUSH_DURATION} steps\n")
    for ckpt, metrics in summary.items():
        print(f"{ckpt} \n -> Avg Reward: {metrics['avg_reward']:.4f} | Avg Length: {metrics['avg_length']:.1f} steps | Fall Rate: {metrics['fall_rate']:.2f} ({metrics['fall_count']}/{NUM_EPISODES})\n")

    env.close()
    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()
