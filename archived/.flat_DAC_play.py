import argparse
import gymnasium
import torch
import torch.nn as nn
import re, os, glob

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
    # Extracts terrain, architecture, timestamp, and seed
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
    parser = argparse.ArgumentParser(description="Evaluate multiple agent checkpoints in Isaac Lab.")
    parser.add_argument(
        "--checkpoints", "-c", 
        nargs="+", 
        required=False, 
        help="List of path(s) to checkpoint .pt files"
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per checkpoint")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    args = parser.parse_args()

    # Load and wrap the Isaac Lab environment once
    env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-H1-Play-v0", num_envs=1)
    env = wrap_env(env)

    num_envs = 1
    device = torch.device("cuda:0")

    action_space = gymnasium.spaces.box.Box(low=float("-inf"), high=float("inf"), shape=(19,))
    observation_space = gymnasium.spaces.box.Box(low=float("-inf"), high=float("inf"), shape=(69,))

    # Instantiate memory
    memory = RandomMemory(memory_size=32, num_envs=num_envs, device=device)

    # Instantiate models
    models = {
        "policy": Policy(observation_space, action_space, device),
        "value": Value(observation_space, action_space, device)
    }

    # Configure and instantiate PPO agent
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

    # Store results for summary
    summary = {}

    # Loop over each checkpoint argument
    folder_list = glob.glob(os.path.join("runs/", "flat_terrain_*_DAC*"))
    sorted_categorically = sorted(folder_list, key=parse_folder_name)
    
    for ckpt in sorted_categorically:
        ckpt = f"{ckpt}/checkpoints/best_agent.pt"
        print(f"\n{'='*50}\nLoading Checkpoint: {ckpt}\n{'='*50}")
        try:
            agent.load(ckpt)
        except Exception as e:
            print(f"Failed to load checkpoint '{ckpt}': {e}")
            continue

        total_rewards = []

        for episode_id in range(args.episodes):
            obs, _ = env.reset()
            episode_reward = 0.0

            for _ in range(args.max_steps):
                with torch.inference_mode():
                    outputs = agent.act(obs, timestep=0, timesteps=0)
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    obs, reward, terminated, truncated, info = env.step(actions)

                    step_reward = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                    episode_reward += step_reward

                    if terminated or truncated:
                        obs, _ = env.reset()

            print(f"[{ckpt}] Episode {episode_id + 1}/{args.episodes}: Reward = {episode_reward:.4f}")
            total_rewards.append(episode_reward)

        avg_reward = sum(total_rewards) / len(total_rewards)
        summary[ckpt] = avg_reward
        print(f"--> Average Reward for {ckpt}: {avg_reward:.4f}")

    # Final Summary Table
    print(f"\n{'='*50}\nEVALUATION SUMMARY\n{'='*50}")
    for ckpt, avg in summary.items():
        print(f"{ckpt:<40} | Avg Reward: {avg:.4f}")

    env.close()
    print("\nSimulation completed.")


if __name__ == "__main__":
    main()
