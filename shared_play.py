import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium
import time
import numpy as np
#from isaacsim.sensors.physics.scripts.effort_sensor import EffortSensor

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


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

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-H1-Play-v0", num_envs=1, 
#headless=True
#cli_args=["--seed", "42"]
)
env = wrap_env(env)
print("number of env: ",env.num_envs)
device = env.device

# effort sensor
#sensor = EffortSensor(
#    prim_path="/World/envs/env_0/Robot/joints/left_hip_pitch",  # Adjust path as needed for your robot's joints
#    sensor_period=0.1,  # Sample rate (in seconds)
#    use_latest_data=True,  # Use the latest data
#    enabled=True  # Ensure sensor is enabled
#)

num_envs = 1
#print(f"Number of Env: {env.num_envs}, {type(env.num_envs)}") # 1
device = torch.device("cuda:0")
#print(f"Env Device: {env.device}, {type(env.device)}") # cuda:0
action_space = gymnasium.spaces.box.Box(low=float("-inf"), high=float("inf"), shape=(19,))
#print(f"Env Action Space: {env.action_space}, {type(env.action_space)}") # Box(-inf, inf, (19,), float32)
observation_space = gymnasium.spaces.box.Box(low=float("-inf"), high=float("inf"), shape=(69,))
#print(f"Env Observation Space: {env.observation_space}, {type(env.observation_space)}") # Box(-inf, inf, (69,), float32)

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=32, num_envs=num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models

models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model



# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 32  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8  # 32 * 1024 / 4096
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
cfg["mixed_precision"] = False
# logging to TensorBoard and write checkpoints (in timesteps)
#cfg["experiment"]["write_interval"] = 80
#cfg["experiment"]["checkpoint_interval"] = 800
#cfg["experiment"]["directory"] = "runs/Isaac-Velocity-Flat-H1-v0"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
            device=device)

checkpoint_path = "shared_agent.pt" # non-quantized model
#checkpoint_path = "quantized_agent.pt" # quantized
agent.load(checkpoint_path)


# Number of episodes you want to run
num_episodes = 10
max_steps = 200
total_rewards = []
vel_x = []
total_energy = []

print(f"Using {checkpoint_path} checkpoints")
start_runtime = time.time()
inference_time = 0

# hz speed
loop_count = 0
start = time.perf_counter()

for episode_id in range(num_episodes):
	obs, _ = env.reset()
	episode_reward = 0
	energy_estimates = []
	for i in range(max_steps):
		with torch.inference_mode():
			start_time = time.time()
			outputs = agent.act(obs, timestep=0, timesteps=0)
			end_time = time.time()
			inference_time += end_time - start_time
			actions = outputs[-1].get("mean_actions", outputs[0])
			obs, reward, terminated, truncated, info = env.step(actions)

			# hz count
			loop_count += 1
			if loop_count % 100 == 0:
				now = time.perf_counter()
				hz = 100 / (now - start)
				print(f"Control rate: {hz:.2f} Hz")
				start = now

			vel_x.append(obs[0][0].tolist())
			
			# energy
			joint_velocities = obs[0][31:50].tolist()
			kinetic_energy = np.sum([v**2 for v in joint_velocities])
			#print(kinetic_energy)
			energy_estimates.append(kinetic_energy)

			#print(reward)
			episode_reward += reward
			if terminated or truncated:
				obs, _ = env.reset()
	#print("Energy consumption:", sum(energy_estimates))
	print(f"Episode {episode_id}: {episode_reward}")
	total_rewards.append(episode_reward)
	total_energy.append(sum(energy_estimates))

end_runtime = time.time()
time_taken = end_runtime - start_runtime
print(f"Time taken to complete the process: {time_taken} seconds")
print(f"Time taken to complete the inference: {inference_time} seconds")

# energy consumption
energy_mean = np.mean(total_energy)
energy_std = np.std(total_energy)
energy_coefficient_of_variation = energy_std / energy_mean
print(f"Energy consumption mean: {energy_mean}")
print(f"Energy consumption std: {energy_std}")
print(f"Energy consumption coefficient: {energy_coefficient_of_variation}")

vel_x_array = np.array(vel_x)

steady_start = int(0.2 * len(vel_x))
steady_state_vel = vel_x_array[steady_start:]

print(f"Overall average speed: {np.mean(vel_x_array):.3f} m/s")
print(f"Steady-state average: {np.mean(steady_state_vel):.3f} m/s")
print(f"Peak speed: {np.max(vel_x_array):.3f} m/s")
print(f"Speed std dev: {np.std(steady_state_vel):.3f} m/s")

# Calculate the average reward over all episodes
average_reward = sum(total_rewards) / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")
env.close()
print("Simulation completed.")

#print(sensor)
