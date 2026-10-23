import os
import time
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import argparse


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Evaluate Computation Footprint."
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-H1-Play-v0",
    help="Name of the task."
)

parser.add_argument(
    "--arch",
    type=str,
    required=True,
    help="Name of the architecture: DAC or UAC."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to checkpoint."
)

args_cli, hydra_args = parser.parse_known_args()

checkpoint_path = args_cli.checkpoint
device = "cuda:0"


# ============================================================
# SHARED ACTOR-CRITIC MODEL (UAC)
# ============================================================
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum")
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU()
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

# ============================================================
# SEPARATE POLICY MODEL (DAC)
# ============================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    def compute(self, inputs, role):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

# ============================================================
# SEPARATE VALUE MODEL (DAC)
# ============================================================
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, 1)
        )
    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# ============================================================
# CREATE MODELS
# ============================================================

models = {}


if "Flat" in args_cli.task and args_cli.arch == "DAC":

    models = {
        "policy": Policy(69, 19, device),
        "value": Value(69, 19, device)
    }


elif "Flat" in args_cli.task and args_cli.arch == "UAC":

    models["policy"] = Shared(
        69,
        19,
        device
    )

    models["value"] = models["policy"]


elif "Stair" in args_cli.task and args_cli.arch == "DAC":

    models = {
        "policy": Policy(256, 19, device),
        "value": Value(256, 19, device)
    }


elif "Stair" in args_cli.task and args_cli.arch == "UAC":

    models["policy"] = Shared(
        256,
        19,
        device
    )

    models["value"] = models["policy"]


else:

    raise ValueError(
        f"Unsupported task/architecture combination: "
        f"{args_cli.task}, {args_cli.arch}"
    )


policy_model = models["policy"]
value_model = models["value"]


# ============================================================
# 1. POLICY PARAMETERS
# ============================================================

policy_total_params = sum(
    p.numel()
    for p in policy_model.parameters()
)

policy_trainable_params = sum(
    p.numel()
    for p in policy_model.parameters()
    if p.requires_grad
)


# ============================================================
# 2. VALUE PARAMETERS
# ============================================================

value_total_params = sum(
    p.numel()
    for p in value_model.parameters()
)

value_trainable_params = sum(
    p.numel()
    for p in value_model.parameters()
    if p.requires_grad
)


# ============================================================
# 3. CHECK WHETHER MODELS ARE SHARED
# ============================================================

shared_model = policy_model is value_model


# ============================================================
# 4. TOTAL UNIQUE PARAMETERS
# ============================================================

if shared_model:

    # UAC:
    # Policy and value are the SAME model.
    total_unique_params = policy_total_params

    total_unique_trainable_params = policy_trainable_params

else:

    # DAC:
    # Policy and value are separate models.
    total_unique_params = (
        policy_total_params +
        value_total_params
    )

    total_unique_trainable_params = (
        policy_trainable_params +
        value_trainable_params
    )


# ============================================================
# PRINT PARAMETER STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PARAMETER STATISTICS")
print("=" * 70)

print(f"Task:                 {args_cli.task}")
print(f"Architecture:         {args_cli.arch}")
print(f"Policy/Value shared:  {shared_model}")

print("\nPolicy:")
print(f"  Total parameters:     {policy_total_params:,}")
print(f"  Trainable parameters: {policy_trainable_params:,}")

print("\nValue:")
print(f"  Total parameters:     {value_total_params:,}")
print(f"  Trainable parameters: {value_trainable_params:,}")

print("\nComplete model:")
print(f"  Total unique parameters:     {total_unique_params:,}")
print(f"  Total trainable parameters:  {total_unique_trainable_params:,}")


# ============================================================
# PARAMETER BREAKDOWN
# ============================================================

print("\n" + "=" * 70)
print("POLICY PARAMETER BREAKDOWN")
print("=" * 70)

for name, param in policy_model.named_parameters():

    print(
        f"{name:40s} "
        f"{param.numel():>12,} "
        f"{str(tuple(param.shape)):>20s}"
    )


# Only print value breakdown separately if models are different
if not shared_model:

    print("\n" + "=" * 70)
    print("VALUE PARAMETER BREAKDOWN")
    print("=" * 70)

    for name, param in value_model.named_parameters():

        print(
            f"{name:40s} "
            f"{param.numel():>12,} "
            f"{str(tuple(param.shape)):>20s}"
        )


# ============================================================
# 5. MODEL MEMORY
# ============================================================

def get_model_memory(model):

    parameter_memory = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    buffer_memory = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )

    return parameter_memory, buffer_memory


policy_parameter_memory, policy_buffer_memory = (
    get_model_memory(policy_model)
)

value_parameter_memory, value_buffer_memory = (
    get_model_memory(value_model)
)


if shared_model:

    total_parameter_memory = policy_parameter_memory
    total_buffer_memory = policy_buffer_memory

else:

    total_parameter_memory = (
        policy_parameter_memory +
        value_parameter_memory
    )

    total_buffer_memory = (
        policy_buffer_memory +
        value_buffer_memory
    )


total_model_memory = (
    total_parameter_memory +
    total_buffer_memory
)


# ============================================================
# PRINT MEMORY
# ============================================================

print("\n" + "=" * 70)
print("MODEL MEMORY")
print("=" * 70)

print(
    f"Policy parameter memory: "
    f"{policy_parameter_memory / 1024**2:.4f} MB"
)

print(
    f"Value parameter memory:  "
    f"{value_parameter_memory / 1024**2:.4f} MB"
)

print(
    f"Total parameter memory:  "
    f"{total_parameter_memory / 1024**2:.4f} MB"
)

print(
    f"Total buffer memory:     "
    f"{total_buffer_memory / 1024**2:.4f} MB"
)

print(
    f"Total model memory:      "
    f"{total_model_memory / 1024**2:.4f} MB"
)

print(
    f"Total model memory:      "
    f"{total_model_memory / 1024**3:.6f} GB"
)


# ============================================================
# 6. CHECKPOINT / DISK SIZE
# ============================================================

if os.path.isfile(checkpoint_path):

    checkpoint_size_bytes = os.path.getsize(
        checkpoint_path
    )

    print("\n" + "=" * 70)
    print("CHECKPOINT / DISK SIZE")
    print("=" * 70)

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Disk size: "
        f"{checkpoint_size_bytes / 1024**2:.4f} MB"
    )

    print(
        f"Disk size: "
        f"{checkpoint_size_bytes / 1024**3:.6f} GB"
    )

else:

    print("\nWARNING:")
    print(f"Checkpoint not found: {checkpoint_path}")

