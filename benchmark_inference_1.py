import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# 1. Parse CLI Arguments
parser = argparse.ArgumentParser(description="Benchmark DRL model inference time and memory usage.")
parser.add_argument("--arch", type=str, default="UAC", choices=["DAC", "UAC"], help="Architecture type (DAC or UAC).")
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "stair"], help="Terrain type to determine input dimension.")
parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"], help="Computation device.")
parser.add_argument("--num_runs", type=int, default=5000, help="Number of forward pass iterations for benchmarking.")
parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations before recording stats.")
args = parser.parse_args()

# Setup device safely
device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
print(f"=== Inference Benchmark Configuration ===")
print(f"Architecture : {args.arch}")
print(f"Terrain      : {args.terrain}")
print(f"Device       : {device}")
print(f"Runs         : {args.num_runs}\n")

# Determine observation dimension based on terrain
obs_dim = 69 if args.terrain == "flat" else 256
act_dim = 19

# 2. Define Architectures
class DAC_Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class DAC_Value(DeterministicMixin, Model):
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

class UAC_Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=2)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU()
        )
        self.mean_layer = nn.Linear(100, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.value_layer = nn.Linear(100, 1)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}

# 3. Instantiate Models and Measure Memory Consumption
if args.arch == "DAC":
    policy_model = DAC_Policy(obs_dim, act_dim, device).to(device)
    value_model = DAC_Value(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Policy": policy_model, "Value": value_model}
else:
    shared_model = UAC_Shared(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Shared (Policy + Value)": shared_model}

# Calculate static parameter memory footprint
total_params = 0
for name, model in models_to_bench.items():
    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_params += params
    print(f"Model [{name}] Parameters: {params:,} | Size in RAM/VRAM: {param_bytes / 1024:.2f} KB")

print(f"Total Combined Parameters: {total_params:,}\n")

# 4. Benchmark Execution Loop (Dummy Input Data)
dummy_inputs = {"states": torch.randn(1, obs_dim, device=device)}

for name, model in models_to_bench.items():
    model.eval()
    
    # Warmup phase
    with torch.inference_mode():
        for _ in range(args.warmup):
            if "Shared" in name:
                _ = model.compute(dummy_inputs, role="policy")
                _ = model.compute(dummy_inputs, role="value")
            else:
                _ = model.compute(dummy_inputs, role="policy")

    latencies = []
    
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            for _ in range(args.num_runs):
                start_event.record()
                if "Shared" in name:
                    _ = model.compute(dummy_inputs, role="policy")
                    _ = model.compute(dummy_inputs, role="value")
                else:
                    _ = model.compute(dummy_inputs, role="policy")
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
    else:
        with torch.inference_mode():
            for _ in range(args.num_runs):
                start = time.perf_counter()
                if "Shared" in name:
                    _ = model.compute(dummy_inputs, role="policy")
                    _ = model.compute(dummy_inputs, role="value")
                else:
                    _ = model.compute(dummy_inputs, role="policy")
                end = time.perf_counter()
                latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_lat = np.mean(latencies)

    print(f"--- Results for {name} ---")
    print(f"  Mean Latency       : {mean_lat:.4f} ms")
    print(f"  Median (P50)       : {p50:.4f} ms")
    print(f"  Tail Latency (P95) : {p95:.4f} ms")
    print(f"  Tail Latency (P99) : {p99:.4f} ms")
    print(f"  Max Stable Hz (P95): {1000.0 / p95:.2f} Hz\n")
