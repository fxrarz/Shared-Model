# python3 benchmark_timing.py --arch UAC --terrain stair --device cuda:0 --num_runs 5000


import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Pure inference timing benchmark")
parser.add_argument("--arch", type=str, default="UAC", choices=["DAC", "UAC"])
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "stair"])
parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"])
parser.add_argument("--num_runs", type=int, default=5000)
parser.add_argument("--warmup", type=int, default=200)
args = parser.parse_args()
# ============================================================
# Device
# ============================================================
if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but CUDA is not available.")
device = torch.device(args.device)
print("=== Pure Inference Timing Benchmark ===")
print("Architecture :", args.arch)
print("Terrain      :", args.terrain)
print("Device       :", device)
print("Runs         :", args.num_runs)
print("Warmup       :", args.warmup)
print()
# ============================================================
# Model dimensions
# ============================================================
obs_dim = 69 if args.terrain == "flat" else 256
act_dim = 19
# ============================================================
# Models
# ============================================================
class DAC_Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ELU(),
            nn.Linear(400, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    def compute(self, inputs, role):
        return (2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {})
class DAC_Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ELU(),
            nn.Linear(400, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
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
    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return (self.mean_layer(self._shared_output), self.log_std_parameter, {})
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return (self.value_layer(shared_output), {})
# ============================================================
# Instantiate model
# ============================================================
if args.arch == "DAC":
    policy_model = DAC_Policy(obs_dim, act_dim, device).to(device)
    value_model = DAC_Value(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Policy": policy_model, "Value": value_model}
else:
    shared_model = UAC_Shared(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Shared (Policy + Value)": shared_model}
# ============================================================
# Model information
# ============================================================
total_params = 0
for name, model in models_to_bench.items():
    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_params += params
    print("Model [{}] Parameters: {:,} | Size: {:.2f} KB".format(name, params, param_bytes / 1024))
print("Total Combined Parameters: {:,}".format(total_params))
print()
# ============================================================
# Input
# ============================================================
dummy_inputs = {"states": torch.randn(1, obs_dim, device=device)}
def run_forward(model, name):
    if "Shared" in name:
        model.compute(dummy_inputs, role="policy")
        model.compute(dummy_inputs, role="value")
    else:
        model.compute(dummy_inputs, role="policy")
# ============================================================
# Benchmark
# ============================================================
for name, model in models_to_bench.items():
    model.eval()
    print("Benchmarking:", name)
    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------
    print("Warmup ({}) ...".format(args.warmup), flush=True)
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_forward(model, name)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete.", flush=True)
    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------
    latencies = []
    print("Running {} timed iterations ...".format(args.num_runs), flush=True)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            for i in range(args.num_runs):
                start_event.record()
                run_forward(model, name)
                end_event.record()
                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event)
                latencies.append(latency)
    else:
        with torch.inference_mode():
            for i in range(args.num_runs):
                start = time.perf_counter()
                run_forward(model, name)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)
    # ========================================================
    # Results
    # ========================================================
    latencies = np.asarray(latencies)
    mean_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    max_latency = np.max(latencies)
    print()
    print("--- Results for {} ---".format(name))
    print("Mean Latency       : {:.4f} ms".format(mean_latency))
    print("Median (P50)       : {:.4f} ms".format(p50))
    print("Tail Latency (P95) : {:.4f} ms".format(p95))
    print("Tail Latency (P99) : {:.4f} ms".format(p99))
    print("Maximum Latency    : {:.4f} ms".format(max_latency))
    print("P95 Throughput     : {:.2f} Hz".format(1000.0 / p95))
    print("Mean Throughput    : {:.2f} Hz".format(1000.0 / mean_latency))
    print()
    print("DONE")
