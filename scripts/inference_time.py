# Example:
# python3 inference_timing.py --arch UAC --terrain stair --device cuda:0 --num_runs 5000

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Policy-only inference timing benchmark")
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
print("=== Policy-Only Inference Timing Benchmark ===")
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
# DAC Policy
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
        mean = self.net(inputs["states"])
        return mean, self.log_std_parameter, {}
# ============================================================
# UAC Shared Policy
# ============================================================
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
    def compute(self, inputs, role):
        if role == "policy":
            shared_output = self.net(inputs["states"])
            mean = self.mean_layer(shared_output)
            return (mean, self.log_std_parameter, {})
        elif role == "value":
            shared_output = self.net(inputs["states"])
            return (self.value_layer(shared_output), {})
# ============================================================
# Instantiate POLICY ONLY
# ============================================================
if args.arch == "DAC":
    policy_model = DAC_Policy(obs_dim, act_dim, device).to(device)
    model_name = "DAC Policy"
else:
    policy_model = UAC_Shared(obs_dim, act_dim, device).to(device)
    model_name = "UAC Policy"
# ============================================================
# Model information
# ============================================================
params = sum(p.numel() for p in policy_model.parameters())
param_bytes = sum(p.numel() * p.element_size() for p in policy_model.parameters())
print("Model [{}] Parameters: {:,} | Size: {:.2f} KB".format(model_name, params, param_bytes / 1024))
print()
# ============================================================
# Input
# ============================================================
dummy_inputs = {"states": torch.randn(1, obs_dim, device=device)}
# ============================================================
# Policy forward function
# ============================================================
def run_policy(model):
    model.compute(dummy_inputs, role="policy")
# ============================================================
# Benchmark
# ============================================================
policy_model.eval()
print("Benchmarking:", model_name)
# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------
print("Warmup ({}) ...".format(args.warmup), flush=True)
with torch.inference_mode():
    for _ in range(args.warmup):
        run_policy(policy_model)
if device.type == "cuda":
    torch.cuda.synchronize()
print("Warmup complete.", flush=True)
# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------
latencies = []
print("Running {} timed policy iterations ...".format(args.num_runs), flush=True)
if device.type == "cuda":
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(args.num_runs):
            start_event.record()
            run_policy(policy_model)
            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event)
            latencies.append(latency)
else:
    with torch.inference_mode():
        for _ in range(args.num_runs):
            start = time.perf_counter()
            run_policy(policy_model)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
# ============================================================
# Results
# ============================================================
latencies = np.asarray(latencies)
mean_latency = np.mean(latencies)
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
max_latency = np.max(latencies)
print()
print("--- Results for {} ---".format(model_name))
print("Mean Latency       : {:.4f} ms".format(mean_latency))
print("Median (P50)       : {:.4f} ms".format(p50))
print("Tail Latency (P95) : {:.4f} ms".format(p95))
print("Tail Latency (P99) : {:.4f} ms".format(p99))
print("Maximum Latency    : {:.4f} ms".format(max_latency))
print("P95 Throughput     : {:.2f} Hz".format(1000.0 / p95))
print("Mean Throughput    : {:.2f} Hz".format(1000.0 / mean_latency))
print()
print("DONE")
