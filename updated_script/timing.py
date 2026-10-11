import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Pure inference timing + memory benchmark")
parser.add_argument("--arch", type=str, default="UAC", choices=["DAC", "UAC"])
parser.add_argument("--terrain", type=str, default="stair", choices=["flat", "stair"])
parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"])
parser.add_argument("--num_runs", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=100, help="Number of iterations per CUDA timing block")
parser.add_argument("--warmup", type=int, default=500)
parser.add_argument("--eval_mode", type=str, default="policy_only", choices=["policy_only", "policy_and_value"],
                    help="Deployment benchmarks should evaluate policy_only.")
args = parser.parse_args()

# ============================================================
# Device Setup
# ============================================================
if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but CUDA is not available.")
device = torch.device(args.device)

print("=== Pure Inference Timing + Memory Benchmark ===")
print(f"Architecture : {args.arch}")
print(f"Terrain      : {args.terrain}")
print(f"Device       : {device}")
print(f"Total Runs   : {args.num_runs}")
print(f"Warmup Runs  : {args.warmup}")
print(f"Eval Mode    : {args.eval_mode}\n")

# ============================================================
# Model Dimensions
# ============================================================
obs_dim = 69 if args.terrain == "flat" else 256
act_dim = 19

# ============================================================
# Network Definitions
# ============================================================
class PolicyDAC(GaussianMixin, Model):
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

    def compute(self, inputs, role="policy"):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class ValueDAC(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, 1)
        )

    def compute(self, inputs, role="value"):
        return self.net(inputs["states"]), {}

class SharedUAC(GaussianMixin, DeterministicMixin, Model):
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

    def compute(self, inputs, role="policy"):
        shared_output = self.net(inputs["states"])
        if role == "policy":
            mean = 2 * torch.tanh(self.mean_layer(shared_output))
            return mean, self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(shared_output), {}

# ============================================================
# Instantiate Model
# ============================================================
if args.arch == "DAC":
    policy_model = PolicyDAC(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Policy": policy_model}
    if args.eval_mode == "policy_and_value":
        value_model = ValueDAC(obs_dim, act_dim, device).to(device)
        models_to_bench["Value"] = value_model
else:
    shared_model = SharedUAC(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Shared Policy": shared_model}

# ============================================================
# Print Model Information
# ============================================================
total_params = 0
for name, model in models_to_bench.items():
    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_params += params
    print(f"Model [{name}] Parameters: {params:,} | Weight Size: {param_bytes / 1024:.2f} KB")
print(f"Total Parameters Evaluated: {total_params:,}\n")

# ============================================================
# Input Setup & Execution Wrapper
# ============================================================
dummy_inputs = {"states": torch.randn(1, obs_dim, device=device)}

def run_forward(model, name):
    if "Shared" in name:
        model.compute(dummy_inputs, role="policy")
        if args.eval_mode == "policy_and_value":
            model.compute(dummy_inputs, role="value")
    else:
        model.compute(dummy_inputs, role="policy")

# ============================================================
# Benchmark Timing + Memory Loop
# ============================================================
for name, model in models_to_bench.items():
    model.eval()
    print(f"Benchmarking: {name}")

    # --- Warmup (also settles cuDNN/cuBLAS algo selection & lazy CUDA allocations) ---
    print(f"Warmup ({args.warmup} iterations) ...", flush=True)
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_forward(model, name)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete.\n", flush=True)

    # --- Memory baseline & peak-tracking setup ---
    # We reset AFTER warmup, so CUDA context init, cuDNN workspace allocation,
    # and any lazy caching-allocator growth are already "paid for" and excluded.
    # What's left is memory attributable to this model's weights + its forward-pass
    # activations, not the whole board's unified memory.
    if device.type == "cuda":
        torch.cuda.synchronize()
        resident_weights_bytes = torch.cuda.memory_allocated(device)  # weights (+ small buffers) resident on GPU
        torch.cuda.reset_peak_memory_stats(device)

    # --- Timing ---
    latencies = []
    print(f"Running {args.num_runs} timed iterations ...", flush=True)

    with torch.inference_mode():
        if device.type == "cuda":
            num_blocks = args.num_runs // args.batch_size
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(num_blocks):
                start_event.record()
                for _ in range(args.batch_size):
                    run_forward(model, name)
                end_event.record()
                torch.cuda.synchronize()

                # Average latency per forward pass in milliseconds
                block_time_ms = start_event.elapsed_time(end_event)
                latencies.extend([block_time_ms / args.batch_size] * args.batch_size)
        else:
            for _ in range(args.num_runs):
                t0 = time.perf_counter()
                run_forward(model, name)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    # --- Memory results ---
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device)     # weights + peak activations during inference
        activation_bytes = peak_bytes - resident_weights_bytes    # extra memory needed just to run forward()

    # --- Timing results ---
    latencies = np.asarray(latencies)
    mean_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    max_latency = np.max(latencies)

    print(f"--- Results for {name} ---")
    print(f"Mean Latency      : {mean_latency:.4f} ms")
    print(f"Median (P50)      : {p50:.4f} ms")
    print(f"Tail Latency (P95): {p95:.4f} ms")
    print(f"Tail Latency (P99): {p99:.4f} ms")
    print(f"Maximum Latency   : {max_latency:.4f} ms")
    print(f"Mean Throughput   : {1000.0 / mean_latency:.2f} Hz")
    if device.type == "cuda":
        print(f"Resident Weight Memory : {resident_weights_bytes / 1024:.2f} KB")
        print(f"Peak Activation Memory : {activation_bytes / 1024:.2f} KB")
        print(f"Peak Total Model Memory: {peak_bytes / 1024:.2f} KB")
    print()

print("DONE")
