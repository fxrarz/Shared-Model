# python3 benchmark_power.py --arch DAC --terrain flat --warmup 20000 --device cuda:0 --batch_size 1 --num_runs 5000

import argparse
import sys
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Dedicated Jetson Power Consumption Benchmark")
parser.add_argument("--arch", type=str, default="UAC", choices=["DAC", "UAC"])
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "stair"])
parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"])
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for model inputs")
parser.add_argument("--num_runs", type=int, default=5000, help="Total inference loops for power sampling")
parser.add_argument("--warmup", type=int, default=20000, help="Warmup iterations to stabilize thermal/clock state")
parser.add_argument("--sample_interval", type=float, default=0.1, help="Power sampling interval in seconds")
args = parser.parse_args()

# ============================================================
# jtop Import Setup
# ============================================================
JTOP_PATH = "/usr/local/lib/python3.6/dist-packages"
if JTOP_PATH not in sys.path:
    sys.path.insert(0, JTOP_PATH)
try:
    from jtop import jtop
except ImportError:
    raise RuntimeError("jtop is not installed or available in the Python environment.")

# ============================================================
# Device Setup
# ============================================================
if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but CUDA is not available.")
device = torch.device(args.device)

print("=== Jetson Power Consumption Benchmark ===")
print(f"Architecture : {args.arch}")
print(f"Terrain      : {args.terrain}")
print(f"Device       : {device}")
print(f"Batch Size   : {args.batch_size}")
print(f"Warmup Runs  : {args.warmup:,}")
print(f"Measured Runs: {args.num_runs:,}")
print(f"Sample Rate  : {args.sample_interval} sec\n")

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
else:
    shared_model = SharedUAC(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Shared Policy": shared_model}

# ============================================================
# Input Setup & Execution Wrapper (Respects Batch Size)
# ============================================================
dummy_inputs = {"states": torch.randn(args.batch_size, obs_dim, device=device)}

def run_forward(model, name):
    if "Shared" in name:
        model.compute(dummy_inputs, role="policy")
        model.compute(dummy_inputs, role="value")
    else:
        model.compute(dummy_inputs, role="policy")

# ============================================================
# Power Monitor Background Thread
# ============================================================
class PowerMonitor:
    def __init__(self, interval):
        self.interval = interval
        self.running = False
        self.samples = []
        self.thread = None
        self.jetson = None

    def start(self):
        print("Initializing jtop for power tracking...", flush=True)
        self.jetson = jtop()
        self.jetson.start()
        time.sleep(0.5)
        self.running = True
        self.thread = threading.Thread(target=self._monitor, name="power-monitor")
        self.thread.daemon = True
        self.thread.start()
        print("Power monitor active.", flush=True)

    def _monitor(self):
        while self.running:
            try:
                if self.jetson is not None and self.jetson.ok():
                    power = self.jetson.power
                    if power and "tot" in power:
                        total = power["tot"]
                        if "power" in total:
                            self.samples.append(float(total["power"]))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.jetson is not None:
            try:
                self.jetson.close()
            except Exception:
                pass
        print("Power monitor stopped.", flush=True)

# ============================================================
# Benchmark Execution Loop
# ============================================================
for name, model in models_to_bench.items():
    model.eval()
    print(f"Target Model: {name}")

    # --- Warmup Phase ---
    print(f"Executing Warmup ({args.warmup:,} iterations) ...", flush=True)
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_forward(model, name)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete. Hardware state stabilized.\n", flush=True)

    # --- Start Power Monitoring ---
    monitor = PowerMonitor(args.sample_interval)
    monitor.start()
    time.sleep(1.0)  # Allow sampling thread to establish a steady baseline

    # --- Power-Measured Inference Loop ---
    print(f"Running {args.num_runs:,} measured inference iterations...", flush=True)
    start_time = time.perf_counter()

    with torch.inference_mode():
        for _ in range(args.num_runs):
            run_forward(model, name)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # Let monitor capture tail stabilization for 1 second
    time.sleep(1.0)
    monitor.stop()

    # --- Results & Aggregation ---
    print(f"\n--- Power Results for {name} ---")
    print(f"Total Execution Time : {elapsed:.4f} sec")
    print(f"Power Samples Captured: {len(monitor.samples)}")

    if monitor.samples:
        samples = np.asarray(monitor.samples)
        avg_mw = np.mean(samples)
        min_mw = np.min(samples)
        max_mw = np.max(samples)

        print(f"Average Power Draw   : {avg_mw:.2f} mW ({avg_mw / 1000.0:.3f} W)")
        print(f"Minimum Power Draw   : {min_mw:.2f} mW ({min_mw / 1000.0:.3f} W)")
        print(f"Maximum Power Draw   : {max_mw:.2f} mW ({max_mw / 1000.0:.3f} W)")
    else:
        print("WARNING: No power samples were collected by jtop.")
    print()

print("DONE")
