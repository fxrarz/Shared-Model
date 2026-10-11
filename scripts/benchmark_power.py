# python3 benchmark_power.py --arch UAC --terrain stair --device cuda:0 --num_runs 5000

import argparse
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Inference + Jetson power benchmark")
parser.add_argument("--arch", type=str, default="UAC", choices=["DAC", "UAC"])
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "stair"])
parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"])
parser.add_argument("--num_runs", type=int, default=5000)
parser.add_argument("--warmup", type=int, default=200)
parser.add_argument("--sample_interval", type=float, default=0.1, help="Power sampling interval in seconds")
args = parser.parse_args()
# ============================================================
# jtop
# ============================================================
JTOP_PATH = "/usr/local/lib/python3.6/dist-packages"
import sys
if JTOP_PATH not in sys.path:
    sys.path.insert(0, JTOP_PATH)
try:
    from jtop import jtop
except ImportError:
    raise RuntimeError("jtop is not installed/available.")
# ============================================================
# Device
# ============================================================
if args.device.startswith("cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable.")
device = torch.device(args.device)
print("=== Inference Power Benchmark ===")
print("Architecture :", args.arch)
print("Terrain      :", args.terrain)
print("Device       :", device)
print("Runs         :", args.num_runs)
print("Warmup       :", args.warmup)
print("Power sample interval : {} sec".format(args.sample_interval))
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
# Model
# ============================================================
if args.arch == "DAC":
    policy_model = DAC_Policy(obs_dim, act_dim, device).to(device)
    value_model = DAC_Value(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Policy": policy_model, "Value": value_model}
else:
    shared_model = UAC_Shared(obs_dim, act_dim, device).to(device)
    models_to_bench = {"Shared (Policy + Value)": shared_model}
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
# Power monitor
# ============================================================
class PowerMonitor:
    def __init__(self, interval):
        self.interval = interval
        self.running = False
        self.samples = []
        self.thread = None
        self.jetson = None
    def start(self):
        print("Starting jtop...", flush=True)
        self.jetson = jtop()
        self.jetson.start()
        time.sleep(0.5)
        self.running = True
        self.thread = threading.Thread(target=self._monitor, name="power-monitor")
        self.thread.daemon = True
        self.thread.start()
        print("Power monitor started.", flush=True)
    def _monitor(self):
        while self.running:
            try:
                if self.jetson is not None and self.jetson.ok():
                    power = self.jetson.power
                    if power:
                        if "tot" in power:
                            total = power["tot"]
                            if "power" in total:
                                value = total["power"]
                                self.samples.append(float(value))
            except Exception as e:
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
# Benchmark
# ============================================================
for name, model in models_to_bench.items():
    model.eval()
    print()
    print("Benchmarking:", name, flush=True)
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
    # Start power monitor
    # --------------------------------------------------------
    monitor = PowerMonitor(args.sample_interval)
    monitor.start()
    time.sleep(0.5)
    # --------------------------------------------------------
    # Run exactly num_runs
    # --------------------------------------------------------
    print("Running {} inference passes...".format(args.num_runs), flush=True)
    start_time = time.perf_counter()
    with torch.inference_mode():
        for i in range(args.num_runs):
            run_forward(model, name)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    # --------------------------------------------------------
    # Stop monitor
    # --------------------------------------------------------
    monitor.stop()
    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    print()
    print("--- Results for {} ---".format(name))
    print("Inference time    : {:.6f} sec".format(elapsed))
    print("Per inference     : {:.6f} ms".format(elapsed * 1000.0 / args.num_runs))
    print("Throughput        : {:.2f} Hz".format(args.num_runs / elapsed))
    print("Power samples     : {}".format(len(monitor.samples)))
    if monitor.samples:
        samples = np.asarray(monitor.samples)
        print("Average power     : {:.2f} mW".format(np.mean(samples)))
        print("Minimum power     : {:.2f} mW".format(np.min(samples)))
        print("Maximum power     : {:.2f} mW".format(np.max(samples)))
    else:
        print("WARNING: No power samples were collected.")
    print()
    print("DONE")
