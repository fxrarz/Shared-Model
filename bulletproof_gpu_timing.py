import torch
import torch.nn as nn
import time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Verifying GPU timing on: {device}")

# Simple model matching your size
model = nn.Sequential(
    nn.Linear(256, 400), nn.ELU(),
    nn.Linear(400, 200), nn.ELU(),
    nn.Linear(200, 100), nn.ELU(),
    nn.Linear(100, 19)
).to(device).eval()

dummy_input = torch.randn(1, 256, device=device)

# 1. ALWAYS WARM UP
with torch.inference_mode():
    for _ in range(200):
        _ = model(dummy_input)

# 2. PROPER CUDA EVENT TIMING
num_runs = 5000
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

latencies = []

with torch.inference_mode():
    for _ in range(num_runs):
        start_event.record()
        
        _ = model(dummy_input)
        
        end_event.record()
        # CRITICAL: This forces CPU to wait for GPU to finish this specific event
        torch.cuda.synchronize() 
        
        latencies.append(start_event.elapsed_time(end_event)) # returns milliseconds natively

latencies = np.array(latencies)
print(f"Verified GPU P95 Latency: {np.percentile(latencies, 95):.4f} ms")
print(f"Verified GPU Mean Latency: {np.mean(latencies):.4f} ms")
