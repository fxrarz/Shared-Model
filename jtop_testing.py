import sys

# Make the system-wide jetson-stats/jtop package available
# inside this virtual environment.
JTOP_PATH = "/usr/local/lib/python3.6/dist-packages"
if JTOP_PATH not in sys.path:
    sys.path.insert(0, JTOP_PATH)

from jtop import jtop
import time
import torch

# Initialize jtop to read hardware stats
with jtop() as jetson:
    if jetson.ok():
        print("Jetson stats initialized. Starting benchmark power tracking...")

        # Check available power rails
        print("Initial Power Status:", jetson.power)

        # Run your inference loop here
        start_time = time.time()
        power_samples = []

        # Example loop simulating your benchmark
        for _ in range(5000):
            # --- Your model forward pass ---
            # dummy_output = model(dummy_input)

            # Capture instantaneous power draw (in mW) from jtop
            if jetson.power and 'tot' in jetson.power:
                power_samples.append(jetson.power['tot']['power'])

        print(f"Benchmark finished in {time.time() - start_time:.2f} seconds.")

        if power_samples:
            avg_power = sum(power_samples) / len(power_samples)
            print(f"Average Total Power Draw During Inference: {avg_power:.2f} mW")

"""
from jtop import jtop
import time
import torch

# Initialize jtop to read hardware stats
with jtop() as jetson:
    if jetson.ok():
        print("Jetson stats initialized. Starting benchmark power tracking...")
        
        # Check available power rails
        print("Initial Power Status:", jetson.power)

        # Run your inference loop here
        start_time = time.time()
        power_samples = []

        # Example loop simulating your benchmark
        for _ in range(5000):
            # --- Your model forward pass ---
            # dummy_output = model(dummy_input)
            
            # Capture instantaneous power draw (in mW) from jtop
            if jetson.power and 'tot' in jetson.power:
                power_samples.append(jetson.power['tot']['power'])

        print(f"Benchmark finished in {time.time() - start_time:.2f} seconds.")
        if power_samples:
            avg_power = sum(power_samples) / len(power_samples)
            print(f"Average Total Power Draw During Inference: {avg_power:.2f} mW")
"""
