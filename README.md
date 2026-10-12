# max power on
sudo nvpmodel -m 0

# max power off
sudo nvpmodel -m 1

# print power mode
sudo nvpmodel -q



for i in {1..10}; do echo "Running iteration $i..."; python3 benchmark_power.py --arch DAC --terrain flat --warmup 20000 --device cuda:0 --batch_size 1 --num_runs 30000; done && 
for i in {1..10}; do echo "Running iteration $i..."; python3 benchmark_power.py --arch UAC --terrain flat --warmup 20000 --device cuda:0 --batch_size 1 --num_runs 30000; done &&
for i in {1..10}; do echo "Running iteration $i..."; python3 benchmark_power.py --arch DAC --terrain stair --warmup 20000 --device cuda:0 --batch_size 1 --num_runs 30000; done &&
for i in {1..10}; do echo "Running iteration $i..."; python3 benchmark_power.py --arch UAC --terrain stair --warmup 20000 --device cuda:0 --batch_size 1 --num_runs 30000; done

---

for i in {1..10}; do echo "Running iteration $i..."; python3 timing.py --arch DAC --terrain flat --warmup 20000 --device cuda:0 --block_size 1 --num_runs 30000; done && 
for i in {1..10}; do echo "Running iteration $i..."; python3 timing.py --arch UAC --terrain flat --warmup 20000 --device cuda:0 --block_size 1 --num_runs 30000; done &&
for i in {1..10}; do echo "Running iteration $i..."; python3 timing.py --arch DAC --terrain stair --warmup 20000 --device cuda:0 --block_size 1 --num_runs 30000; done && 
for i in {1..10}; do echo "Running iteration $i..."; python3 timing.py --arch UAC --terrain stair --warmup 20000 --device cuda:0 --block_size 1 --num_runs 30000; done

---

mkdir -p logs
for i in {1..10}; do
  for terrain in flat stair; do
    for arch in DAC UAC; do
      echo "Run $i: $arch $terrain"
      python3 updated_script/benchmark_power.py --arch $arch --terrain $terrain --warmup 20000 \
        --device cuda:0 --batch_size 1 --num_runs 30000 | tee -a logs/power_${arch}_${terrain}.log
    done
  done
done

---

mkdir -p logs
for i in {1..10}; do
  for terrain in flat stair; do
    for arch in DAC UAC; do
      echo "Run $i: $arch $terrain"
      python3 updated_script/timing.py --arch $arch --terrain $terrain --warmup 20000 \
        --device cuda:0 --block_size 1 --num_runs 30000 | tee -a logs/timing_${arch}_${terrain}.log
    done
  done
done
