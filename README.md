# Shared-Model

## Train
```
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Stairway-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Stairway-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Stairway-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Stairway-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Stairway-H1-v0 --headless --seed 999 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Stairway-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Stairway-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Stairway-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Stairway-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Stairway-H1-v0 --headless --seed 999 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/dac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 999 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./1_train/uac.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 999
```

## Plot
```
python3 3_plot/plot_loss_95.py --tag "Loss / Policy loss" --ylabel "Policy Loss" --layout vertical --output policy_loss
python3 3_plot/plot_loss_95.py --tag "Loss / Value loss" --ylabel "Value Loss" --layout vertical --output value_loss
python3 3_plot/plot_loss_95.py --tag "Policy / Standard deviation" --ylabel "Policy Standard Deviation" --layout vertical --output standard_deviation
python3 3_plot/plot_loss_95.py --tag "Learning / Learning rate" --ylabel "Learning Rate" --layout vertical --output learning_rate
python3 3_plot/plot_loss_95.py --tag "Reward / Total reward (mean)" --ylabel "Total Reward (mean)" --layout vertical --output total_reward
python3 3_plot/plot_loss_95.py --tag "Info / Episode_Reward/track_lin_vel_xy_exp" --ylabel "Linear Velocity" --layout vertical --output linear_velocity
python3 3_plot/plot_loss_95.py --tag "Info / Episode_Reward/track_ang_vel_z_exp" --ylabel "Angular Velocity" --layout vertical --output angular_velocity
python3 3_plot/plot_loss_95.py --tag "Info / Episode_Reward/feet_air_time" --ylabel "Feet Air Time" --layout vertical --output feet_air_time

```

## Test
```
python3 2_test/Perturbation/benchmark.py --task Isaac-Velocity-Flat-H1-Play-v0 --arch DAC --push_force 100.0 && \
python3 2_test/Perturbation/benchmark.py --task Isaac-Velocity-Flat-H1-Play-v0 --arch UAC --push_force 100.0 && \
python3 2_test/Perturbation/benchmark.py --task Isaac-Stairway-H1-Play-v0 --arch DAC --push_force 100.0 && \
python3 2_test/Perturbation/benchmark.py --task Isaac-Stairway-H1-Play-v0 --arch UAC --push_force 100.0 && \
python3 2_test/Perturbation/benchmark.py --task Isaac-Velocity-Flat-H1-Play-v0 --arch DAC --push_force 120.0 && \
python3 2_test/Perturbation/benchmark.py --task Isaac-Velocity-Flat-H1-Play-v0 --arch UAC --push_force 120.0 && \
python3 2_test/Perturbation/benchmark.py --task Isaac-Stairway-H1-Play-v0 --arch DAC --push_force 120.0 && \
python3 2_test/Perturbation/benchmark.py --task Isaac-Stairway-H1-Play-v0 --arch UAC --push_force 120.0
```

## Eval
```
for SEED in 42 123 456 789 999; do
    echo "=========================================="
    echo "Running evaluation suite with seed: $SEED"
    echo "=========================================="
    
    python3 2_test/Evaluation/dac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --seed $SEED --headless && \
    python3 2_test/Evaluation/uac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --seed $SEED --headless && \
    python3 2_test/Evaluation/dac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --seed $SEED --headless && \
    python3 2_test/Evaluation/uac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --seed $SEED --headless
done
```

## Model Stats
```
python3 model_stats.py --arch DAC --task Flat --checkpoint /home/whitebot/Workspace/Research/Shared-Model/runs/flat_terrain_nogui_DAC_2026-09-21_07-39-05_seed_42/checkpoints/agent_2400.pt
python3 model_stats.py --arch UAC --task Flat --checkpoint /home/whitebot/Workspace/Research/Shared-Model/runs/flat_terrain_nogui_UAC_2026-09-21_09-09-35_seed_42/checkpoints/agent_2400.pt
python3 model_stats.py --arch DAC --task Stair --checkpoint /home/whitebot/Workspace/Research/Shared-Model/runs/stair_terrain_nogui_DAC_2026-09-20_20-59-22_seed_42/checkpoints/agent_7200.pt
python3 model_stats.py --arch UAC --task Stair --checkpoint /home/whitebot/Workspace/Research/Shared-Model/runs/stair_terrain_nogui_UAC_2026-09-21_02-41-31_seed_42/checkpoints/agent_7200.pt
```

## Continuous Learning
```
python3 2_test/CL/multi_agent_sim.py --task Isaac-Slope-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch dac
python3 2_test/CL/multi_agent_sim.py --task Isaac-Slope-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch uac
python3 2_test/CL/multi_agent_sim.py --task Isaac-Box-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch dac
python3 2_test/CL/multi_agent_sim.py --task Isaac-Box-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch uac

python3 2_test/CL/multi_agent_sim.py --task Isaac-Slope-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch dac --seed 999
python3 2_test/CL/multi_agent_sim.py --task Isaac-Slope-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch uac --seed 999
python3 2_test/CL/multi_agent_sim.py --task Isaac-Box-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch dac --seed 999
python3 2_test/CL/multi_agent_sim.py --task Isaac-Box-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch uac --seed 999
```

## Catestrophic Forgetting
```
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Slope-H1-v0 --headless --seed 42 --checkpoint runs/flat_terrain_DAC_nogui_seed_42_2026-09-26_00-26-14/checkpoints/best_agent.pt && 
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Slope-H1-v0 --headless --seed 42 --checkpoint runs/flat_terrain_UAC_nogui_seed_42_2026-09-26_02-00-27/checkpoints/best_agent.pt && 
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Box-H1-v0 --headless --seed 42 --checkpoint runs/stair_terrain_DAC_nogui_seed_42_2026-09-25_13-18-32/checkpoints/best_agent.pt && 
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Box-H1-v0 --headless --seed 42 --checkpoint runs/stair_terrain_UAC_nogui_seed_42_2026-09-25_19-12-50/checkpoints/best_agent.pt
```

```
for SEED in 42 123 456 789 999; do
    echo "=========================================="
    echo "Running evaluation suite with seed: $SEED"
    echo "=========================================="
    
    python3 2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --seed $SEED --checkpoint runs/flat_terrain_DAC_nogui_seed_42_2026-09-26_00-26-14/checkpoints/best_agent.pt && \
    python3 2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --seed $SEED --checkpoint runs/ablation3_slope_terrain_DAC_nogui_seed_42_2026-10-10_10-38-48/checkpoints/best_agent.pt && \
    python3 2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --seed $SEED --checkpoint runs/flat_terrain_UAC_nogui_seed_42_2026-09-26_02-00-27/checkpoints/best_agent.pt && \
    python3 2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --seed $SEED --checkpoint runs/ablation3_slope_terrain_UAC_nogui_seed_42_2026-10-10_11-45-28/checkpoints/best_agent.pt && \
    python3 2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --seed $SEED --checkpoint runs/stair_terrain_DAC_nogui_seed_42_2026-09-25_13-18-32/checkpoints/best_agent.pt && \
    python3 2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --seed $SEED --checkpoint runs/ablation3_box_terrain_DAC_nogui_seed_42_2026-10-10_12-47-17/checkpoints/best_agent.pt && \
    python3 2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --seed $SEED --checkpoint runs/stair_terrain_UAC_nogui_seed_42_2026-09-25_19-12-50/checkpoints/best_agent.pt && \
    python3 2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --seed $SEED --checkpoint runs/ablation3_box_terrain_UAC_nogui_seed_42_2026-10-10_13-57-14/checkpoints/best_agent.pt
done
```

```
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --headless --seed 42 --checkpoint runs/flat_terrain_DAC_nogui_seed_42_2026-09-26_00-26-14/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --headless --seed 42 --checkpoint runs/ablation3_slope_terrain_DAC_nogui_seed_42_2026-10-10_10-38-48/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --checkpoint runs/stair_terrain_DAC_nogui_seed_42_2026-09-25_13-18-32/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --checkpoint runs/ablation3_box_terrain_DAC_nogui_seed_42_2026-10-10_12-47-17/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --headless --seed 42 --checkpoint runs/flat_terrain_UAC_nogui_seed_42_2026-09-26_02-00-27/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Velocity-Flat-H1-Play-v0 --headless --seed 42 --checkpoint runs/ablation3_slope_terrain_UAC_nogui_seed_42_2026-10-10_11-45-28/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --checkpoint runs/stair_terrain_UAC_nogui_seed_42_2026-09-25_19-12-50/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac_test.py --num_envs 1 --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --checkpoint runs/ablation3_box_terrain_UAC_nogui_seed_42_2026-10-10_13-57-14/checkpoints/best_agent.pt
```

## Ablation
```
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Slope-H1-v0 --headless --seed 42 --checkpoint runs/flat_terrain_DAC_nogui_seed_42_2026-09-26_00-26-14/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Slope-H1-v0 --headless --seed 42 --checkpoint runs/flat_terrain_UAC_nogui_seed_42_2026-09-26_02-00-27/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Box-H1-v0 --headless --seed 42 --checkpoint runs/stair_terrain_DAC_nogui_seed_42_2026-09-25_13-18-32/checkpoints/best_agent.pt && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Box-H1-v0 --headless --seed 42 --checkpoint runs/stair_terrain_UAC_nogui_seed_42_2026-09-25_19-12-50/checkpoints/best_agent.pt
```

```
for SEED in 42 123 456 789 999; do
    echo "=========================================="
    echo "Running evaluation suite with seed: $SEED"
    echo "=========================================="
    
    /home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Slope-H1-v0 --headless --seed $SEED && \
    /home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Slope-H1-v0 --headless --seed $SEED && \
    /home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Box-H1-v0 --headless --seed $SEED && \
    /home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Box-H1-v0 --headless --seed $SEED
done
```

```
python3 2_test/Ablation/loss_scaling.py --task Isaac-Velocity-Flat-H1-v0 --value_loss_scale 8.0 --headless --seed 42 && \
python3 2_test/Ablation/loss_scaling.py --task Isaac-Velocity-Flat-H1-v0 --value_loss_scale 2.0 --headless --seed 42 && \
python3 2_test/Ablation/loss_scaling.py --task Isaac-Velocity-Flat-H1-v0 --value_loss_scale 0.0 --headless --seed 42 && \
python3 2_test/Ablation/loss_scaling.py --task Isaac-Stairway-H1-v0 --value_loss_scale 8.0 --headless --seed 42 && \
python3 2_test/Ablation/loss_scaling.py --task Isaac-Stairway-H1-v0 --value_loss_scale 2.0 --headless --seed 42 && \
python3 2_test/Ablation/loss_scaling.py --task Isaac-Stairway-H1-v0 --value_loss_scale 0.0 --headless --seed 42 && \
python3 2_test/Ablation/sharing_depth.py --task Isaac-Velocity-Flat-H1-v0 --arch 2.0 --headless --seed 42 && \
python3 2_test/Ablation/sharing_depth.py --task Isaac-Velocity-Flat-H1-v0 --arch 1.0 --headless --seed 42 && \
python3 2_test/Ablation/sharing_depth.py --task Isaac-Stairway-H1-v0 --arch 2.0 --headless --seed 42 && \
python3 2_test/Ablation/sharing_depth.py --task Isaac-Stairway-H1-v0 --arch 1.0 --headless --seed 42
```

```
python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 110.0 --push_interval 10 && \
python3 Play_UAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 110.0 --push_interval 10 && \
python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 110.0 --push_interval 10 && \
python3 Play_UAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 110.0 --push_interval 10
```

```
python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 90.0 --push_interval 10 && \
python3 Play_UAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 90.0 --push_interval 10 && \
python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 90.0 --push_interval 10 && \
python3 Play_UAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 90.0 --push_interval 10
```

```
python3 UAC_Loss_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --value_loss_scale 8.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --value_loss_scale 3.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --value_loss_scale 2.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --value_loss_scale 1.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --value_loss_scale 0.5 && \
python3 UAC_Loss_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --value_loss_scale 0.0 && \

python3 UAC_Loss_Ablation.py --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --value_loss_scale 8.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --value_loss_scale 6.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --value_loss_scale 5.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --value_loss_scale 3.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --value_loss_scale 1.0 && \
python3 UAC_Loss_Ablation.py --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --value_loss_scale 0.5 && \
python3 UAC_Loss_Ablation.py --task Isaac-Stairway-H1-Play-v0 --headless --seed 42 --value_loss_scale 0.0
```

```
python3 UAC_Arch_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --arch 2.0 && \
python3 UAC_Arch_Ablation.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 --arch 1.0 && \
python3 UAC_Arch_Ablation.py --task Isaac-Stairway-H1-v0 --headless --seed 42 --arch 2.0 && \
python3 UAC_Arch_Ablation.py --task Isaac-Stairway-H1-v0 --headless --seed 42 --arch 1.0 
```

```
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Stairway-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Stairway-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Stairway-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Stairway-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Stairway-H1-v0 --headless --seed 999 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./Train_DAC.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 999
```




import sys
JTOP_PATH = "/usr/local/lib/python3.6/dist-packages"
if JTOP_PATH not in sys.path:
    sys.path.insert(0, JTOP_PATH)

try:
    from jtop import jtop
except ImportError:
    raise RuntimeError("jtop is not installed or available in the Python environment.")

with jtop() as jetson:
    if jetson.ok():
        stats = jetson.stats

print(stats)


import time
def temp():
  with jtop() as jetson:
      if jetson.ok():
          stats = jetson.stats
      print(stats)

for i in range(50):
  temp()
  time.sleep(2)
