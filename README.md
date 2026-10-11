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

## Continuous Learning
```
python3 2_test/CL/multi_agent_sim.py --task Isaac-Slope-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch dac
python3 2_test/CL/multi_agent_sim.py --task Isaac-Slope-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch uac
python3 2_test/CL/multi_agent_sim.py --task Isaac-Box-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch dac
python3 2_test/CL/multi_agent_sim.py --task Isaac-Box-H1-v0 --headless --num_envs 250 --num_episodes 10 --max_steps 1000 --arch uac
```

## Catestrophic Forgetting
```
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Slope-H1-v0 --headless --seed 42 --checkpoint runs/flat_terrain_DAC_nogui_seed_42_2026-09-26_00-26-14/checkpoints/best_agent.pt && 
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Slope-H1-v0 --headless --seed 42 --checkpoint runs/flat_terrain_UAC_nogui_seed_42_2026-09-26_02-00-27/checkpoints/best_agent.pt && 
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/dac.py --task Isaac-Box-H1-v0 --headless --seed 42 --checkpoint runs/stair_terrain_DAC_nogui_seed_42_2026-09-25_13-18-32/checkpoints/best_agent.pt && 
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./2_test/Ablation/ablation3/uac.py --task Isaac-Box-H1-v0 --headless --seed 42 --checkpoint runs/stair_terrain_UAC_nogui_seed_42_2026-09-25_19-12-50/checkpoints/best_agent.pt
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



