# Shared-Model

```
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Stairway-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Stairway-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Stairway-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Stairway-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Stairway-H1-v0 --headless --seed 999 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Stairway-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Stairway-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Stairway-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Stairway-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Stairway-H1-v0 --headless --seed 999 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./separate_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 999 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 42 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 123 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 456 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 789 && \
/home/whitebot/Workspace/Research/IsaacLab/isaaclab.sh -p ./shared_train.py --task Isaac-Velocity-Flat-H1-v0 --headless --seed 999
```

```
python3 Play_DAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 100.0 --push_interval 10 && \
python3 Play_UAC_Perturbation.py --task Isaac-Velocity-Flat-H1-Play-v0 --push_force 100.0 --push_interval 10 && \
python3 Play_DAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 100.0 --push_interval 10 && \
python3 Play_UAC_Perturbation.py --task Isaac-Stairway-H1-Play-v0 --push_force 100.0 --push_interval 10
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



