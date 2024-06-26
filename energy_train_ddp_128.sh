#!/bin/bash

wandb online 

python -m torch.distributed.launch --nproc_per_node=4 --master_port=5672  --use_env train_ddp.py \
  --project_name 'shapenet-EBM' \
  --data_dir '/nfs/turbo/jjparkcv-turbo-large/instant3d/instant-splatting/data/cars/' \
  --batch_size 16  \
  --energy_mode \
  --noise_schedule 'squaredcos_cap_v2' \
  --seed 42 \
  --test_guidance_scale 8 \
  --epochs 100 \
  --log_frequency 50 \
  --num_classes "" \
  --learning_rate 1e-04 \
  --checkpoints_dir './checkpoints/Energy_object_chkpt_shapenet_alldata' \
  --outputs_dir './checkpoints/Energy_object_chkpt_shapenet_alldata' \
  --shapenet  \
  --uncond 
