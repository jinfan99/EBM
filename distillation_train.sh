#!/bin/bash

# wandb online 

CUDA_VISIBLE_DEVICES=0 python my_train.py \
  --batch_size 16  \
  --energy_mode \
  --noise_schedule 'squaredcos_cap_v2' \
  --seed 42 \
  --test_guidance_scale 8 \
  --num_iters 100000 \
  --log_frequency 50 \
  --num_classes "" \
  --learning_rate 1e-04 \
  --checkpoints_dir './checkpoints/Energy_object_chkpt_shapenet_distill_1000' \
  --outputs_dir './checkpoints/Energy_object_chkpt_shapenet_distill_1000' \
  --shapenet  \
  --uncond 
