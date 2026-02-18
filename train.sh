#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
config_file=$2
guidance_type=$3
guidance_scale=$4
phase_num=$5

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --opt ./configs/${guidance_type}/${config_file}.yaml \
    --guidance_type $guidance_type \
    --phase_num ${phase_num} \
    --num_ddim_timesteps 50 \
    --guid_scale $guidance_scale

# bash train.sh 0 spiderman sctd 7.5 5