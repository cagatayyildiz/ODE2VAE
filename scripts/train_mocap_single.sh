#!/bin/bash
python train.py \
  --task mocap_single \
  --q 3 \
  --Hf 30 \
  --He 30 \
  --Hd 30 \
  --amort_len 3 \
  --batch_size 10 \
  --activation_fn "tanh" \
  --eta 0.003
