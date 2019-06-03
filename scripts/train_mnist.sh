#!/bin/bash
python train.py \
  --task mnist \
  --q 20 \
  --Hf 50 \
  --amort_len 3 \
  --batch_size 40 \
  --activation_fn "relu" \
  --eta 0.001 \
  --gamma 0.1 \
  --NF_enc 8 \
  --NF_dec 16