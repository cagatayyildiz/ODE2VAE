#!/bin/bash
python train.py \
  --task mnist_nonuniform \
  --q 8 \
  --Hf 50 \
  --amort_len 1 \
  --batch_size 60 \
  --activation_fn "relu" \
  --eta 0.001 \
  --gamma 0.1 \
  --NF_enc 8 \
  --NF_dec 12
