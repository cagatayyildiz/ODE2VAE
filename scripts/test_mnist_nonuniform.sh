#!/bin/bash
python test.py --data_root data --task mnist_nonuniform --ckpt checkpoints/mnist_nonuniform/mnist_nonuniform_q8_inst1_fopt2_enc8_dec12
# python test.py --data_root data --task mnist_nonuniform --ckpt checkpoints/mnist_nonuniform/neurips-results
