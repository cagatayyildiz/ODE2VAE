# ode2vae
TensorFlow implementation of [Deep generative second order ODEs with Bayesian neural networks](https://arxiv.org/pdf/1905.10994.pdf) by Çağatay Yıldız, Markus Heinonen and Harri Lahdesmäki.

We tackle the problem of learning low-rank latent representations of possibly high-dimensional sequential data trajectories. Our model extends Variational Auto-Encoders (VAEs) for sequential data with a latent space governed by a continuous-time probabilistic ordinary differential equation (ODE). We propose
1. a powerful second order ODE that allows modelling the latent dynamic ODE state decomposed as position and momentum
2. a deep Bayesian neural network to infer latent dynamics

<img src="main_fig.png" width="600px"/>

## Setup
The code is developed and tested on python3.7 and TensorFlow 1.13. [hickle](https://pypi.org/project/hickle/) library is also needed to load the datasets. 

Training and test scripts are placed in the [`scripts`](./scripts) directory. In order to run reproduce an experiment, run the following command from the project root folder:
```
./scripts/train_bballs.sh
```

## Datasets
The datasets can be downloaded from [here](https://www.dropbox.com/sh/q8l6zh2dpb7fi9b/AACX3OVDEBxjHMcwx_Ik6cyha?dl=0). The folders contain
1. preprocessed walking sequences from [CMU mocap library](http://mocap.cs.cmu.edu/)
2. rotating mnist dataset generated using [this implementation](https://github.com/ChaitanyaBaweja/RotNIST)
3. bouncing ball dataset generated using [the code](http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar) provided with the original paper.
