# Anti-Concentrated Confidence Bonuses (ACB)

This repository contains a PyTorch implementation of the anti-concentrated confidence bonus for promoting exploration in deep reinforcement learning. For more information, check out our ICLR 2022 paper, [Anti-Concentrated Confidence Bonuses for Scalable Exploration](https://arxiv.org/abs/2110.11202). Our code was built off of [jcwleo's impelementation of Random Network Distillation](https://github.com/jcwleo/random-network-distillation-pytorch). 

# Dependencies

Required dependencies can be found in `setup.py`.

# Running an experiment


`python run.py --intrinsic acb --env breakout`\
runs an experiment using ACB in the Atari game Breakout.

`python run.py --intrinsic rnd --env seaquest --extrinsic`\
runs an experiment using RND in the atari game Seaquest. The `extrinsic` flag allows the agent to be trained jointly on intrinsic and extrinsic rewards; by default only the specified intrinsic rewards are used.


