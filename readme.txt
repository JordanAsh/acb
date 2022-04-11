This code is based on https://github.com/jcwleo/random-network-distillation-pytorch, a popular PyTorch implementation of the original OpenAI Tensorflow version. The entry point is train.py, which allows the user to select the type of intrinsic reward (RND or ACB), the environment, and other variables. As an example,

python train.py --intrinsic acb --env breakout

will train an agent on the breakout environment using ACB bonuses. Replace "acb" with "rnd" to switch to RND. Adding "--extrinsic 1" allows the agent to also train on extrinsic rewards (in addition to the specified intrinsic bonus).

The default settings are consistent with what is used in our paper. For the ICM baseline, we used a popular PyTorch implementation, https://github.com/jcwleo/curiosity-driven-exploration-pytorch. Dependencies can be found in setup.py.
