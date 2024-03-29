from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe
import pdb
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import hashlib
from nn_grads_proj import *

def main():
    if args.env == 'montezuma': default_config['EnvID'] = 'MontezumaRevengeNoFrameskip-v4'
    if args.env == 'asterix': default_config['EnvID'] = 'AsterixNoFrameskip-v4'
    if args.env == 'breakout': default_config['EnvID'] = 'BreakoutNoFrameskip-v4'
    if args.env == 'pong': default_config['EnvID'] = 'PongNoFrameskip-v4'
    if args.env == 'beamrider': default_config['EnvID'] = 'BeamRiderNoFrameskip-v4'
    if args.env == 'qbert': default_config['EnvID'] = 'QbertNoFrameskip-v4'
    if args.env == 'riverraid': default_config['EnvID'] = 'RiverraidNoFrameskip-v4'
    if args.env == 'seaquest': default_config['EnvID'] = 'SeaquestNoFrameskip-v4'
    if args.env == 'spaceinvaders': default_config['EnvID'] = 'SpaceInvadersNoFrameskip-v4'
    if args.env == 'pitfall': default_config['EnvID'] = 'PitfallNoFrameskip-v4'
    if args.env == 'gravitar': default_config['EnvID'] = 'GravitarNoFrameskip-v4'
    if args.env == 'solaris': default_config['EnvID'] = 'SolarisNoFrameskip-v4'
    if args.env == 'privateeye': default_config['EnvID'] = 'PrivateEyeNoFrameskip-v4'
    if args.env == 'venture': default_config['EnvID'] = 'VentureNoFrameskip-v4'
    train_method = default_config['TrainMethod']


    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    print({section: dict(config[section]) for section in config.sections()})

    if env_type == 'mario':
        env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2
    env.close()
    sz = 84

    if 'Breakout' in env_id:
        output_size -= 1

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])
    eta = float(default_config['ETA'])
    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, sz, sz))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    pre_obs_norm_step = 0
    if args.intrinsic == 'rnd': pre_obs_norm_step = 50
    discounted_reward = RewardForwardFilter(int_gamma)

    agent = RNDAgent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        intrinsic=args.intrinsic,
        extrinsic=args.extrinsic,
        numAux=args.numAux,
        var=args.var,
        lr=args.lr,
        optFun=args.optFun,
        tau=args.tau,
        iterateAve=args.iterateAve,
        initPull=args.initPull,
    )

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError



    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, sz, sz])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    allStates = set()
    allStatesFull = set()
    print('random exploration for state normalization...')
    next_obs = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr, sf = parent_conn.recv()
            next_obs.append(s[3, :, :].reshape([1, sz, sz]))
            allStates.add(hashlib.sha1(s[3,:,:]).hexdigest())
            allStatesFull.add(hashlib.sha1(s).hexdigest())

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('done')
    #allStates = set()
    #allStatesFull = set()
    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np, total_full_states = \
            [], [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1
        burnIn = 0
        if 'rnd_sk' in args.intrinsic: burnIn = args.burnIn
        # Step 1. n-step rollout
        for _ in range(num_step):

            if 'curious' in args.intrinsic: actions, value, policy = agent.get_action(np.float32(states) / 255.)
            else: actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)
            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs, full_states = [], [], [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr, sf = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s[3, :, :].reshape([1, sz, sz]))
                allStates.add(hashlib.sha1(s[3,:,:]).hexdigest())
                allStatesFull.add(hashlib.sha1(s).hexdigest())
                full_states.append(sf)


            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)
            full_states = np.stack(full_states)
            

            # total reward = int reward + ext Reward
            if args.intrinsic == 'rnd':
                intrinsic_reward = agent.compute_intrinsic_reward(
                    ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                intrinsic_reward = np.hstack(intrinsic_reward)
            if args.intrinsic == 'acb': 
                intrinsic_reward = agent.compute_intrinsic_reward(np.float32(next_states) / 255, obs=None, actions=None)
                if global_update < burnIn: intrinsic_reward[:] = 0.
            if args.intrinsic == 'rand':
                intrinsic_reward = np.random.rand(len(states))
             
            sample_i_rall += intrinsic_reward[sample_env_idx]
            #print('avBonus', np.mean(intrinsic_reward), flush=True)
            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            if 'curious' not in args.intrinsic: 
                total_ext_values.append(value_ext)
                total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())
            total_full_states.append(full_states)


            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0


        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)


        # --------------------------------------------------
        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, sz, sz])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, sz, sz])
        total_logging_policy = np.vstack(total_policy_np)
        total_int_values = np.stack(total_int_values).transpose()
        total_ext_values = np.stack(total_ext_values).transpose()



        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)

        ######################
        norm1 = torch.norm(torch.cat([p.view(-1) for p in agent.auxWeights.parameters()])).item()
        norm2 = torch.norm(torch.cat([p.view(-1) for p in agent.auxWeightsAv.parameters()])).item()
        #####################


        print('stats', global_update, reward_rms.var, reward_rms.mean, np.mean(rewards), np.mean(intrinsic_reward), np.mean(total_int_reward), np.max(total_int_reward), np.min(total_int_reward), norm1, norm2, flush=True)
        print('progress', global_update, sample_episode, sample_step, len(allStates), len(allStatesFull), flush=True)
        if global_update >= burnIn: reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)


        # add ext adv and int adv
        total_adv = int_adv * int_coef 
        if args.extrinsic: total_adv = total_adv + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        if args.intrinsic == 'rnd':
            agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                              total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                              total_policy)

        else:
            agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                              total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                              total_policy)

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy', help='dummy input, useful for running multiple iterates', type=int, default=0)
    parser.add_argument('--intrinsic', help='rnd or acb', type=str, default='none')
    parser.add_argument('--env', help='', type=str, default='breakout')
    parser.add_argument('--extrinsic', help='use extrinsic signal?', action='store_true')
    parser.add_argument('--burnIn', help='how long to let bonus ''burn in'' before use', type=int, default=0)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--initPull', help='aux weight regularization penalty (towards initialization)', type=float, default=1e3)
    parser.add_argument('--tau', help='tail average parameter', type=float, default=1e-6)
    parser.add_argument('--optFun', help='aux weight optimizer', type=str, default='rmsprop')
    parser.add_argument('--var', help='noise variance', type=float, default=1)
    parser.add_argument('--numAux', help='ensemble size', type=int, default=128)
    parser.add_argument('--iterateAve', help='use iterate averaging on aux weights?', type=int, default=1)

    args = parser.parse_args()

    main()
