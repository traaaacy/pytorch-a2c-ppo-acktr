import os, sys, argparse

import numpy as np
import torch

from ppo.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from ppo.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


# Workaround to unpickle old model files
import ppo.a2c_ppo_acktr
sys.modules['a2c_ppo_acktr'] = ppo.a2c_ppo_acktr
sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='ScratchItchJaco-v0',
                    help='environment to train on (default: ScratchItchJaco-v0)')
parser.add_argument('--load-dir', default='./trained_models/ppo/',
                    help='directory to save agent logs (default: ./trained_models/ppo/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None,
                    args.add_timestep, device='cpu', allow_early_resets=False)

# Determine the observation lengths for the robot and human, respectively
obs = env.reset()
action = torch.tensor([env.action_space.sample()])
_, _, _, info = env.step(action)
dual_robots = 'dual_robots' in info[0]
if dual_robots:
    obs_robot_len = info[0]['obs_robot_len'] // 2
    action_robot_len = info[0]['action_robot_len'] // 2
    obs_robot1 = obs[:, :obs_robot_len]
    obs_robot2 = obs[:, obs_robot_len:]
    if len(obs_robot1[0]) != obs_robot_len or len(obs_robot2[0]) != obs_robot_len:
        print('robot 1 obs shape:', len(obs_robot1[0]), 'obs space robot shape:', (obs_robot_len,))
        print('robot 2 obs shape:', len(obs_robot2[0]), 'obs space robot shape:', (obs_robot_len,))
        exit()

env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None,
                    args.add_timestep, device='cpu', allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
if dual_robots:
    actor_critic_robot1, actor_critic_robot2, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
else:
    actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

if dual_robots:
    recurrent_hidden_states_robot1 = torch.zeros(1, actor_critic_robot1.recurrent_hidden_state_size)
    recurrent_hidden_states_robot2 = torch.zeros(1, actor_critic_robot2.recurrent_hidden_state_size)
else:
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()
if dual_robots:
    obs_robot1 = obs[:, :obs_robot_len]
    obs_robot2 = obs[:, obs_robot_len:]

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    with torch.no_grad():
        if dual_robots:
            value_robot1, action_robot1, _, recurrent_hidden_states_robot1 = actor_critic_robot1.act(
                obs_robot1, recurrent_hidden_states_robot1, masks, deterministic=args.det)
            value_robot2, action_robot2, _, recurrent_hidden_states_robot2 = actor_critic_robot2.act(
                obs_robot2, recurrent_hidden_states_robot2, masks, deterministic=args.det)
        else:
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    if dual_robots:
        action = torch.cat((action_robot1, action_robot2), dim=-1)
        obs, reward, done, infos = env.step(action)
        obs_robot1 = obs[:, :obs_robot_len]
        obs_robot2 = obs[:, obs_robot_len:]
    else:
        obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
