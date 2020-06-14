import os, sys, argparse

import numpy as np
import torch

from ppo.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from ppo.a2c_ppo_acktr.utils import get_vec_normalize

# workaround to unpickle old model files
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

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                    None, None, args.add_timestep, device='cpu', allow_early_resets=False)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

iteration = 0
rewards = []
forces = []
task_successes = []
for iteration in range(100):
    done = False
    reward_total = 0.0
    force_total = 0.0
    force_list = []
    task_success = 0.0
    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(obs, recurrent_hidden_states, masks, deterministic=args.det)
        iteration += 1

        # Obser reward and next obs
        obs, reward, done, info = env.step(action)
        reward = reward.numpy()[0, 0]
        reward_total += reward
        force_list.append(info[0]['total_force_on_human'])
        task_success = info[0]['task_success']

        masks.fill_(0.0 if done else 1.0)

    rewards.append(reward_total)
    forces.append(np.mean(force_list))
    task_successes.append(task_success)
    print('Reward total:', reward_total, 'Mean force:', np.mean(force_list), 'Task success:', task_success)
    sys.stdout.flush()

print('Rewards:', rewards)
print('Reward Mean:', np.mean(rewards))
print('Reward Std:', np.std(rewards))

print('Forces:', forces)
print('Force Mean:', np.mean(forces))
print('Force Std:', np.std(forces))

print('Task Successes:', task_successes)
print('Task Success Mean:', np.mean(task_successes))
print('Task Success Std:', np.std(task_successes))
sys.stdout.flush()

