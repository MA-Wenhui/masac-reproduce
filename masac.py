from core import device
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
import core as core
from utils.logx import EpochLogger
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2
import os
import pickle
import time
import argparse
import logging
import torch.nn.functional as F

import concurrent.futures

num_agent = 3
max_step_per_epoch = 25
obs_dim = 18
act_dim = 5

# update_after = 2000
# update_every = 50
# start_steps=1024

update_after = 50000
update_every = 10
start_steps = 50000

batch_size = 1024
env_name = "simple_spread_v2"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def print_with_color(*args, color='white'):
    colors = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m'
    }
    reset = '\033[0m'
    if color in colors:
        print(colors[color], end='')
    print(*args, end='')
    print(reset)


class SAC:
    def __init__(self, agent_name, agent_id, obs_dim,result_dir, agent_n, act_dim, act_limit, replay_size,hidden_sizes=[64, 64],
                 gamma=0.99,
                #   alpha=0.0,
                 alpha=0.1,
                 lr=0.0003,
                 polyak=0.995,
                 ):
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.agent_n = agent_n
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.polyak = polyak

        self.ac = core.MLPActorCritic(
            obs_dim, act_dim, act_limit, hidden_sizes=hidden_sizes, agent_num=agent_n)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=lr)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # update actor
    def compute_loss_pi(self,data_n):
        o_n = torch.cat([data_i['obs'] for data_i in data_n], dim=1)
        a_n = torch.cat([data_i['act'] for data_i in data_n], dim=1)
        
        pi, logp_pi = self.ac.pi(data_n[self.agent_id]['obs'])
        start_column = self.agent_id*5
        a_n[:, start_column:start_column+5] = pi[:, :]
        
        q1_pi = self.ac.q1(o_n, a_n)
        q2_pi = self.ac.q2(o_n, a_n)
        q_pi = torch.min(q1_pi, q2_pi)
        logger.info(f'{self.agent_name}: q1_pi: {q1_pi}')
        logger.info(f'{self.agent_name}: q2_pi: {q2_pi}')

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        logger.info(f'{self.agent_name}: self.alpha * logp_pi.mean(): {self.alpha * logp_pi.mean()}')
        logger.info(f'{self.agent_name}: q_pi.mean(): {q_pi.mean()}')
        logger.info(f'{self.agent_name}: actor loss: {loss_pi.item()}')
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi)

        return loss_pi, pi_info

    # update critic
    def compute_loss_q(self,data_n):
        o_n = torch.cat([data_i['obs'] for data_i in data_n], dim=1)
        a_n = torch.cat([data_i['act'] for data_i in data_n], dim=1)

        o2_n = torch.cat([data_i['obs2'] for data_i in data_n], dim=1)
        a2_n = torch.cat([data_i['act2'] for data_i in data_n], dim=1)

        r = data_n[self.agent_id]['rew']
        d = data_n[self.agent_id]['done']

        q1 = self.ac.q1(o_n, a_n)
        q2 = self.ac.q2(o_n, a_n)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(data_n[self.agent_id]['obs'])
            start_column = self.agent_id*5
            a2_n[:, start_column:start_column+5] = a2[:, :]

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2_n, a2_n)
            q2_pi_targ = self.ac_targ.q2(o2_n, a2_n)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * \
                (1 - d) * (q_pi_targ - self.alpha * logp_a2)
        
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        # print_with_color("\t\t",self.agent_name,"loss_q1:",loss_q1,color='green')
        loss_q2 = ((q2 - backup)**2).mean()
        # print_with_color("\t\t",self.agent_name,"loss_q2:",loss_q2,color='green')
        loss_q = loss_q1 + loss_q2
        logger.info(f'{self.agent_name}: loss_q1: {loss_q1.item()}')
        logger.info(f'{self.agent_name}: loss_q2: {loss_q2.item()}')
        logger.info(f'{self.agent_name}: loss_q: {loss_q.item()}')
        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach(),
                      Q2Vals=q2.detach())

        return loss_q, q_info


    def add_replay_buffer(self, obs, act, r, next_obs, done):
        self.replay_buffer.store(obs, act, r, next_obs, done)

def sample_agents(all_agents_sac):  
    data_n = []
    # sample all agent's buffer
    for agent_sac in all_agents_sac:
        data_n.append(agent_sac.replay_buffer.sample_batch(batch_size))

    # add all agent's next_act from each agent's current policy
    for agent_sac in all_agents_sac:
        a2_list, logp_a2_list = agent_sac.ac.pi(data_n[agent_sac.agent_id]['obs'])
        data_n[agent_sac.agent_id]['act2'] = a2_list
    return data_n
        

def update_agents(all_agents_sac):
    # start update
    for agent_sac in all_agents_sac:
        data_n = sample_agents(all_agents_sac)
        # update critic 
        agent_sac.q_optimizer.zero_grad()
        loss_q, q_info = agent_sac.compute_loss_q(data_n)
        # print("\t\tloss_q",loss_q)
        loss_q.backward()
        # torch.nn.utils.clip_grad_norm_(agent_sac.q_params, 0.5)
        agent_sac.q_optimizer.step()
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in agent_sac.q_params:
            p.requires_grad = False

        # update actor
        agent_sac.pi_optimizer.zero_grad()
        loss_pi, pi_info = agent_sac.compute_loss_pi(data_n)
        # print("\t\tloss_pi",loss_pi)
        loss_pi.backward()
        # torch.nn.utils.clip_grad_norm_(agent_sac.ac.pi.parameters(), 0.5)
        agent_sac.pi_optimizer.step()
        for p in agent_sac.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(agent_sac.ac.parameters(), agent_sac.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(agent_sac.polyak)
                p_targ.data.add_((1 - agent_sac.polyak) * p.data)
        

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.act2_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     act2=self.act2_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def get_action(o, agent_sac, deterministic=False):
    return agent_sac.ac.act(torch.as_tensor(o, dtype=torch.float32),
                            deterministic)


def get_env(e, ep_len=25, render_mode='rgb_array'):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if e == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(
            max_cycles=ep_len, continuous_actions=True, render_mode=render_mode)
    if e == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(
            max_cycles=ep_len, continuous_actions=True, render_mode=render_mode)
    if e == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(
            max_cycles=ep_len, continuous_actions=True, render_mode=render_mode)

    new_env.reset(options={'landmarks_pos': [np.array([1.0, 1.0]), np.array(
        [1.0, -1.0]), np.array([-1.0, -1.0])]})
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(
            new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])

    return new_env, _dim_info


def get_running_reward(arr: np.ndarray, window=100):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(arr)
    for i in range(window - 1):
        running_reward[i] = np.mean(arr[:i + 1])
    for i in range(window - 1, len(arr)):
        running_reward[i] = np.mean(arr[i - window + 1:i + 1])
    return running_reward



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--render_mode', type=str, default='rgb_array', help='render mode',
                        choices=['human', 'rgb_array'])
    parser.add_argument('--exchange_pos', type=str, default='n', help='exchange agent pos',
                        choices=['y', 'n'])
    parser.add_argument('--replay_size', type=int, default=1000000, help='max size')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)
    
    logger = setup_logger(os.path.join(result_dir, 'masac.log'))

    
    env, dim_info = get_env(env_name, max_step_per_epoch, args.render_mode)

    all_agents = env.agents
    agents_sac = [SAC(agent_name=agent, agent_id=idx, obs_dim=obs_dim, act_dim=act_dim, agent_n=num_agent,
                      act_limit=env.action_space(agent).high[0],result_dir=result_dir,replay_size=args.replay_size) for idx, agent in enumerate(env.agents)]
    step = 0  # global step counter
    agent_num = env.num_agents
    episode_num1 = 30000
    episode_num2 = 70000
    episode_num = episode_num1 + episode_num2
    episode_rewards = {agent_id: np.zeros(
        episode_num) for agent_id in env.agents}
    episode = 0

    t_start = time.time()


    for episode in range(episode_num):
        if args.exchange_pos == 'y':
            options = {'landmarks_pos': [np.array([1.0, 1.0]), np.array(
                [1.0, -1.0]), np.array([-1.0, -1.0])],
                'agents_pos': [np.array([0.5, 0.5]), np.array(
                    [0.5, -0.5]), np.array([-0.5, -0.5])]} if episode < episode_num1 else {'landmarks_pos': [np.array([1.0, 1.0]), np.array(
                        [1.0, -1.0]), np.array([-1.0, -1.0])],
                'agents_pos': [np.array([-0.5, -0.5]), np.array(
                    [0.5, 0.5]), np.array([0.5, -0.5])]}
            obs = env.reset(options=options)
        else:
            obs = env.reset()

        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < start_steps:
                action = {agent_id: env.action_space(
                    agent_id).sample() for agent_id in env.agents}
            else:
                action = {agent_id: get_action(obs[agent_id],
                                               agents_sac[idx]) for idx, agent_id in enumerate(all_agents)}

            next_obs, reward, terminat, truncat, info = env.step(action)
            done = terminat or truncat
            # print(step,"\n------")
            # print("\t",action,"\n------")
            # print("\t",next_obs,"\n------")
            # print("\t",reward,"\n------")
            # print("\t",done,"\n------")
            for idx, agent_id in enumerate(all_agents):
                agents_sac[idx].add_replay_buffer(
                    obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= update_after and step % update_every == 0:  # learn every
                for j in range(update_every):
                    update_agents(agents_sac)
                # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                #     for j in range(update_every):
                #         futures = []
                #         for idx, agent in enumerate(all_agents):
                #             futures.append(executor.submit(update_agent ,idx, agents_sac))
                #         for future in concurrent.futures.as_completed(futures):
                #             result = future.result()

            obs = next_obs
        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}, '
            message += f'time:{round(time.time()-t_start, 3)}'
            print(message)
            t_start = time.time()

        if (episode + 1) % 1000 == 0:  # plot rewards
            fig, ax = plt.subplots()
            x = range(1, episode_num + 1)
            for agent_id, reward in episode_rewards.items():
                ax.plot(x, reward, label=agent_id)
                ax.plot(x, get_running_reward(reward))
            ax.legend()
            ax.set_xlabel('episode')
            ax.set_ylabel('reward')
            title = f'training result of masac solve {env_name}'
            ax.set_title(title)
            plt.savefig(os.path.join(result_dir, title))
            plt.close()

        if (episode) % 1000 == 0:
            torch.save(
                {sac.agent_name: sac.ac for sac in agents_sac},  # actor parameter
                os.path.join(result_dir, 'model_'+str(episode)+'.pt')
            )
            with open(os.path.join(result_dir, 'rewards.pkl'), 'wb') as f:  # save training data
                pickle.dump({'rewards': episode_rewards}, f)

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of masac solve {env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
    plt.close()
