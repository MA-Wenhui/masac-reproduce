from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
import spinup.algos.pytorch.masac.core as core
from spinup.utils.logx import EpochLogger
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2
import os
import pickle
import time

import concurrent.futures

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs("masac", 0)

logger = EpochLogger(**logger_kwargs)
logger.save_config(locals())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

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
    def __init__(self, agent_name, obs_dim, agent_n, act_dim, act_limit, hidden_sizes=[64, 64],
                 gamma=0.99,
                 alpha=0.1,
                 lr=0.0003,
                 polyak=0.995):
        self.agent_name = agent_name
        self.agent_n = agent_n
        self.ac = core.MLPActorCritic(
            obs_dim*agent_n, act_dim*agent_n, act_limit, hidden_sizes=hidden_sizes)
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.polyak = polyak
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters())

        var_counts = tuple(core.count_vars(module)
                           for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        logger.log(
            '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)
        replay_size = 1000000
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_pi(self, data_n):
        o_n = torch.cat([data_i['obs'] for data_i in data_n],dim=1)
        pi, logp_pi = self.ac.pi(o_n)
        q1_pi = self.ac.q1(o_n, pi)
        q2_pi = self.ac.q2(o_n, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # print_with_color("\t\t",self.agent_name,"loss_pi:",loss_pi,color='blue')
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def compute_loss_q(self, data_n,data_self):
        o_n = torch.cat([data_i['obs'] for data_i in data_n],dim=1)
        a_n= torch.cat([data_i['act'] for data_i in data_n],dim=1)
        o2_n = torch.cat([data_i['obs2'] for data_i in data_n],dim=1)

        r = data_self['rew']
        d = data_self['done']
        
        q1 = self.ac.q1(o_n, a_n)
        q2 = self.ac.q2(o_n, a_n)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2_n, logp_a2_n = self.ac.pi(o2_n)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2_n, a2_n)
            q2_pi_targ = self.ac_targ.q2(o2_n, a2_n)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r.to(device) + self.gamma * \
                (1 - d.to(device)) * (q_pi_targ - self.alpha * logp_a2_n)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        # print_with_color("\t\t",self.agent_name,"loss_q1:",loss_q1,color='green')
        # print_with_color("\t\t",self.agent_name,"loss_q2:",loss_q2,color='green')

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    def update(self,data_n):
        # print(self.agent_name, "update in")
        # First run one gradient descent step for Q1 and Q2

        data_self = self.replay_buffer.sample_batch(batch_size)

        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data_n,data_self)
        # print("\t\tloss_q",loss_q)
        loss_q.backward()
        self.q_optimizer.step()
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data_n)
        # print(loss_pi.item(),pi_info)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return

    def add_replay_buffer(self, obs, act, r, next_obs, done):
        self.replay_buffer.store(obs, act, r, next_obs, done)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size,1), dtype=np.float32)
        self.done_buf = np.zeros((size,1), dtype=np.float32)
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
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def get_action(o, agent_sac, deterministic=False):
    tensor_dict = {}
    for key, value in o.items():
        tensor_dict[key] = torch.tensor(value)

    # Concatenate the tensors along the last dimension (axis=1)
    output_tensor = torch.cat(list(tensor_dict.values()), dim=0)
    return agent_sac.ac.act(torch.as_tensor(output_tensor, dtype=torch.float32),
                            deterministic)




def get_env(env_name, ep_len=25,render_mode='rgb_array'):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len,continuous_actions=True,render_mode=render_mode)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len,continuous_actions=True,render_mode=render_mode)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles=ep_len,continuous_actions=True,render_mode=render_mode)

    new_env.reset(options={'landmarks_pos':[np.array([5.0, 5.0]), np.array(
        [5.0, -5.0]), np.array([-5.0, -5.0])]})
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        print(new_env.action_space(agent_id).shape[0])
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
    num_agent = 3
    max_step_per_epoch = 25
    obs_dim = 18
    act_dim = 5
    # update_after = 1000
    # update_every = 100
    # start_steps=10000
    update_after = 2000
    update_every = 10
    start_steps = 10000
    
    batch_size = 1024
    env_name="simple_spread_v2"
    # create folder to save result
    env_dir = os.path.join('./results', env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(env_name,max_step_per_epoch)

    all_agents = env.agents
    agents_sac = [SAC(agent, obs_dim=obs_dim, act_dim=act_dim, agent_n=num_agent,
                      act_limit=env.action_space(agent).high[0]) for idx, agent in enumerate(env.agents)]
    step = 0  # global step counter
    agent_num = env.num_agents 
    episode_num1 = 300000
    episode_num2 = 700000 
    episode_num = episode_num1+ episode_num2
    episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}
    episode = 0
    
    t_start = time.time()
    options1={'landmarks_pos':[np.array([5.0, 5.0]), np.array(
        [5.0, -5.0]), np.array([-5.0, -5.0])],
        'agents_pos':[np.array([2.0, 2.0]), np.array(
        [2.0, -2.0]), np.array([-2.0, -2.0])]}

    options2={'landmarks_pos':[np.array([5.0, 5.0]), np.array(
        [5.0, -5.0]), np.array([-5.0, -5.0])],
        'agents_pos':[np.array([-2.0, -2.0]), np.array(
        [2.0, 2.0]), np.array([2.0, -2.0])]}
    
    for episode in range(episode_num):
        obs = env.reset(options= options1 if episode<episode_num1 else options2)
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < start_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = {agent_id: np.reshape(get_action(obs,
                               agents_sac[idx]), (3, 5))[idx] for idx,agent_id in enumerate(all_agents)}

            next_obs, reward, terminat , truncat, info = env.step(action)
            done = terminat or truncat
            # print(step,"\n------")
            # print("\t",action,"\n------")
            # print("\t",next_obs,"\n------")
            # print("\t",reward,"\n------")
            # print("\t",done,"\n------")
            for idx,agent_id in enumerate(all_agents):
                agents_sac[idx].add_replay_buffer(
                    obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= update_after and step % update_every == 0:  # learn every 
                for j in range(update_every):
                    for idx, agent in enumerate(all_agents):
                        data_n = [agents_sac[idx].replay_buffer.sample_batch(batch_size) for idx in range(num_agent)]
                        agents_sac[idx].update(data_n)

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

        if (episode) % 10000 == 0:
            torch.save(
                {sac.agent_name: sac.ac for sac in agents_sac},  # actor parameter
                os.path.join(result_dir, 'model_'+str(episode)+'.pt')
            )
            with open(os.path.join(result_dir, 'rewards.pkl'), 'wb') as f:  # save training data
                pickle.dump({'rewards': episode_rewards}, f)
            
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

