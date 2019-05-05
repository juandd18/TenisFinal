import numpy as np
import random
import copy
from collections import namedtuple, deque

from network import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import soft_update, hard_update, gumbel_softmax, onehot_from_logits, OUNoise,ReplayBuffer,ReplayBufferOption

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent():

    def __init__(self,  num_in_pol, num_out_pol, num_in_critic, hidden_dim_actor=120,
    hidden_dim_critic=64,lr_actor=0.001,lr_critic=0.001,batch_size=128,
    max_episode_len=100,tau=0.001,gamma = 0.99,agent_name='one', discrete_action=False,random_seed=15):
    
        self.state_size = num_in_pol
        self.action_size = num_out_pol
        self.seed = random_seed
        random.seed(self.seed)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.policy = Actor(num_in_pol, num_out_pol, self.seed).to(device)
        self.target_policy = Actor(num_in_pol, num_out_pol, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)


        self.critic = Critic(num_in_pol, num_out_pol, self.seed).to(device)
        self.target_critic = Critic(num_in_pol, num_out_pol, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.noise = OUNoise(num_out_pol, self.seed)

        self.replay_buffer = ReplayBufferOption(600000,self.batch_size,15)

        

    def step(self, state, action, reward, next_state, done,t_step):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.batch_size*2:
            experiences = self.replay_buffer.sample()
            self.update(experiences, self.gamma)

    def act(self, state, explore=True):
        state = torch.from_numpy(state).float().to(device)
        self.policy.eval()
        with torch.no_grad():
            action = self.policy(state).cpu().data.numpy()
        self.policy.train()
        if explore:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset_noise(self):
        self.noise.reset()

    def update(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)


        next_actions = self.target_policy(next_states)
        q_targets_next = self.target_critic(next_states, next_actions)

        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.51)
        self.critic_optimizer.step()

        predicted_actions = self.policy(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.51)
        self.actor_optimizer.step()

        soft_update(self.critic, self.target_critic, self.tau)
        soft_update(self.policy, self.target_policy, self.tau)



