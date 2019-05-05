import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from network import Actor,Critic
from utils import soft_update, hard_update, gumbel_softmax, onehot_from_logits, OUNoise,ReplayBuffer,ReplayBufferOption

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MSELoss = torch.nn.MSELoss()

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim_actor=120,
    hidden_dim_critic=64,lr_actor=0.01,lr_critic=0.01,batch_size=64,
    max_episode_len=100,tau=0.02,gamma = 0.99,agent_name='one', discrete_action=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = Actor(num_in_pol, num_out_pol,hidden_dim=hidden_dim_actor).to(device)
        self.critic = Critic(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim_critic).to(device)
        self.target_policy = Actor(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim_actor).to(device)
        self.target_critic = Critic(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim_critic).to(device)
        

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)

        self.policy = self.policy.cuda()
        self.critic = self.critic.cuda()
        self.target_policy = self.target_policy.cuda()
        self.target_critic = self.target_critic.cuda()

        self.agent_name = agent_name
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        #self.replay_buffer = ReplayBuffer(1e7)
        self.replay_buffer = ReplayBufferOption(600000,self.batch_size,15)
        self.max_replay_buffer_len = batch_size * max_episode_len
        self.replay_sample_index = None
        self.niter = 0
        self.eps = 8.0
        self.eps_decay = 1/(150*5)

        self.exploration = OUNoise(num_out_pol)
        self.discrete_action = discrete_action

        self.num_history = 2
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def act(self, obs, explore=True):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs : Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        state = torch.from_numpy(obs).float().to(device)
        
        self.policy.eval()
        with torch.no_grad():
            self.policy = self.policy.cuda()
            action = self.policy(state)
            action = action.cpu().data.numpy()
        self.policy.train()
        # continuous action
        if explore:
            action +=  self.exploration.sample()
        return np.clip(action, -1, 1)

    def step(self, agent_id, state, action, reward, next_state, done,t_step):
    

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        #self.replay_buffer.add(state, action, reward, next_state, done)
        if t_step % self.num_history == 0:
            # Save experience / reward
            
            self.replay_buffer.add(self.states, self.actions, self.rewards, self.next_states, self.dones)
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []

        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > self.batch_size:
            
            obs, acs, rews, next_obs, don = self.replay_buffer.sample()     
            self.update(agent_id ,obs,  acs, rews, next_obs, don,t_step)
        


    def update(self, agent_id, obs, acs, rews, next_obs, dones ,t_step, logger=None):
    
        self.target_policy = self.target_policy.cuda()
        self.target_critic = self.target_critic.cuda()
        self.critic = self.critic.cuda()
        
        obs = torch.from_numpy(obs).float().to(device)
        acs = torch.from_numpy(acs).float().to(device)
        rews = torch.from_numpy(rews[:,agent_id]).float().to(device)
        next_obs = torch.from_numpy(next_obs).float().to(device)
        dones = torch.from_numpy(dones[:,agent_id]).float().to(device)

        acs = acs.view(-1,2)
                
        # --------- update critic ------------ #        
        
        
        all_trgt_acs = self.target_policy(next_obs) 
        q_trgt_nxt = self.target_critic(next_obs,all_trgt_acs)
    
        target_value = rews + (self.gamma * q_trgt_nxt * (1 - dones)) 
 
        actual_value = self.critic(obs,acs)
        vf_loss = F.mse_loss(actual_value, target_value.detach())

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.800)
        self.critic_optimizer.step()

        # --------- update actor --------------- #
        curr_pol_out = self.policy(obs)

        pol_loss = -self.critic(obs,curr_pol_out).mean()
        
        self.policy_optimizer.zero_grad()
        pol_loss.backward()

        #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.800)
        self.policy_optimizer.step()

        self.update_all_targets()
        self.eps -= self.eps_decay
        self.eps = max(self.eps, 0)
        

        if logger is not None:
            logger.add_scalars('agent%i/losses' % self.agent_name,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        
        soft_update(self.critic, self.target_critic, self.tau)
        soft_update(self.policy, self.target_policy, self.tau)
   
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])