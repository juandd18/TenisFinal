#few changes to 
#https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    Neural (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=128, seed=15, nonlin=nn.ELU):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.bn4 = nn.BatchNorm1d(hidden_dim*2)
        self.fc4 = nn.Linear(hidden_dim*2, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        """
        Inputs:
            state (PyTorch Matrix): Batch of observations
            action (PyTorch Matrix): Batch of actions
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        #selu = nn.SELU()
        
        x = self.bn0(state)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(self.bn4(x)))

class Critic(nn.Module):
    """
    Neural (can be used as value or policy)
    """
    def __init__(self, input_dim,  action_size, hidden_dim=128, nonlin=nn.ELU,seed=15):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+action_size, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc4 = nn.Linear(hidden_dim*2, 1)
        self.nonlin = nonlin
        self.reset_parameters()


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state,actions):
        """
        Inputs:
            state (PyTorch Matrix): Batch of observations
            action (PyTorch Matrix): Batch of actions
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        #selu = nn.SELU()
        input_d = self.fc1(self.bn0(state))
        h1 = F.relu(input_d)
        x_join = torch.cat((h1, actions), dim=1)
        
        input_t = self.fc2(x_join)
        h2 = F.relu(input_t)
        h3 = F.relu(self.fc3(h2))
        final = self.fc4(h3)
        return final