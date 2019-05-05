# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed=15, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        self.state = copy.copy(self.mu)
        random.seed(self.seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() - .5 for i in range(len(x))])
        self.state = x + dx
        return self.state