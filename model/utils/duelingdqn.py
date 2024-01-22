import numpy as np
import torch as th
import warnings
from typing import Any, Dict, List, Optional, Type, Union

from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork, BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor, create_mlp)
from stable_baselines3.common.type_aliases import (
    GymEnv, MaybeCallback, Schedule)
from stable_baselines3.common.utils import (
    get_linear_fn, get_parameters_by_name, polyak_update)

class DuelingQNetwork(QNetwork):
    """
    Dueling Q-Network architecture.
    """

    def __init__(self, 
                 observation_space: spaces.Space, 
                 action_space: spaces.Discrete, 
                 features_extractor: BaseFeaturesExtractor, 
                 features_dim: int, 
                 net_arch: Optional[List[int]] = None, 
                 activation_fn: Type[th.nn.Module] = th.nn.ReLU, 
                 normalize_images: bool = True):
        super(DuelingQNetwork, self).__init__(
            observation_space, action_space, features_extractor, 
            features_dim, net_arch, activation_fn, normalize_images)

        # Define the advantage and value streams
        self.advantage = create_mlp(
            features_dim, action_space.n, net_arch, activation_fn)
        self.value = create_mlp(
            features_dim, 1, net_arch, activation_fn)

        # Replace the original q_net with the dueling architecture
        self.q_net = th.nn.ModuleList([
            th.nn.Sequential(*self.advantage), 
            th.nn.Sequential(*self.value)])

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)

        # Compute advantage and value streams
        advantage = self.q_net[0](features)
        value = self.q_net[1](features)

        # Combine the streams
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class DuelingDQNPolicy(DQNPolicy):
    """
    Policy class with a Dueling Q-Network for Dueling DQN.
    """

    def make_q_net(self) -> DuelingQNetwork:
        # Make sure we always have separate networks
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)
