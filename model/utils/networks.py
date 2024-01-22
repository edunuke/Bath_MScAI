"""
Author: Eduardo Perez Denadai
Date: 2022-10-25
Reference:
1. https://stable-baselines3.readthedocs.io/en/v2.1.0/guide/custom_policy.html#custom-feature-extractor
"""

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor
from gymnasium.spaces import Space


class GRUNetwork(BaseFeaturesExtractor):
    """
    GRU-based feature extractor for reinforcement learning agents.

    Args:
        observation_space (Space): The observation space of the environment.
        first_layer (int, optional): The size of the first layer. Defaults to 300.
        output_layer (int, optional): The size of the output feature dimension. Defaults to 200.
    """

    def __init__(self, 
                 observation_space: Space, 
                 first_layer: int = 300, 
                 first_layer_dropout: float = 0.2,
                 output_layer: int = 100,
                 output_layer_dropout: float = 0.2
                 ):
        
        super(GRUNetwork, self).__init__(observation_space, output_layer)

        self.pre_gru = nn.Sequential(
            nn.Linear(observation_space.shape[0], first_layer, bias=False),
            nn.BatchNorm1d(first_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=first_layer_dropout)
        )

        self.gru = nn.GRU(first_layer,
                          output_layer,
                          batch_first=True,
                          bias=False)

        self.post_gru = nn.Sequential(
            nn.Linear(output_layer, output_layer),
            nn.BatchNorm1d(output_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=output_layer_dropout)
        )

    def forward(self, observations: Tensor) -> Tensor:
        """
        Forward pass through the GRU-based feature extractor.

        Args:
            observations (Tensor): Input observations.

        Returns:
            Tensor: Extracted features.
        """
        x = self.pre_gru(observations)
        x = x.unsqueeze(1)
        x, _ = self.gru(x)
        x = x.squeeze(1)
        x = self.post_gru(x)
        return x


class GRUNetworkBidirectional(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: Space, 
                 first_layer: int = 300, 
                 first_layer_dropout: float = 0.2,
                 gru_hidden_layer: int = 50,  # Half of output_layer size for bidirectional
                 output_layer: int = 100,
                 output_layer_dropout: float = 0.2
                 ):
        super(GRUNetworkBidirectional, self).__init__(observation_space, output_layer)

        self.pre_gru = nn.Sequential(
            nn.Linear(observation_space.shape[0], first_layer, bias=False),
            nn.BatchNorm1d(first_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=first_layer_dropout)
        )

        # Note that the number of features for each direction in the GRU is half the output_layer size
        self.gru = nn.GRU(first_layer,
                          gru_hidden_layer,  # Half of output_layer size for bidirectional
                          batch_first=True,
                          bidirectional=True,  # Make the GRU bidirectional
                          bias=False)

        # The output will be twice the gru_hidden_layer because it's bidirectional
        self.post_gru = nn.Sequential(
            nn.Linear(2 * gru_hidden_layer, output_layer),  # Adjust the input size
            nn.BatchNorm1d(output_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=output_layer_dropout)
        )

    def forward(self, observations: Tensor) -> Tensor:
        x = self.pre_gru(observations)
        x = x.unsqueeze(1)
        x, _ = self.gru(x)
        x = x.squeeze(1)
        x = self.post_gru(x)
        return x




class LSTMNetwork(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for reinforcement learning agents.

    Args:
        observation_space (Space): The observation space of the environment.
        first_layer (int, optional): The size of the first layer. Defaults to 300.
        output_layer (int, optional): The size of the output feature dimension. Defaults to 200.
    """

    def __init__(self, 
                 observation_space: Space, 
                 first_layer: int = 300, 
                 first_layer_dropout: float = 0.2,
                 output_layer: int = 100,
                 output_layer_dropout: float = 0.2
                 ):
        
        super(LSTMNetwork, self).__init__(observation_space, output_layer)

        self.pre_lstm = nn.Sequential(
            nn.Linear(observation_space.shape[0], first_layer, bias=False),
            nn.BatchNorm1d(first_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=first_layer_dropout)
        )

        self.lstm = nn.LSTM(first_layer,
                            output_layer,
                            batch_first=True,
                            bias=False)

        self.post_lstm = nn.Sequential(
            nn.Linear(output_layer, output_layer),
            nn.BatchNorm1d(output_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=output_layer_dropout)
        )

    def forward(self, observations: Tensor) -> Tensor:
        """
        Forward pass through the LSTM-based feature extractor.

        Args:
            observations (Tensor): Input observations.

        Returns:
            Tensor: Extracted features.
        """
        x = self.pre_lstm(observations)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.post_lstm(x)
        return x


class FCNetwork(BaseFeaturesExtractor):
    """
    Fully connected (FC) network-based feature extractor for reinforcement learning agents.

    Args:
        observation_space (Space): The observation space of the environment.
        first_layer (int, optional): The size of the first layer. Defaults to 300.
        output_layer (int, optional): The size of the output feature dimension. Defaults to 200.
    """

    def __init__(self, 
                 observation_space: Space, 
                 first_layer: int = 300, 
                 first_layer_dropout: float = 0.2,
                 output_layer: int = 100,
                 output_layer_dropout: float = 0.1
                 ):
        
        super(FCNetwork, self).__init__(observation_space, output_layer)

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], first_layer),
            nn.BatchNorm1d(first_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=first_layer_dropout),
            nn.Linear(first_layer, output_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=output_layer_dropout),
        )

    def forward(self, observations: Tensor) -> Tensor:
        """
        Forward pass through the FC network-based feature extractor.

        Args:
            observations (Tensor): Input observations.

        Returns:
            Tensor: Extracted features.
        """
        return self.net(observations)


class ConvNetwork(BaseFeaturesExtractor):
    """
    Convolutional neural network-based feature extractor for reinforcement learning agents.

    Args:
        observation_space (Space): The observation space of the environment.
        first_layer (int, optional): The size of the first layer. Defaults to 300.
        output_layer (int, optional): The size of the output feature dimension. Defaults to 200.
    """

    def __init__(self, 
                 observation_space: Space, 
                 first_layer: int = 300, 
                 first_layer_dropout: float = 0.2,
                 output_layer: int = 100,
                 output_layer_dropout: float = 0.1
                 ):
        
        super(ConvNetwork, self).__init__(observation_space, output_layer)

        self.pre_layers = nn.Sequential(
            nn.Linear(observation_space.shape[0], first_layer, bias=True),
            nn.BatchNorm1d(first_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=first_layer_dropout)
        )

        self.conv = nn.Conv1d(in_channels=first_layer, out_channels=output_layer, kernel_size=3, stride=1, padding=1)

        self.post_layers = nn.Sequential(
            nn.Linear(output_layer, output_layer, bias=True),
            nn.BatchNorm1d(output_layer),
            nn.RReLU(lower=0.1, upper=0.5),
            nn.Dropout(p=output_layer_dropout)
        )

    def forward(self, observations: Tensor) -> Tensor:
        """
        Forward pass through the convolutional neural network-based feature extractor.

        Args:
            observations (Tensor): Input observations.

        Returns:
            Tensor: Extracted features.
        """
        x = self.pre_layers(observations)
        x = x.unsqueeze(2)
        x = self.conv(x)
        x = x.squeeze(2)
        x = self.post_layers(x)
        return x
