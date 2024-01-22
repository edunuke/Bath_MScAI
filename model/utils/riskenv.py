"""
Author: Eduardo Perez Denadai
Date: 2022-10-25
Description: Risk Management Environment
"""
import numpy as np
import pandas as pd
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from sklearn.utils import compute_class_weight
from sklearn.metrics import (f1_score, 
                             recall_score, 
                             accuracy_score)


class RiskManagementEnv(gym.Env):
    """
    Initializes the Gymnasium environment.

    Parameters:
            df (DataFrame): The input DataFrame containing the features and default column.
            debug (bool): A flag indicating whether to print debug information.
            scaled_features (bool): A flag indicating whether the features should be scaled.
            accepts_discrete_action (bool): A flag indicating whether the agent accepts discrete actions.
            features_col (list): A list of feature column names.
            default_col (str): The name of the default column.
            obs_dim (int): The dimension of the observation space.
            client_dim (int): The dimension of the client space.
            action_dim (int): The dimension of the action space.
            reward_delay_steps (int, optional): The number of steps to delay the reward. Defaults to 1.
            seed (int, optional): The random seed. Defaults to 123.
            model_name (str, optional): The name of the model. Defaults to "".

    Returns:
        None
     """
    def __init__(self, df, debug, scaled_features, accepts_discrete_action,
                 features_col, default_col, obs_dim, client_dim, action_dim,
                 rng, eward_delay_steps=1, seed=123, model_name=""):

        super().__init__()
        self.debug = debug
        self.rng = rng
        self.df = df[features_col + [default_col]].copy().astype(np.float32)
        self.client_list = self.df.loc[0].index.tolist()
        self.features = features_col
        self.default_col = default_col
        self.client_dim = client_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_delay_steps = 1
        self.action_outcome_history = deque(maxlen=self.reward_delay_steps)
        self.accepts_discrete_action = accepts_discrete_action
        self.state = None
        self.month = 0
        self.max_month = self.df.index.get_level_values(0).max()
        self.current_client_id = self.rng.choice(self.client_list)
        self.max_client_id = self.df.index.get_level_values(1).max()
        self.default_rate = 0.0
        self.model_name = model_name
        self.reward = 0
        self.action_hist = []
        self.penalty = 0
        self.count = 0
        self.step_counter = 0
        self.N_SAMPLES = 50
        self.sampled_clients = []
        self.action_space = self._define_action_space(self.accepts_discrete_action)
        self.observation_space = self._define_observation_space(scaled_features, features_col, df)

    def _define_action_space(self, accepts_discrete_action):
        """
        Define the action space for the agent.

        Parameters:
            accepts_discrete_action (bool): Whether the agent accepts discrete actions.

        Returns:
            spaces.Space: The defined action space.
        """
        if accepts_discrete_action:
            return spaces.Discrete(self.action_dim)
        else:
            return spaces.Box(low=0, high=1, shape=(self.action_dim - 1,), dtype="int")

    def _define_observation_space(self, scaled_features, features_col, df):
        """
        Generates the observation space for the agent based on the given scaled features, features column, and dataframe.

        Parameters:
            scaled_features (bool): A boolean value indicating whether the features should be scaled.
            features_col (str): The name of the column containing the features in the dataframe.
            df (pandas.DataFrame): The dataframe containing the features.

        Returns:
            gym.spaces.Box: The observation space for the agent.
        """
        if scaled_features:
            return spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        else:
            low = df[features_col].min().values
            high = df[features_col].max().values
            return spaces.Box(low=low, high=high, shape=(self.obs_dim,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Parameters:
        - seed (int or None): The random seed to use for initializing the environment. If None, a random seed will be used.
        - options (dict or None): Additional options for resetting the environment. If None, default options will be used.

        Returns:
        - state (numpy.ndarray): The initial state of the environment.
        - info (dict): Additional information about the reset.
        """
        self.reward = 0
        self.step_counter = 0
        self.action_outcome_history.clear()
        self.month = 0
        self.current_client_id = self.rng.choice(self.client_list)
        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        return self.state, {}

    def _calculate_delayed_reward(self,):
        """
        Calculate the delayed reward based on the action-outcome history.

        Returns:
            reward (int): The calculated delayed reward.

        Notes:
            - The delayed reward is calculated based on the action-outcome history.
            - The reward is determined based on the values of 'acts' (actions) and 'outs' (outcomes).
        """
        reward = 0
        acts =[]
        outs = []
        for action, outcome in self.action_outcome_history:
            acts.append(action)
            outs.append(outcome)

        # cost matrix 
        if (acts[0] == 0) & (outs[-1] == 0): #TN
            reward = 1+self.class_weight[1]
        if (acts[0] == 1) & (outs[-1] == 1): # TP
            reward = 2+self.class_weight[1]
        if (acts[0] == 0) &  (outs[-1] == 1): # FN
            reward = -2-self.class_weight[0]
        if (acts[0] == 1) & (outs[-1] == 0): #FP
            reward = -1-self.class_weight[0]
        return reward

    def _scale_reward(self, reward):
        """
        Scales a given reward value between 0 and 1 based on the minimum and maximum reward values.

        Parameters:
            reward (float): The reward value to be scaled.

        Returns:
            float: The scaled reward value between 0 and 1.
        """
        max_reward = 2+self.class_weight[1]
        min_reward = -2-self.class_weight[0]
        return (reward - min_reward) / (max_reward - min_reward)

    def _calculate_penalty(self):
        """
        Calculate the penalty for the current month based on the default rate and the action history.

        Returns:
            None
        """
        actual_default_rate = self.df.loc[self.month][self.default_col].sum() / self.N_SAMPLES
        self.default_rate = sum(self.action_hist) / self.N_SAMPLES

        if len(self.action_hist) < self.df.loc[self.month].shape[0]:
            self.action_hist.extend([0] * (self.df.loc[self.month].shape[0] - len(self.action_hist)))
        
        self.penalty = -(1/2) * abs(actual_default_rate - self.default_rate)
        
        actual_defaults = self.df.loc[self.month][self.default_col].values
        self.penalty += f1_score(actual_defaults, self.action_hist)
        self.penalty += recall_score(actual_defaults, self.action_hist)

        self.action_hist = []

    def step(self, action):
        """
        Updates the environment state based on the given action and returns the 
        updated state, reward, done flag, and additional information.

        Parameters:
            action (any): The action taken by the agent.

        Returns:
            state (numpy.ndarray): The updated state of the environment.
            reward (float): The reward obtained after taking the action.
            done (bool): A flag indicating if the episode is done.
            False (bool): A flag indicating if the episode has terminated with success.
            {} (dict): Additional information about the step.
        """
        done = False
        self.sampled_clients.append(self.current_client_id)
        self.current_default = self.df.loc[self.month][self.default_col].mean()
        self.class_weight = compute_class_weight('balanced', classes=list(set(self.df.loc[self.month][self.default_col])),
                                                 y=self.df.loc[self.month][self.default_col])

        if not self.accepts_discrete_action:
            action = int(action.round())

        self.action_hist.append(action)
        default = self.df.loc[self.month, self.current_client_id][self.default_col]
        self.action_outcome_history.append((action, default))

        if len(self.sampled_clients) >= self.max_client_id:
            self.action_hist = []
            self.month += 1
            self.action_outcome_history.clear()
            self.sampled_clients = []

        if self.debug and (self.count % 100 == 0):
            print(f"default: {self.current_default:>5.3f} - penalty: {self.penalty:>5.2f} "
                  f"reward: {self.reward:>5.2f} - client: {self.current_client_id:>6} "
                  f"month: {self.month:>3} - done: {done}")

        if self.month >= self.max_month:
            done = True
            return self.state, self.reward, done, False, {}

        self.step_counter += 1
        if self.step_counter >= self.N_SAMPLES:
            self._calculate_penalty()
            self.step_counter = 0

        if len(self.action_outcome_history) == self.reward_delay_steps:
            self.reward =  self._calculate_delayed_reward() + self._scale_reward(self.penalty)
            self.action_outcome_history.clear()
        else:
            self.reward += 0
        
        self.count += 1
        while True:
            next_client = self.rng.choice(self.client_list)
            if next_client not in self.sampled_clients:
                self.current_client_id = next_client
                break

        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        return self.state, self.reward, done, False, {}

    def render(self, mode='human', close=False):
        """
        Render the environment.
        
        Parameters:
            mode (str): The rendering mode. Defaults to 'human'.
            close (bool): Whether to close the environment after rendering. Defaults to False.
        
        Returns:
            None
        """
        pass


################################################################ Adds penalty and reward history to state ################################################################

class RiskManagementEnvDynaState(gym.Env):
    """
    Initializes the Gymnasium environment.

    Parameters:
            df (DataFrame): The input DataFrame containing the features and default column.
            debug (bool): A flag indicating whether to print debug information.
            scaled_features (bool): A flag indicating whether the features should be scaled.
            accepts_discrete_action (bool): A flag indicating whether the agent accepts discrete actions.
            features_col (list): A list of feature column names.
            default_col (str): The name of the default column.
            obs_dim (int): The dimension of the observation space.
            client_dim (int): The dimension of the client space.
            action_dim (int): The dimension of the action space.
            reward_delay_steps (int, optional): The number of steps to delay the reward. Defaults to 1.
            seed (int, optional): The random seed. Defaults to 123.
            model_name (str, optional): The name of the model. Defaults to "".

    Returns:
        None
     """
    def __init__(self, df, debug, scaled_features, accepts_discrete_action,
                 features_col, default_col, obs_dim, client_dim, action_dim,
                 rng, eward_delay_steps=1, seed=123, model_name=""):

        super().__init__(df, debug, scaled_features, accepts_discrete_action,
                 features_col, default_col, obs_dim, client_dim, action_dim,
                 rng, eward_delay_step, seed, model_name)

    def _define_observation_space(self, scaled_features, features_col, df):
        """
        Define the observation space for the environment.

        Parameters:
            scaled_features (bool): Flag indicating whether the features should be scaled.
            features_col (str): Column name for the features in the dataframe.
            df (pandas.DataFrame): The input dataframe containing the features.

        Returns:
            gym.Space: The defined observation space.
        """
        if scaled_features:
            return spaces.Box(low=-1, high=1, shape=(self.obs_dim + 2,), dtype=np.float32)  # +2 for default_rate and penalty
        else:
            low = np.append(df[features_col].min().values, [0, -np.inf])  # Extend the low with -inf for new features
            high = np.append(df[features_col].max().values, [0, np.inf])  # Extend the high with inf for new features
            return spaces.Box(low=low, high=high, shape=(self.obs_dim + 2,), dtype=np.float32)  # +2 for default_rate and penalty

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Parameters:
        - seed (int or None): The random seed to use for initializing the environment. If None, a random seed will be used.
        - options (dict or None): Additional options for resetting the environment. If None, default options will be used.

        Returns:
        - state (numpy.ndarray): The initial state of the environment.
        - info (dict): Additional information about the reset.
        """
        self.reward = 0
        self.step_counter = 0
        self.action_outcome_history.clear()
        self.month = 0
        self.current_client_id = self.rng.choice(self.client_list)
        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        self.state = np.append(self.state, [self.default_rate, self.penalty])
        self.state = self.state.astype(np.float32)
        return self.state, {}


    def step(self, action):
        """
        Updates the environment state based on the given action and returns the 
        updated state, reward, done flag, and additional information.

        Parameters:
            action (any): The action taken by the agent.

        Returns:
            state (numpy.ndarray): The updated state of the environment.
            reward (float): The reward obtained after taking the action.
            done (bool): A flag indicating if the episode is done.
            False (bool): A flag indicating if the episode has terminated with success.
            {} (dict): Additional information about the step.
        """
        done = False
        self.sampled_clients.append(self.current_client_id)
        self.current_default = self.df.loc[self.month][self.default_col].mean()
        self.class_weight = compute_class_weight('balanced', classes=list(set(self.df.loc[self.month][self.default_col])),
                                                 y=self.df.loc[self.month][self.default_col])

        if not self.accepts_discrete_action:
            action = int(action.round())

        self.action_hist.append(action)
        default = self.df.loc[self.month, self.current_client_id][self.default_col]
        self.action_outcome_history.append((action, default))

        if len(self.sampled_clients) >= self.max_client_id:
            self.action_hist = []
            self.month += 1
            self.action_outcome_history.clear()
            self.sampled_clients = []

        if self.debug and (self.count % 100 == 0):
            print(f"default: {self.current_default:>5.3f} - penalty: {self.penalty:>5.2f} "
                  f"reward: {self.reward:>5.2f} - client: {self.current_client_id:>6} "
                  f"month: {self.month:>3} - done: {done}")

        if self.month >= self.max_month:
            done = True
            return self.state, self.reward, done, False, {}

        self.step_counter += 1
        if self.step_counter >= self.N_SAMPLES:
            self._calculate_penalty()
            self.step_counter = 0

        if len(self.action_outcome_history) == self.reward_delay_steps:
            self.reward =  self._calculate_delayed_reward() + self._scale_reward(self.penalty)
            self.action_outcome_history.clear()
        else:
            self.reward += 0
        
        self.count += 1
        while True:
            next_client = self.rng.choice(self.client_list)
            if next_client not in self.sampled_clients:
                self.current_client_id = next_client
                break

        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        self.state = np.append(self.state, [self.default_rate, self.penalty])
        self.state = self.state.astype(np.float32)

        return self.state, self.reward, done, False, {}
        
################################################################ Episode ends every month Env ################################################################


class RiskManagementEnvMonthlyEpisodes(RiskManagementEnv):
    def __init__(self, df, debug, scaled_features, accepts_discrete_action,
                 features_col, default_col, obs_dim, client_dim, action_dim,
                 rng, reward_delay_steps=1, seed=123, model_name=""):
        """
        Initializes an instance of the class RiskManagementEnvMonthlyEpisodes
        it inherits from the class RiskManagementEnv and modifies the _reset() and _step() methods.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            debug (bool): A flag indicating whether to enable debug mode.
            scaled_features (bool): A flag indicating whether to use scaled features.
            accepts_discrete_action (bool): A flag indicating whether the model accepts discrete actions.
            features_col (str): The name of the column containing the features.
            default_col (str): The name of the column containing the default values.
            obs_dim (int): The dimension of the observation space.
            client_dim (int): The dimension of the client space.
            action_dim (int): The dimension of the action space.
            rng (np.random.RandomState): The random number generator object.
            reward_delay_steps (int, optional): The number of steps to delay the reward. Defaults to 1.
            seed (int, optional): The seed for the random number generator. Defaults to 123.
            model_name (str, optional): The name of the model. Defaults to "".

        Returns:
            None
        """

        super().__init__(df, debug, scaled_features, accepts_discrete_action,
                         features_col, default_col, obs_dim, client_dim, action_dim,
                         rng, reward_delay_steps, seed, model_name)

    def reset(self, seed=None, options=None):
        self.reward = 0
        self.step_counter = 0
        self.action_outcome_history.clear()
        self.current_client_id = self.rng.choice(self.client_list)
        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        return self.state, {}

    
    def step(self, action):
        done = False
        self.sampled_clients.append(self.current_client_id)
        self.current_default = self.df.loc[self.month][self.default_col].mean()
        self.class_weight = compute_class_weight('balanced', classes=list(set(self.df.loc[self.month][self.default_col])),
                                                 y=self.df.loc[self.month][self.default_col])

        if not self.accepts_discrete_action:
            action = int(action.round())

        self.action_hist.append(action)
        default = self.df.loc[self.month, self.current_client_id][self.default_col]
        self.action_outcome_history.append((action, default))

        if len(self.sampled_clients) >= self.max_client_id:
            self.action_hist = []
            self.month += 1
            self.action_outcome_history.clear()
            self.sampled_clients = []
            done = True
            print(f"default: {self.current_default:>5.3f} - penalty: {self.penalty:>5.2f} "
                  f"reward: {self.reward:>5.2f} - client: {self.current_client_id:>6} "
                  f"month: {self.month:>3} - done: {done}")
            return self.state, self.reward, done, False, {}

        if self.month >= self.max_month:
            self.month = 0
            done = True


        self.step_counter += 1
        if self.step_counter >= self.N_SAMPLES:
            self._calculate_penalty()
            self.step_counter = 0

        if len(self.action_outcome_history) == self.reward_delay_steps:
            self.reward =  self._calculate_delayed_reward() + self._scale_reward(self.penalty)
            self.action_outcome_history.clear()
        else:
            self.reward += 0
        
        self.count += 1
        while True:
            next_client = self.rng.choice(self.client_list)
            if next_client not in self.sampled_clients:
                self.current_client_id = next_client
                break

        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        return self.state, self.reward, done, False, {}



################################################################ MultiTarget Env ################################################################
class RiskManagementEnvMultiTarget(gym.Env):
    def __init__(self, df, debug, scaled_features, accepts_discrete_action,
                 features_col, default_col, obs_dim, client_dim, action_dim,
                 seed=123, model_name=""):
        super().__init__()
        self.debug = debug
        self.df = df[features_col + [default_col]].copy().astype(np.float32)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.features = features_col
        self.default_col = default_col
        self.client_dim = client_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_delay_steps = 1
        self.action_outcome_history = deque(maxlen=self.reward_delay_steps)
        self.accepts_discrete_action = accepts_discrete_action
        self.state = None
        self.month = 0
        self.max_month = df.index.get_level_values(0).max()
        self.current_client_id = self.rng.integers(self.client_dim)
        self.model_name = model_name
        self.reward = 0
        self.action_hist = []
        self.penalty = 0
        self.count = 0
        self.step_counter = 0
        self.N_SAMPLES = 25
        self.current_defaults = np.zeros(self.action_dim)
        self.action_space = self._define_action_space(accepts_discrete_action)
        self.observation_space = self._define_observation_space(scaled_features, features_col, df)

    def _define_action_space(self, accepts_discrete_action):
        if accepts_discrete_action:
            return spaces.Discrete(self.action_dim)
        else:
            # Adjust this if continuous actions are needed for multiple labels
            return spaces.Discrete(self.action_dim)

    def _define_observation_space(self, scaled_features, features_col, df):
        if scaled_features:
            return spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        else:
            low = df[features_col].min().values
            high = df[features_col].max().values
            return spaces.Box(low=low, high=high, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.reward = 0
        self.step_counter = 0
        self.action_outcome_history.clear()
        self.month = 0
        self.current_client_id = self.rng.integers(self.client_dim)
        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        return self.state, {}

    def _calculate_delayed_reward(self):
        if self.action_outcome_history:
            acts, outs = zip(*self.action_outcome_history)
            unique_classes = np.unique(outs)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=np.array(outs))
            rewards = dict(zip(unique_classes, class_weights))
            return rewards.get(acts[0], -1) # simple reward scheme
        else:
            return -1

    def _scale_reward(self, reward):
        return np.clip(reward, -1, 1)

    def _calculate_penalty(self):
        # Assuming each action corresponds to a class prediction, calculate penalties accordingly
        actual_default_rates = [self.df[self.df[self.default_col] == i].shape[0] / self.client_dim for i in range(self.action_dim)]
        penalties = [-abs(rate - self.current_defaults[i]) / (rate if rate > 0 else 1) for i, rate in enumerate(actual_default_rates)]
        self.penalty = sum(penalties) / len(penalties)
        self.action_hist.clear()

    def _update_current_defaults(self):
        class_counts = np.bincount(self.action_hist, minlength=self.action_dim)
        self.current_defaults = class_counts / class_counts.sum()

    def step(self, action):
        done = False
        self.current_defaults = self.df[self.df[self.default_col] == action].shape[0] / self.client_dim
        self.action_hist.append(action)
        self._update_current_defaults()
        self.class_weight = compute_class_weight('balanced', classes=list(set(self.df.loc[self.month][self.default_col])),
                                                 y=self.df.loc[self.month][self.default_col])

        default = self.df.loc[self.month, self.current_client_id][self.default_col]
        self.action_outcome_history.append((action, default))

        if self.debug & self.count % 100 == 0:
            print(f"reward: {self.reward :>3.5f} - Client: {self.current_client_id} -"
                  f"Action: {action} - Default: {default} - client: {self.month:>3}")

        self.step_counter += 1
        if self.step_counter >= self.N_SAMPLES:
            self._calculate_penalty()
            self.step_counter = 0

        if len(self.action_outcome_history) == self.reward_delay_steps:
            self.reward = self._calculate_delayed_reward()/2 # + self.penalty
            self.action_outcome_history.clear()
        
        self.month = (self.month + 1) % (self.max_month + 1)
        self.current_client_id = self.rng.integers(self.client_dim)
        self.state = self.df.loc[self.month, self.current_client_id][self.features].values

        if self.month == 0:
            done = True

        self.count += 1
        return self.state, self.reward, done, False, {}

    def render(self, mode='human', close=False):
        # Rendering logic here
        pass


################################################################ BOOTSTRAPPED ENV ################################################################

class RiskManagementEnvBootstrapped(gym.Env):
    def __init__(self, 
                 df: pd.DataFrame,
                 debug: bool,
                 scaled_features: bool,
                 accepts_discrete_action: bool,
                 features_col: list, 
                 default_col: str,
                 obs_dim: int, 
                 client_dim: int, 
                 action_dim: int,
                 seed: int = 123,
                 model_name: str = ""):
        super().__init__()

        # Initialize your environment here
        self.debug = debug
        self.df = df[features_col+[default_col]].copy().astype(np.float32)
        self.seed = seed
        self.features = features_col
        self.default_col = default_col
        self.client_dim = client_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_delay_steps = 1
        self.action_outcome_history = deque(maxlen=self.reward_delay_steps)

        # define spaces
        self.accepts_discrete_action = accepts_discrete_action
        
        if self.accepts_discrete_action:
            self.action_space = spaces.Discrete(self.action_dim)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim-1,), dtype="int")
        
        if scaled_features:
            self.observation_space = spaces.Box(low=-1,
                                            high=1, 
                                            shape=(self.obs_dim,), dtype=np.float32)
        else:
            low = self.df[self.features].min().values
            high = self.df[self.features].max().values
            self.observation_space = spaces.Box(low=low,
                                                high=high, 
                                                shape=(self.obs_dim,), dtype=np.float32)

        self.state = None
        self.month = 0
        self.max_month = self.df.index.get_level_values(0).max()
        self.current_client_id = 0
        self.max_client_id = self.df.index.get_level_values(1).max()
        self.default_rate = 0.0
        self.rng = np.random.default_rng(self.seed)
        self.model_name = model_name
        self.reward = 0
        self.action_hist = []#np.zeros(int(self.max_client_id)+1)
        self.penalty = 0
        self.count = 0
        self.step_counter = 0
        self.N_SAMPLES = 100
        self.current_default_history = deque(maxlen=10)  # Adjust length as needed
        self.running_mean_default = 0.0

    def reset(self, seed=None, options=None):
        self.reward = 0
        self.penalty = 0
        self.step_counter = 0 
        self.action_outcome_history.clear()
        self.running_mean_default = 0.0
        self.month = 0
        self.current_client_id = 0
        self.state = self.df.loc[self.month, self.current_client_id][self.features].values
        return self.state, {}

    
    def _calculate_delayed_reward(self):
        acts =[]
        outs = []
        for action, outcome in self.action_outcome_history:
            acts.append(action)
            outs.append(outcome)
        return (self.class_weight[1]+self.current_default) if (acts[0]-outs[-1]) == 0 else (-self.class_weight[0]+self.current_default)
    
    def _scale_reward(self, reward):
        # Define the maximum and minimum possible rewards
        max_reward = self.current_default  # Max of absolute reward values
        min_reward = -self.current_default  # Symmetric around zero
        scaled_reward = (reward - min_reward) / (max_reward - min_reward)

        return scaled_reward

    def _calculate_penalty(self):
        actual_default_rate = self.df.loc[self.month][self.default_col].sum() / self.N_SAMPLES
        self.default_rate = sum(self.action_hist) / self.N_SAMPLES

        if len(self.action_hist) < self.df.loc[self.month].shape[0]:
            self.action_hist.extend([0] * (self.df.loc[self.month].shape[0] - len(self.action_hist)))
        
        self.penalty = - 1 * abs(actual_default_rate - self.default_rate)
        
        actual_defaults = self.df.loc[self.month][self.default_col].values
        self.penalty += f1_score(actual_defaults, self.action_hist)
        self.penalty += recall_score(actual_defaults, self.action_hist)
        self.penalty += accuracy_score(actual_defaults, self.action_hist)
        # self.penalty /= 10
        self.action_hist = []

    def _update_running_mean_default(self):
            if self.current_default_history:
                self.running_mean_default = np.mean(self.current_default_history)

    def _calculate_current_default(self):
        # Implement the logic to calculate current_default based on self.action_hist
        if len(self.action_hist) >= self.N_SAMPLES:
            self.current_default = np.mean(self.action_hist[-self.N_SAMPLES:])


    def step(self, action):
        done = False
        self.current_default = self.df.loc[self.month][self.default_col].mean()
        self.current_default_history.append(self.current_default)
        self._update_running_mean_default()
        self.class_weight = compute_class_weight('balanced', 
                                                 classes=list(set(self.df.loc[self.month][self.default_col])),
                                                 y=self.df.loc[self.month][self.default_col])

        # Convert action if necessary
        if not self.accepts_discrete_action:
            action = action.round()[0]
        
        self.action_hist.append(action)
        if len(self.action_hist) == self.N_SAMPLES:
            self._calculate_current_default()
            self.action_hist.clear()  # Reset the history after updating

        
        self._calculate_current_default()
        default = self.df.loc[self.month, self.current_client_id][self.default_col]
        self.action_outcome_history.append((action, default))

        # Update the step counter and calculate the penalty if needed
        self.step_counter += 1
        if self.step_counter >= self.N_SAMPLES:
            self._calculate_penalty()  
            self.step_counter = 0

        # Check and give the delayed reward
        if len(self.action_outcome_history) == self.reward_delay_steps:
            self.reward =  self._calculate_delayed_reward() + self.penalty/10
            self.action_outcome_history.clear()
        else:
            self.reward += 0  # No reward given until the delay period is over


        # Condition to move to the next month (SHOULD BE RUNNT)
        if (abs(self.current_default - self.running_mean_default) < 1e-6) & (self.current_client_id == self.max_client_id):
            self.month += 1
            self.action_hist.clear()
            self.action_outcome_history.clear()
            self.running_mean_default = 0.0
            self.current_client_id = 0


        if self.debug and (self.count % 1 == 0):
            print(f"default: {self.current_default:>5.6f} mean: {self.running_mean_default:>5.6f} penalty: {self.penalty :>5.2f} reward: {self.reward :>5.2f} - client: {self.current_client_id :>6} - month: {self.month} -")# flush=True, end="\r")

        # Check if the last month is reached
        if self.month >= self.max_month:
            done = True
            return self.state, self.reward, done, False, {}

        # Get the next state
        self.count += 1
        self.current_client_id += 1
        self.state = self.df.loc[self.month, self.current_client_id][self.features].values    
        return self.state, self.reward, done, False, {}

    def render(self, mode='human', close=False):
        pass