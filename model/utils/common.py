import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from sklearn.metrics import (mean_squared_error, r2_score)



def get_csv_files(folder_path):
    """
    Get a list of all CSV files in the given folder.

    Args:
        folder_path (str): The path to the folder where the CSV files are located.

    Returns:
        list: A list of file paths to the CSV files.
    """
    csv_files = []
    for file_path in glob.glob(folder_path + '/**/*.csv', recursive=True):
        csv_files.append(file_path)
    return csv_files

def get_dataset(path):
    """
    Reads a dataset from a CSV file located at the given `path` and performs the following operations:
    
    - Sets the column `date_code` as the index of the DataFrame.
    - Encodes the `client_id` column by mapping each unique client ID to a corresponding integer index.
    - Converts all columns, except `date`, to the `float32` data type.
    
    Parameters:
        path (str): The file path of the CSV dataset.
        
    Returns:
        pandas.DataFrame: The dataset with the above transformations applied.
    """
    df = pd.read_csv(path, index_col="date_code")
    df["client_id"] = df["client_id"].map({client_id: i for i, client_id in enumerate(df["client_id"].unique())})
    df = df.astype({col: "float32" for col in df.columns if col != "date"})

    df.reset_index(drop=False, inplace=True)
    return df


def create_logger(log_path):
    """
    Creates a logger at the specified log path.

    Args:
        log_path (str): The path where the log files will be created.

    Returns:
        logger: The logger object that is created.
    """
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["csv"])
    return logger


def predict_from_env(model, env, n_steps):
    obs = env.reset()
    rew = []
    act = []
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rew.append(reward[0])
        act.append(action[0])
        if step % 100 == 0:
            print(f"action: {action} - reward : {reward} - done: {done}")
    return rew, act


def predict_from_frame(model, features):
    """
    Predict Actions in Regular binary environment
    """
    actions = []
    date_codes = []
    for step in features.iterrows():
        date, _ = step[0]
        obs = step[1]
        date_codes.append(date)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action))
    return actions, date_codes

def predict_from_frame_v2(model, features, default_rate=0.0, penalty=0.0):
    """
    Predict actions in multitarget enviornment
    """
    actions = []
    date_codes = []
    for step in features.iterrows():
        date, _ = step[0]
        obs = step[1]
        obs = np.append(obs, [default_rate, penalty])  # Append default_rate and penalty to each observation
        obs = obs.reshape(1, -1)  # Reshape to have a batch dimension
        date_codes.append(date)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action))
    return actions, date_codes


def running_mean_last_n_samples(data, n_samples):

    running_means = []

    for i in range(n_samples - 1, len(data)):
        last_n_samples = data[i - n_samples + 1 : i + 1]  # Extract the last n_samples samples
        mean = np.mean(last_n_samples)  # Calculate the mean
        running_means.append(mean)

    return np.array(running_means)


def running_mean_per_class(data, n_samples, n_classes):
    # Initialize an array to hold the running means for each class
    running_means = np.zeros((n_classes, len(data) - n_samples + 1))

    # Iterate over each class and calculate the running mean
    for class_index in range(n_classes):
        class_data = np.where(np.array(data) == class_index, 1, 0)
        for i in range(n_samples - 1, len(data)):
            last_n_samples = class_data[i - n_samples + 1: i + 1]  # Extract the last n_samples for this class
            running_means[class_index, i - n_samples + 1] = np.mean(last_n_samples)

    return running_means



def plot_running_means_with_palette(running_means, title='Running Means', show_legend=True):
    """
    Plots running means for each class using a color palette suitable for any number of classes.
    This function now plots on the current active figure, allowing multiple calls to plot on the same graph.
    
    Args:
        running_means (numpy.ndarray): Array of running means where each row corresponds to a class.
        title (str): Title for the plot.
        show_legend (bool): Whether to show the legend or not. Defaults to True.
    """
    # Use a colormap to generate colors for plotting.
    colormap = plt.cm.viridis
    colors = [colormap(i) for i in np.linspace(0, 1, running_means.shape[0])]
    
    for i in range(running_means.shape[0]):
        cumsum = np.cumsum(running_means[i, :])
        cumavg = cumsum / np.arange(1, len(running_means[i, :]) + 1)
        plt.plot(cumavg, color=colors[i], linestyle="dashed", label=f"Class {i}" if show_legend else "_nolegend_")
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Average')
    if show_legend:
        plt.legend()



def VecEnvMonitor(env_id, log_dir, env, data, **kwargs):
    """
    Monitor wrapper for vectorized gym environments.
    Writes to different log files per environment.

    Returns:
        Monitor: The initialized Monitor object.

    """
    def _init():
        environment = env(data, debug=True, scaled_features=True, accepts_discrete_action=True, **kwargs)
        log_file = os.path.join(log_dir, f"{env_id}.monitor.csv")
        return Monitor(environment, filename=log_file)
    return _init




def plot_default_rate_history(date_codes, actual, actions, dates_from_codes, title, save_path=None):
    """
    Generate a plot of the default rate history.

    Parameters:
        date_codes (list): A list of date codes.
        actual (list): A list of actual default rate values.
        actions (list): A list of predicted default rate values.
        dates_from_codes (dict): A dictionary mapping date codes to formatted dates.
        title (str): The title of the plot.

    Returns:
        None
    """
    result = pd.DataFrame({
        "dates": date_codes,
        "actual": actual,
        "pred": actions
    }).groupby("dates").mean()

    # Map index to the keys of date_code_mapping and format dates
    result.index = result.index.map(dates_from_codes)
    result.index = pd.to_datetime(result.index, format="%Y-%m-%d")

    plt.figure(figsize=(10, 5))
    plt.plot(result.actual, color="red", linestyle="-", label="actual")
    plt.plot(result.pred, color="gray", linestyle="--", label="predicted")

    # Set titles and labels
    plt.suptitle(title, fontsize=10)
    plt.title(f"RMSE : {np.sqrt(mean_squared_error(result.actual, result.pred)):1.5f} - R2 : {r2_score(result.actual, result.pred):1.5f}", fontsize=10)
    plt.ylabel("Reward")
    plt.xlabel("date")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_reward_history(reward, title,  window_size=10, save_path=None):
    """
    Plots the reward history for a given data set.

    Args:
        data (pandas.DataFrame): The data set containing the reward history.
        reward (numpy.ndarray): The array of rewards.
        title (str): The title of the plot.

    Returns:
        None
    """
    run_mean_rew = running_mean_last_n_samples(reward, window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(run_mean_rew, linestyle="dashed", color="gray", label="reward")
    plt.plot((np.cumsum(run_mean_rew) / np.arange(1, len(run_mean_rew) + 1)), 
             color="red", linestyle="-", label="running mean")
    plt.title(title, fontsize=10)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def plot_actions_running_mean(actions, title, window_size=10, save_path=None):
    """
    Generate a running mean plot of actions.

    Parameters:
        data (pandas.DataFrame): The data containing the actions.
        actions (pandas.Series): The actions to plot.
        title (str): The title of the plot.

    Returns:
        None

    Raises:
        None
    """
    run_mean_act = running_mean_last_n_samples(actions, window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(run_mean_act, label="Actions Running Mean", color="gray", linestyle="dashed")
    plt.plot(np.cumsum(run_mean_act) / np.arange(1, len(run_mean_act) + 1), 
             color="red", linestyle="-", label="Cumulative Mean")
    plt.title(title, fontsize=10)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def exponential_decay_schedule(initial_lr: float = 1e-1, decay_rate: float = 0.99, decay_steps: int = 1000):
    """
    Exponential decay learning rate schedule.
    :param initial_lr: (float) Initial learning rate.
    :param decay_rate: (float) Rate of decay.
    :param decay_steps: (int) Number of steps for decay.
    :return: (callable)
    """
    def func(progress_remaining: float):
        current_step = (1 - progress_remaining) * decay_steps
        return initial_lr * (decay_rate ** current_step)

    return func