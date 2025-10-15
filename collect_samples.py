import os
# Set environment variables to limit threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, TD3, PPO, SAC
from tqdm import tqdm

from finrl.config import INDICATORS, TRAINED_MODEL_DIR

random_seed = 21102

def collect_single_trajectory(env_kwargs, model, train_data):
    # Initialize the environment
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    env = StockTradingEnv(
        df=train_data,
        turbulence_threshold=70,
        risk_indicator_col='vix',
        **env_kwargs
    )

    observations = []
    actions = []
    rewards = []
    dones = []

    state, _ = env.reset()
    done = False

    while not done:
        # Use the trained A2C model to select actions
        model.set_random_seed(random_seed)
        action, _ = model.predict(state, deterministic=True)
        
        next_state, reward, done, truncated, info = env.step(action)

        observations.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        state = next_state

    # Compute next_observations by shifting observations by one step
    observations_np = np.array(observations)
    next_observations_np = np.roll(observations_np, -1, axis=0)
    next_observations_np[-1] = observations_np[-1]  # Last next_observation is same as last state
    
    trajectory = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminals': np.array(dones),
        'next_observations': next_observations_np
    }

    return trajectory

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect trajectories from trained RL agents")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["a2c", "ddpg", "td3", "ppo", "sac"],
        help="Algorithms to collect trajectories from.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Datasets to evaluate on.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Trajectories per model")
    parser.add_argument("--output-dir", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_choices = args.models
    dataset_choices = args.datasets
    episodes_per_model = args.episodes

    # Ensure the 'data' directory exists
    data_dir = args.output_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for train_or_test in dataset_choices:
        # Load the dataset
        data_file = f'{train_or_test}_data.csv'
        data = pd.read_csv(data_file)
        data = data.set_index(data.columns[0])
        data.index.names = ['']

        # Define environment parameters
        stock_dimension = len(data.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }

        ensemble_trajectories = []

        for model_choice in model_choices:
            model_path = os.path.join(TRAINED_MODEL_DIR, f"agent_{model_choice}")
            if not os.path.exists(f"{model_path}.zip") and not os.path.exists(model_path):
                print(f"Skipping {model_choice.upper()} - no trained model found at {model_path}")
                continue
            if model_choice == 'a2c':
                model = A2C.load(model_path)
            elif model_choice == 'ddpg':
                model = DDPG.load(model_path)
            elif model_choice == 'td3':
                model = TD3.load(model_path)
            elif model_choice == 'ppo':
                model = PPO.load(model_path)
            elif model_choice == 'sac':
                model = SAC.load(model_path)
            else:
                raise ValueError(f"Unknown model choice: {model_choice}")

            for episode in range(episodes_per_model):
                trajectory = collect_single_trajectory(env_kwargs, model, data)

                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = (
                    f'{train_or_test}_{model_choice}_trajectory_{episode + 1}_{current_time}.pkl'
                )
                output_path = os.path.join(data_dir, output_filename)
                with open(output_path, 'wb') as f:
                    pickle.dump([trajectory], f)

                print(
                    f"Saved trajectory {episode + 1}/{episodes_per_model} for {model_choice.upper()} "
                    f"on {train_or_test} data to '{output_path}'"
                )

                ensemble_trajectories.append(trajectory)

        # Save the ensemble trajectories to a pickle file
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ensemble_filename = (
            f'{train_or_test}_ensemble_trajectories_{len(ensemble_trajectories)}_{current_time}.pkl'
        )
        ensemble_path = os.path.join(data_dir, ensemble_filename)
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_trajectories, f)

        print(
            f"Saved ensemble of {len(ensemble_trajectories)} trajectories on {train_or_test} data "
            f"to '{ensemble_path}'"
        )
