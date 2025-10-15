"""Train baseline RL agents used to create offline datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from finrl.config import INDICATORS, RESULTS_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure


DEFAULT_TIMESTEPS = {
    "a2c": 50_000,
    "ddpg": 50_000,
    "td3": 50_000,
    "sac": 70_000,
    "ppo": 200_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="train_data.csv")
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["ppo"],
        choices=list(DEFAULT_TIMESTEPS.keys()),
        help="Which algorithms to train.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override the default number of training timesteps.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Where to store training logs.",
    )
    parser.add_argument(
        "--models-dir",
        default=TRAINED_MODEL_DIR,
        help="Where to save trained models.",
    )
    return parser.parse_args()


def build_env(train_csv: Path) -> tuple[StockTradingEnv, int, int]:
    train = pd.read_csv(train_csv)
    train = train.set_index(train.columns[0])
    train.index.names = [""]

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    env = StockTradingEnv(df=train, **env_kwargs)
    sb_env, _ = env.get_sb_env()
    return env, sb_env, stock_dimension


def train_algo(agent: DRLAgent, name: str, timesteps: int, results_dir: Path, models_dir: Path) -> None:
    print(f"Training {name.upper()} for {timesteps} timesteps")
    model = agent.get_model(name)
    tmp_path = Path(results_dir) / name
    tmp_path.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(tmp_path), ["stdout", "csv"]))
    trained = agent.train_model(model=model, tb_log_name=name, total_timesteps=timesteps)
    save_path = Path(models_dir) / f"agent_{name}"
    trained.save(str(save_path))
    print(f"Saved {name.upper()} model to {save_path}")


def main() -> None:
    args = parse_args()
    check_and_make_directories([args.models_dir, args.results_dir])

    env, env_train, _ = build_env(Path(args.train_data))
    agent = DRLAgent(env=env_train)

    requested_algos = set(args.algos)
    timesteps = {
        algo: args.timesteps if args.timesteps is not None else DEFAULT_TIMESTEPS[algo]
        for algo in requested_algos
    }

    for algo in requested_algos:
        train_algo(agent, algo, timesteps[algo], Path(args.results_dir), Path(args.models_dir))


if __name__ == "__main__":
    main()
