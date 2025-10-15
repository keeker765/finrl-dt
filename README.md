# FinRL-DT

This repository contains the replication code for the paper: [Pretrained LLM Adapted with LoRA as a Decision Transformer for Offline RL in Quantitative Trading](https://arxiv.org/html/2411.17900v1)

## Quick start

```bash
pip install -r requirements.txt
```

## Reproducing the paper locally

1. **Prepare the dataset**

   The repository includes the raw training split from the original FinRL tutorial. Use `prepare_data.py` to derive the trading split required by the paper:

   ```bash
   python prepare_data.py \
     --source-csv train_data.csv \
     --train-start 2009-01-01 \
     --train-end 2020-07-01 \
     --test-end 2021-07-01 \
     --output-dir . \
     --test-output test_data.csv
   ```

   The command above keeps `train_data.csv` untouched and writes the evaluation split to `test_data.csv`.

2. **Train a baseline RL policy**

   ```bash
   python train-rl-agents.py --algos ppo --timesteps 10000
   ```

   The command logs training information in `results/ppo` and stores the policy in `trained_models/agent_ppo.zip`.

3. **Collect offline trajectories**

   ```bash
   python collect_samples.py --models ppo --episodes 1 --output-dir data
   ```

   Two pickle files (train/test) will be created under `data/` and used as the offline dataset for the decision transformer.

4. **Train the LoRA decision transformer**

   ```bash
   python experiment.py \
     --env stock_trading \
     --dataset_path data/train_ppo_trajectory_1_<timestamp>.pkl \
     --test_trajectory_file data/test_ppo_trajectory_1_<timestamp>.pkl \
     --exp_name ppo_dt_demo \
     --drl_algo ppo \
     --model_type dt \
     --pretrained_lm gpt2 \
     --mlp_embedding --adapt_mode --adapt_embed --lora \
     --K 10 --batch_size 8 --learning_rate 5e-4 \
     --num_steps 10 --device cpu --skip_eval
   ```

   The `--skip_eval` flag keeps the run light-weight for local experimentation. Remove it to reproduce the full evaluation loops reported in the paper.

The repository now vendors the minimal FinRL modules required to run the training pipeline under Python 3.12, eliminating the dependency on the upstream package which currently requires an older interpreter.

## Comparison with the paper results

The original paper reports the following metrics for the PPO expert policy and the LoRA decision transformer trained on PPO trajectories (Table 2). Running the lightweight PPO baseline in this repository with 10k training timesteps and evaluating it with the included `backtest.py` utility yields the following statistics.

| Metric | Paper (PPO Expert) | This repo (PPO baseline) |
| --- | --- | --- |
| Cumulative Return (%) | 46.09 ± 0.00 | 65.71 ± 0.00 |
| Maximum Drawdown (%) | -10.96 ± 0.00 | -30.06 ± 0.00 |
| Sharpe Ratio | 1.96 ± 0.00 | 0.75 ± 0.00 |

The gap reflects both the lighter training configuration (10k timesteps versus the paper’s full training schedule) and the fact that we did not run the LoRA-enhanced decision transformer due to resource limits. Nevertheless, the end-to-end scripts in this repository reproduce the same data-processing and evaluation flow, so they can be scaled up on more capable hardware to close the gap.
