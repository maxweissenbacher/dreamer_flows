import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy


def load_runs_from_wandb_project(path):
    api = wandb.Api()
    df = pd.DataFrame()
    for run in api.runs(path=path):
        rewards = []
        last_rewards = []
        if not run.state == "finished":
            print(f"Run with ID {run.id} is not finished. Skipping this run.")
            continue
        # print(run.config)
        for i, row in run.history(keys=["rollout_eval_episode/mean_rewards"]).iterrows():
            rewards.append(row["rollout_eval_episode/mean_rewards"])
        for i, row in run.history(keys=["rollout_eval_episode/mean_last_rewards"]).iterrows():
            last_rewards.append(row["rollout_eval_episode/mean_last_rewards"])
        df['reward', run.id] = rewards
        df['last_reward', run.id] = last_rewards

    return df


def load_config_from_wandb_project(path):
    api = wandb.Api()
    for run in api.runs(path=path):
        if not run.state == "finished":
            print(f"Run with ID {run.id} is not finished. Skipping this run.")
            continue
        return run.config
    return None


if __name__ == '__main__':
    # Specify the WANDB project and the config key to the relevant hyperparameter here
    project_path = "dreamer_AUC_consistency_test"

    # Load everything
    cfg = load_config_from_wandb_project(path=project_path)
    eval_steps = cfg['run']['eval_rollout_steps']
    df = load_runs_from_wandb_project(path=project_path)

    print(df)

    # Extract mean and std error time series
    Z = {}

    metric = df[[c for c in df.columns if c[0] == "reward"]]
    metric /= eval_steps  # Normalise by dividing by number of evaluation episode steps
    Z["reward"] = np.asarray(metric)

    metric = df[[c for c in df.columns if c[0] == "last_reward"]]
    Z["last_reward"] = np.asarray(metric)

    # Compute AUC using the mean curve
    aucs_reward = []
    aucs_last_reward = []
    for i in range(Z["reward"].shape[1]):
        y = np.abs(Z["reward"][:, i])
        auc = scipy.integrate.simpson(y)
        aucs_reward.append(auc)

    for i in range(Z["last_reward"].shape[1]):
        y = np.abs(Z["last_reward"][:, i])
        auc = scipy.integrate.simpson(y)
        aucs_last_reward.append(auc)

    aucs_reward = np.asarray(aucs_reward)
    aucs_last_reward = np.asarray(aucs_last_reward)

    normalised_std_reward = np.std(aucs_reward) / np.mean(aucs_reward)
    normalised_std_last_reward = np.std(aucs_last_reward) / np.mean(aucs_last_reward)
    max_dev_reward = np.max((aucs_reward - np.mean(aucs_reward)) / np.mean(aucs_reward))
    max_dev_last_reward = np.max((aucs_last_reward - np.mean(aucs_last_reward)) / np.mean(aucs_last_reward))

    print("Normalised std (std/mean) for AUC computed from rewards", normalised_std_reward)
    print("Normalised std (std/mean) for AUC computed from last_rewards", normalised_std_last_reward)

    print("Maximum deviation from mean (max[(x-mean)/mean]) for AUC computed from reward", max_dev_reward)
    print("Maximum deviation from mean (max[(x-mean)/mean]) for AUC computed from last reward", max_dev_last_reward)

