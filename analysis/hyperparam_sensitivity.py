import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy


def convert_path_replace(input_string):
    return 'run.config' + "['" + input_string.replace('.', "']['") + "']"


def load_runs_from_wandb_project(path, hyperparam_str):
    api = wandb.Api()
    df = pd.DataFrame()
    for run in api.runs(path=path):
        rewards = []
        last_rewards = []
        if not run.state == "finished":
            print(f"Run with ID {run.id} is not finished. Skipping this run.")
            continue
        # print(run.config)
        hyperparam = eval(convert_path_replace(hyperparam_str))
        # hyperparam = eval(run.config['env'])['num_sensors']
        for i, row in run.history(keys=["rollout_eval_episode/mean_reward"]).iterrows():
            rewards.append(row["rollout_eval_episode/mean_reward"])
        for i, row in run.history(keys=["rollout_eval_episode/last_reward"]).iterrows():
            last_rewards.append(row["rollout_eval_episode/last_reward"])
        df[hyperparam, 'reward', run.id] = rewards
        df[hyperparam, 'last_reward', run.id] = last_rewards

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
    hyperparam_str = 'loss_scales.rep'
    project_path = "dreamer_HYPERPARAM_betarep"

    # Load everything
    cfg = load_config_from_wandb_project(path=project_path)
    eval_steps = cfg['run']['eval_steps']
    df_different_params = load_runs_from_wandb_project(
        path=project_path,
        hyperparam_str=hyperparam_str,
    )
    df_default_params = load_runs_from_wandb_project(
        path="dreamer_HYPERPARAM_DEFAULTS",
        hyperparam_str=hyperparam_str,
    )
    df = pd.concat([df_default_params, df_different_params], axis=1)

    print(df)

    # Values of indices
    hyperparams = set([c[0] for c in df.keys()])
    hyperparams = sorted(hyperparams)

    # Extract mean and std error time series
    Z_reward = {}
    Z_last_reward = {}
    for param in hyperparams:
        metric = df[[c for c in df.columns if c[0] == param and c[1] == "reward"]]
        metric /= eval_steps  # Normalise by dividing by number of evaluation episode steps
        mean = np.mean(np.abs(metric), axis=1)
        std_error = np.std(np.abs(metric), axis=1)/np.sqrt(metric.shape[1])
        Z_reward[(param, "mean")] = mean
        Z_reward[(param, "std_error")] = std_error

        metric = df[[c for c in df.columns if c[0] == param and c[1] == "last_reward"]]
        mean = np.mean(np.abs(metric), axis=1)
        std_error = np.std(np.abs(metric), axis=1) / np.sqrt(metric.shape[1])
        Z_last_reward[(param, "mean")] = np.asarray(mean)
        Z_last_reward[(param, "std_error")] = np.asarray(std_error)

    # Compute AUC using the mean curve
    aucs_reward = {}
    aucs_last_reward = {}
    for param in hyperparams:
        y = Z_reward[(param, "mean")]
        auc = scipy.integrate.simpson(y)
        aucs_reward[param] = auc

        y = Z_last_reward[(param, "mean")]
        auc = scipy.integrate.simpson(y)
        aucs_last_reward[param] = auc

    print("AUC computed using mean reward", aucs_reward)
    print("AUC computed using last reward", aucs_last_reward)

    # Find the ratio between MAX and MIN
    importance_reward = max(aucs_reward.values()) / min(aucs_reward.values())
    importance_last_reward = max(aucs_last_reward.values()) / min(aucs_last_reward.values())

    print(f"Hyperparameter with config-path {hyperparam_str} has importance {importance_reward:.5f} (computed with "
          f"mean reward) and {importance_last_reward:.5f} (computed with last reward)")

    # Line plots as a function of time
    for param in hyperparams:
        plt.plot(Z_reward[(param, "mean")], label=f"{param}")
        plt.fill_between(
            range(len(Z_reward[(param, "mean")])),
            Z_reward[(param, "mean")] - 1.96 * Z_reward[(param, "std_error")],
            Z_reward[(param, "mean")] + 1.96 * Z_reward[(param, "std_error")],
            alpha=0.2,
        )
    plt.legend()
    plt.title(hyperparam_str)
    plt.savefig(f"{hyperparam_str}.png", dpi=300)

