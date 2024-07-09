import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle


def convert_path_replace(input_string):
    return 'run.config' + "['" + input_string.replace('.', "']['") + "']"


def load_runs_from_wandb_project(path, hyperparams):
    api = wandb.Api()
    all_data = []
    columns = []
    df = pd.DataFrame()
    for run in api.runs(path=path):
        rewards = []
        last_rewards = []
        if not run.state == "finished":
            print(f"Run with ID {run.id} is not finished. Skipping this run.")
            continue
        # print(run.config)
        run_parameters = []
        for param in hyperparams:
            run_parameters.append(eval(convert_path_replace(param)))
        for i, row in run.history(keys=["rollout_eval_episode/mean_rewards"]).iterrows():
            rewards.append(row["rollout_eval_episode/mean_rewards"])
        for i, row in run.history(keys=["rollout_eval_episode/mean_last_rewards"]).iterrows():
            last_rewards.append(row["rollout_eval_episode/mean_last_rewards"])
        all_data.append(rewards)
        columns.append((*run_parameters, 'reward', run.id))
        all_data.append(last_rewards)
        columns.append((*run_parameters, 'last_reward', run.id))
        # df[(*run_parameters, 'reward', run.id)] = rewards
        # df[(*run_parameters, 'last_reward', run.id)] = last_rewards

    df = pd.DataFrame(all_data).T
    df.columns = columns

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
    project_path = "dreamer_GRIDSEARCH_H_nu00005"
    hyperparams = ['imag_horizon']

    # Load everything
    cfg = load_config_from_wandb_project(path=project_path)
    eval_steps = cfg['run']['eval_rollout_steps']
    nu = cfg["KS"]["nu"]
    df = load_runs_from_wandb_project(
        path=project_path,
        hyperparams=hyperparams,
    )

    print(df)

    # Values of indices
    param1 = set([c[0] for c in df.keys()])
    param1 = sorted(param1)

    # Extract mean and std error time series
    AUCS_mean = np.zeros((len(param1),))
    AUCS_std = np.zeros((len(param1),))
    FINAL_mean = np.zeros((len(param1),))
    FINAL_std = np.zeros((len(param1),))
    FINAL_AVG_mean = np.zeros((len(param1),))
    FINAL_AVG_std = np.zeros((len(param1),))
    window = 5
    number_of_runs = 0

    for i, p1 in enumerate(param1):
        metric = df[[c for c in df.columns if c[0] == p1 and c[1] == "reward"]]
        if not metric.empty:
            number_of_runs = max(number_of_runs, metric.shape[-1])
            metric /= eval_steps  # Normalise by dividing by number of evaluation episode steps
            metric = np.abs(np.asarray(metric))
            auc = scipy.integrate.simpson(metric, axis=0)
            final_reward = np.mean(metric[-1, :])
            final_reward_avg = np.mean(metric[-window:, :], axis=0)
            AUCS_mean[i] = np.nanmean(auc)
            AUCS_std[i] = np.nanstd(auc)
            FINAL_mean[i] = np.nanmean(final_reward)
            FINAL_std[i] = np.nanstd(final_reward)
            FINAL_AVG_mean[i] = np.nanmean(final_reward_avg)
            FINAL_AVG_std[i] = np.nanstd(final_reward_avg)
        else:
            AUCS_mean[i] = np.nan
            AUCS_std[i] = np.nan
            FINAL_mean[i] = np.nan
            FINAL_std[i] = np.nan
            FINAL_AVG_mean[i] = np.nan
            FINAL_AVG_std[i] = np.nan


    # ------ PLOTTING -----------
    # Find the indices of the minimum value in AUCS
    min_idx = np.unravel_index(np.argmin(AUCS_mean, axis=None), AUCS_mean.shape)
    min_param1 = param1[min_idx[0]]
    # Contour plot
    plt.plot(param1, AUCS_mean)
    plt.fill_between(
        param1,
        AUCS_mean - 1.96 * AUCS_std / np.sqrt(number_of_runs),
        AUCS_mean + 1.96 * AUCS_std / np.sqrt(number_of_runs),
        alpha=0.3,
    )
    plt.title(f"AUCs for nu={nu}")
    plt.xlabel(hyperparams[0])
    plt.xticks(param1, param1)
    # Add scatter plot for minimum
    plt.scatter(min_param1, AUCS_mean[min_idx[0]], color='red', marker='o')
    plt.savefig(f"aucs_H_nu{str(nu).replace('.','-')}.png", dpi=300)
    plt.close()

    """
    # SAVE DATAFRAME
    df.to_csv('data.csv', index=True)
    with open('config.pkl', 'wb') as f:
        pickle.dump(cfg, f)
    """

    # Print summary
    print(f"The maximum number of repetitions for each hyperparameter is {number_of_runs}.")
    print(f"The total number of runs downloaded is {len(df.columns)}.")
    print(f"The complete list of hyperparameters included is {param1}, with names {hyperparams}")
