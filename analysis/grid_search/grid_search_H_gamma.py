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
    project_path = "dreamer_GRIDSEARCH_3_H_gamma_nu004"
    hyperparams = ['imag_horizon', 'horizon']

    # Load everything
    cfg = load_config_from_wandb_project(path=project_path)
    eval_steps = cfg['run']['eval_rollout_steps']
    df = load_runs_from_wandb_project(
        path=project_path,
        hyperparams=hyperparams,
    )

    print(df)

    # Values of indices

    param1 = set([c[0] for c in df.keys()])
    param2 = set([c[1] for c in df.keys()])

    param1 = sorted(param1)
    param2 = sorted(param2)

    # Extract mean and std error time series
    AUCS = np.zeros((len(param1), len(param2)))
    FINAL = np.zeros((len(param1), len(param2)))
    FINAL_AVG = np.zeros((len(param1), len(param2)))
    window = 5

    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            metric = df[[c for c in df.columns if c[0] == p1 and c[1] == p2 and c[2] == "reward"]]
            if not metric.empty:
                metric /= eval_steps  # Normalise by dividing by number of evaluation episode steps
                metric = np.abs(np.asarray(metric)).flatten()
                auc = scipy.integrate.simpson(metric)
                final_reward = np.mean(metric[-1:])
                final_reward_avg = np.mean(metric[-window:])
                AUCS[i, j] = auc
                FINAL[i, j] = final_reward
                FINAL_AVG[i, j] = final_reward_avg
            else:
                AUCS[i, j] = np.nan
                FINAL[i, j] = np.nan
                FINAL_AVG[i, j] = np.nan


    # ------ PLOTTING -----------
    # Find the indices of the minimum value in AUCS
    min_idx = np.unravel_index(np.argmin(AUCS, axis=None), AUCS.shape)
    min_param1 = param1[min_idx[0]]
    min_param2 = param2[min_idx[1]]
    # Contour plot
    plt.contourf(param1, param2, AUCS.T, levels=20)
    plt.title(f"AUCS")
    plt.colorbar()
    plt.xlabel(hyperparams[0])
    plt.ylabel(hyperparams[1])
    plt.xticks(param1, param1)
    plt.yticks(param2, param2)
    # Add scatter plot for minimum
    plt.scatter(min_param1, min_param2, color='red', marker='o')
    plt.savefig(f"aucs_contour.png", dpi=300)
    plt.close()

    # ------ PLOTTING -----------
    # Find the indices of the minimum value in FINAL
    min_idx = np.unravel_index(np.argmin(FINAL, axis=None), FINAL.shape)
    min_param1 = param1[min_idx[0]]
    min_param2 = param2[min_idx[1]]
    plt.contourf(param1, param2, FINAL.T, levels=20)
    plt.title(f"FINAL")
    plt.colorbar()
    plt.xlabel(hyperparams[0])
    plt.ylabel(hyperparams[1])
    plt.xticks(param1, param1)
    plt.yticks(param2, param2)
    # Add scatter plot for minimum
    plt.scatter(min_param1, min_param2, color='red', marker='o')
    plt.savefig(f"final_reward_contour.png", dpi=300)
    plt.close()


    # ------ PLOTTING -----------
    for i, p1 in enumerate(param1):
        # plt.plot(param2, AUCS[i, :], label=f"{hyperparams[0]}={p1}")
        plt.plot(AUCS[i, :], label=f"{hyperparams[0]}={p1}")
    plt.title(f"AUCS")
    plt.xlabel(hyperparams[1])
    plt.ylabel("AUC")
    # plt.xticks(param2, param2)
    plt.legend()
    plt.savefig(f"aucs_line_plot_1.png", dpi=300)
    plt.close()

    # ------ PLOTTING -----------
    for i, p1 in enumerate(param1):
        plt.plot(param2, FINAL[i, :], label=f"{hyperparams[0]}={p1}")
    plt.title(f"FINAL")
    plt.xlabel(hyperparams[1])
    plt.ylabel("Final reward")
    plt.xticks(param2, param2)
    plt.legend()
    plt.savefig(f"final_reward_line_plot_1.png", dpi=300)
    plt.close()

    # ------ PLOTTING -----------
    for i, p1 in enumerate(param1):
        plt.plot(param2, FINAL_AVG[i, :], label=f"{hyperparams[0]}={p1}")
    plt.title(f"Final reward averaged over {window} last rewards")
    plt.xlabel(hyperparams[1])
    plt.ylabel("Final avg reward")
    plt.xticks(param2, param2)
    plt.legend()
    plt.savefig(f"final_avg_reward_line_plot_1.png", dpi=300)
    plt.close()

    # ------ PLOTTING -----------
    for i, p2 in enumerate(param2):
        plt.plot(param1, AUCS[:, i], label=f"{hyperparams[1]}={p2}")
    plt.title(f"AUC")
    plt.xlabel(hyperparams[0])
    plt.ylabel("AUC")
    plt.xticks(param1, param1)
    plt.legend()
    plt.savefig(f"aucs_line_plot_2.png", dpi=300)
    plt.close()

    # ------ PLOTTING -----------
    for i, p2 in enumerate(param2):
        plt.plot(param1, FINAL[:, i], label=f"{hyperparams[1]}={p2}")
    plt.title(f"FINAL")
    plt.xlabel(hyperparams[0])
    plt.ylabel("Final reward")
    plt.xticks(param1, param1)
    plt.legend()
    plt.savefig(f"final_reward_line_plot_2.png", dpi=300)
    plt.close()

    # ------ PLOTTING -----------
    for i, p2 in enumerate(param2):
        plt.plot(param1, FINAL_AVG[:, i], label=f"{hyperparams[1]}={p2}")
    plt.title(f"Final reward averaged over {window} last rewards")
    plt.xlabel(hyperparams[0])
    plt.ylabel("Final avg reward")
    plt.xticks(param1, param1)
    plt.legend()
    plt.savefig(f"final_avg_reward_line_plot_2.png", dpi=300)
    plt.close()


    # ------ PLOTTING -----------
    plt.contourf(param1, param2, FINAL_AVG.T, levels=20)
    plt.title(f"FINAL reward avg over {window} last rewards")
    plt.colorbar()
    plt.xlabel(hyperparams[0])
    plt.ylabel(hyperparams[1])
    plt.xticks(param1, param1)
    plt.yticks(param2, param2)
    # Add scatter plot for minimum
    plt.scatter(min_param1, min_param2, color='red', marker='o')
    plt.savefig(f"final_avg_reward_contour.png", dpi=300)
    plt.close()


    # ------ PLOTTING -----------
    # Line plots as a function of time
    for p1 in param1:
        for p2 in param2:
            metric = df[[c for c in df.columns if c[0] == p1 and c[1] == p2 and c[2] == "reward"]]
            if not metric.empty:
                metric /= eval_steps  # Normalise by dividing by number of evaluation episode steps
                metric = np.abs(np.asarray(metric)).flatten()
                plt.plot(metric, label=f"{p1}_{p2}")
    plt.legend()
    plt.title(f"Evaluation rewards")
    plt.savefig(f"line_plots.png", dpi=300)

    # SAVE DATAFRAME
    df.to_csv('data.csv', index=True)
    with open('config.pkl', 'wb') as f:
        pickle.dump(cfg, f)

