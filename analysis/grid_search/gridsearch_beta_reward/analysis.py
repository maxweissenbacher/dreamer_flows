import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle


if __name__ == '__main__':
    # Enable LateX use
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    eval_steps = 150

    # This code is used to generate a plot that has all the data
    # It requires the presence of the data_XXX.csv files (see below)
    # GENERATING ONE PLOT WITH ALL THE DATA
    # Load from CSV
    nus = [0.005, 0.01, 0.02, 0.04, 0.08]
    dfs = []
    for nu in nus:
        filename = f"data_{str(nu).replace('.','-')}.csv"
        df = pd.read_csv(filename)
        df = df.drop(columns=['Unnamed: 0'])
        dfs.append(df)

    # Values of indices
    param1 = set([eval(c)[0] for c in dfs[0].keys()])
    param1 = sorted(param1)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    # Create plots
    for df, nu in zip(dfs, nus):
        # Extract mean and std error time series
        AUCS_mean = np.zeros((len(param1),))
        AUCS_std = np.zeros((len(param1),))
        FINAL_mean = np.zeros((len(param1),))
        FINAL_std = np.zeros((len(param1),))
        FINAL_AVG_mean = np.zeros((len(param1),))
        FINAL_AVG_std = np.zeros((len(param1),))
        window = 5
        number_of_runs = 0

        # Set up figure for convergence plot
        fig4, ax4 = plt.subplots()
        cmap = plt.get_cmap('tab10')  # Use the 'Set3' colormap
        colors = cmap(np.linspace(0, 1, len(param1)))

        for i, p1 in enumerate(param1):
            metric = df[[c for c in df.columns if eval(c)[0] == p1 and eval(c)[1] == "reward"]]
            if not metric.empty:
                number_of_runs = max(number_of_runs, metric.shape[-1])
                metric /= eval_steps  # Normalise by dividing by number of evaluation episode steps
                metric = np.abs(np.asarray(metric))
                auc = scipy.integrate.simpson(metric, axis=0)
                final_reward = metric[-1, :]
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

            # Plot reward history
            ax4.plot(np.mean(metric, axis=1), label=r"$\beta_{\text{vector}}=$" + f" {p1}", color=colors[i])
            ax4.fill_between(
                range(len(np.mean(metric, axis=1))),
                np.mean(metric, axis=1) - 1.96 * np.std(metric, axis=1) / np.sqrt(5),
                np.mean(metric, axis=1) + 1.96 * np.std(metric, axis=1) / np.sqrt(5),
                alpha=0.2,
                color=colors[i],
            )

        ax4.legend()
        ax4.set_ylabel("Reward")
        ax4.set_xlabel("Steps")
        ax4.set_xticks([2*k+1 for k in range(10)], [f"{(2*k+2)*10}k" for k in range(10)])
        fig4.savefig(f"reward_history_nu_{str(nu).replace('.','-')}.png", dpi=300, bbox_inches='tight')

        # ------ PLOTTING -----------
        # Find the indices of the minimum value in final reward
        min_idx = np.unravel_index(np.argmin(FINAL_mean, axis=None), FINAL_mean.shape)
        min_param1 = param1[min_idx[0]]
        # Contour plot
        ax2.plot(param1, FINAL_mean, label=r"$\nu=$"+f" {nu}")
        ax2.fill_between(
            param1,
            FINAL_mean - 1.96 * FINAL_std / np.sqrt(number_of_runs),
            FINAL_mean + 1.96 * FINAL_std / np.sqrt(number_of_runs),
            alpha=0.3,
        )
        # Add scatter plot for minimum
        ax2.scatter(min_param1, FINAL_mean[min_idx[0]], color='red', marker='o')

        # ------ PLOTTING -----------
        # Find the indices of the minimum value in final average reward
        min_idx = np.unravel_index(np.argmin(FINAL_AVG_mean, axis=None), FINAL_AVG_mean.shape)
        min_param1 = param1[min_idx[0]]
        # Contour plot
        ax3.plot(param1, FINAL_AVG_mean, label=r"$\nu=$" + f" {nu}")
        ax3.fill_between(
            param1,
            FINAL_AVG_mean - 1.96 * FINAL_AVG_std / np.sqrt(number_of_runs),
            FINAL_AVG_mean + 1.96 * FINAL_AVG_std / np.sqrt(number_of_runs),
            alpha=0.3,
        )
        # Add scatter plot for minimum
        ax3.scatter(min_param1, FINAL_AVG_mean[min_idx[0]], color='red', marker='o')

        # ------ PLOTTING -----------
        # Find the indices of the minimum value in AUCS
        min_idx = np.unravel_index(np.argmin(AUCS_mean, axis=None), AUCS_mean.shape)
        min_param1 = param1[min_idx[0]]
        # Contour plot
        ax1.plot(param1, AUCS_mean, label=r"$\nu=$" + f" {nu}")
        ax1.fill_between(
            param1,
            AUCS_mean - 1.96 * AUCS_std / np.sqrt(number_of_runs),
            AUCS_mean + 1.96 * AUCS_std / np.sqrt(number_of_runs),
            alpha=0.3,
        )
        # Add scatter plot for minimum
        ax1.scatter(min_param1, AUCS_mean[min_idx[0]], color='red', marker='o')

    ax1.set_xlabel(r"$\beta_{\text{vector}}$")
    ax1.set_ylabel("AUC")
    ax1.set_xticks(param1, param1)
    ax1.set_xscale('log')
    ax1.legend()
    fig1.savefig(f"aucs_combined.png", dpi=300)

    ax2.set_xlabel(r"$\beta_{\text{vector}}$")
    ax2.set_ylabel("Final reward")
    ax2.set_xticks(param1, param1)
    ax2.set_xscale('log')
    ax2.legend()
    fig2.savefig(f"final_reward_combined.png", dpi=300)

    ax3.set_xlabel(r"$\beta_{\text{vector}}$")
    ax3.set_ylabel("Final average reward")
    ax3.set_xticks(param1, param1)
    ax3.set_xscale('log')
    ax3.legend()
    fig3.savefig(f"final_avg_reward_combined.png", dpi=300)
