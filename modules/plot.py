import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass
try:
    plt.rcParams["text.usetex"] = True
except:
    pass


def default_rcParams():
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 18})


def idx(row, column, total_rows=1):
    if total_rows == 1:
        return column
    return (row, column)


def plot_test_scores(frame_scores, fname):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(frame_scores.shape[0]), frame_scores)
    ax.set_xlabel("Test example")
    ax.set_ylabel("Score")
    fig.suptitle(
        f"Average relative L2 error in test set{'{:10.3f}'.format(np.mean(frame_scores))}"
    )
    fig.savefig(fname=fname)


def plot_l2_error_evolution_unformatted(
    scores_cl, scores_olr, scores_ol, T_in, T_out, deltaT, fname
):
    fig, ax = plt.subplots()
    x = np.arange(T_in, T_in + scores_cl.shape[1]) * deltaT
    anchored = scores_cl
    y_std = np.std(anchored, axis=0)
    y_mean = np.mean(anchored, axis=0)
    ax.plot(x, y_mean, label=r"Closed loop")
    # ax.fill_between(x, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.3)

    anchored = scores_olr
    y_std = np.std(anchored, axis=0)
    y_mean = np.mean(anchored, axis=0)
    ax.plot(x, y_mean, label=r"Open loop with reset")
    # ax.fill_between(x, y_mean - y_std, y_mean + y_std, color='green', alpha=0.3)

    anchored = scores_ol
    y_std = np.std(anchored, axis=0)
    y_mean = np.mean(anchored, axis=0)
    ax.plot(x, y_mean, label=r"Open loop")
    # ax.fill_between(x, y_mean - y_std, y_mean + y_std, color='red', alpha=0.3)

    ax.legend()
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.set_xlabel(r"$t$ [min]")
    fig.tight_layout()
    fig.savefig(fname=fname)


def plot_l2_error_evolution_formatted(
    scores_cl, scores_olr, scores_ol, T_in, T_out, deltaT, fname
):
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update(
        {"ytick.labelsize": 22, "xtick.labelsize": 22, "axes.labelsize": 30},
    )  # Y tick font size
    k = 0
    r = 150
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(T_in + T_out, T_in + T_out + scores_cl.shape[1]) * deltaT
    anchored = scores_cl
    # anchored = np.sort(anchored, axis=0)#[k:-k]
    y_mean = np.mean(anchored, axis=0)
    y_rolling = pd.Series(y_mean).rolling(r).mean().to_numpy()
    # ax.plot(x, y_mean, label=r'Closed loop')
    ax.semilogy(x, y_rolling, label=r"Closed loop")

    anchored = scores_olr
    # anchored = np.sort(anchored, axis=0)#[k:-k]
    y_mean = np.mean(anchored, axis=0)
    y_rolling = pd.Series(y_mean).rolling(r).mean().to_numpy()
    # ax.plot(x, y_mean, label=r'Open loop with reset')
    ax.semilogy(x, y_rolling, label=r"Open loop with reset", linestyle="--")

    anchored = scores_ol
    # anchored = np.sort(anchored, axis=0)#[k:-k]
    y_mean = np.mean(anchored, axis=0)
    y_rolling = pd.Series(y_mean).rolling(r).mean().to_numpy()
    # ax.plot(x, y_mean, label=r'Open loop')
    ax.semilogy(x, y_rolling, label=r"Open loop", linestyle="-.")

    ax.legend(fontsize=22)
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.set_xlabel(r"$t$ [min]")
    ax.grid(axis="y")
    ax.set_yscale("log")
    ax.set_ylim(0.11, 0.3)
    ax.grid(which="minor")
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(5)
    fig.tight_layout()
    fig.savefig(fname=fname)
    default_rcParams()


def plot_accuracy_robustness(df, fname):
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update(
        {"ytick.labelsize": 22, "xtick.labelsize": 30, "axes.labelsize": 30},
    )  # Y tick font size

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="dataset",
        y="errors",
        hue="model",
        data=df,
        hue_order=["Closed loop", "Open loop with reset", "Open loop"],
    )

    plt.xlabel("Dataset")
    plt.ylabel(r"Relative $L_2$ error")
    plt.legend(title="Model", loc="upper right", fontsize=22)
    plt.tight_layout()
    plt.savefig(fname=fname)
    default_rcParams()


def plot_inspection(
    frame_pred,
    frame_true,
    deltaX,
    deltaT,
    T_in,
    T_out,
    scaled_solution,
    fname,
    frame_true_noisy=None,
    xbcs=None,
    frame_ol=None,
    frame_olr=None,
    score_pred=None,
    score_ol=None,
    score_olr=None,
    sensors=[],
    gp_error=True,
):
    total_rows = 2 + (frame_ol is not None) * 1 + (frame_olr is not None) * 1
    total_columns = 3
    if scaled_solution == True:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    fig, axs = plt.subplots(
        nrows=total_rows,
        ncols=total_columns,
        sharex=True,
        sharey=False,
        figsize=(5 * total_columns, 5 * total_rows),
    )

    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (frame_true.shape[1] - 1), frame_true.shape[1])
    _t, _x = np.meshgrid(t, x)

    r = 0
    j = 0
    rho = frame_true
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r"$t$")
    axs[i].set_ylabel(r"$x$")
    axs[i].set_title(r"Exact $\rho(x, t)$")
    axs[i].axvline(x=deltaT * T_in, c="red")
    k = T_in
    while k + T_out < frame_pred.shape[1]:
        k += T_out
        axs[i].axvline(x=k * deltaT, c="red")
    for sensor in sensors:
        axs[i].axhline(y=sensor, c="black")

    j += 1
    if frame_true_noisy is not None:
        rho = frame_true_noisy
        i = idx(r, j, total_rows)
        im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r"$t$")
        axs[i].set_ylabel(r"$x$")
        axs[i].set_title(r"Noisy exact $\rho(x, t)$")
        axs[i].axvline(x=deltaT * T_in, c="red")
        for sensor in sensors:
            axs[i].axhline(y=sensor, c="black")
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()

    j += 1
    if xbcs is not None:
        i = idx(r, j, total_rows)
        # Tout fewer measurements since only correction to IC
        xx = np.linspace(0, deltaX * (xbcs.shape[0] - 1), xbcs.shape[0])
        tt = np.linspace(0, deltaT * (xbcs.shape[1] - 1), xbcs.shape[1])
        _tt, _xx = np.meshgrid(tt, xx)
        if gp_error:
            rho = np.abs(xbcs)
        else:
            rho = xbcs
        if gp_error:
            im = axs[i].pcolor(_tt, _xx, rho, cmap="jet")
        else:
            im = axs[i].pcolor(_tt, _xx, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r"$t$")
        axs[i].set_ylabel(r"$x$")
        axs[i].set_title(
            f"{'Abs of  ' if gp_error else ''}GP {'error ' if gp_error else ''}given measurements"
        )
        for sensor in sensors:
            axs[i].axhline(y=sensor, c="black")
        axs[i].axvline(x=deltaT * T_in, c="red")
        if (frame_pred.shape[1] - T_in) / T_out <= 5:
            tvline = T_in + T_out
            while tvline < frame_pred.shape[1]:
                axs[i].axvline(x=deltaT * tvline, c="red")
                tvline += T_out
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()

    if r == total_rows - 1:
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname=fname)
            plt.close()
        return
    j = 0
    r += 1
    rho = frame_pred
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r"$t$")
    axs[i].set_ylabel(r"$x$")
    axs[i].set_title(r"Main observer, $\hat{\rho}(x,t)$")
    axs[i].axvline(x=deltaT * T_in, c="red")
    if (frame_pred.shape[1] - T_in) / T_out <= 5:
        tvline = T_in + T_out
        while tvline < frame_pred.shape[1]:
            axs[i].axvline(x=deltaT * tvline, c="red")
            tvline += T_out

    j += 1
    i = idx(r, j, total_rows)
    rho = np.abs(frame_true - frame_pred)
    im = axs[i].pcolor(_t, _x, rho, cmap="jet")
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r"$t$")
    axs[i].set_ylabel(r"$x$")
    axs[i].set_title("Absolute error")
    for sensor in sensors:
        axs[i].axhline(y=sensor, c="black")
    j += 1
    if score_pred is not None:
        i = idx(r, j, total_rows)
        axs[i].plot(np.arange(T_in, T_in + len(score_pred)) * deltaT, score_pred)
        axs[i].set_xlabel(r"t")
        axs[i].set_title(r"$L_2$ error")
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()

    if r == total_rows - 1:
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname=fname)
            plt.close()
        return
    j = 0
    r += 1
    if frame_ol is not None:
        i = idx(r, j, total_rows)
        rho = frame_ol
        im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r"$t$")
        axs[i].set_ylabel(r"$x$")
        axs[i].set_title(r"Open-loop observer, $\hat{\rho}(x,t)$")
        j += 1
        i = idx(r, j, total_rows)
        _t, _x = np.meshgrid(t, x)
        rho = np.abs(frame_true - frame_ol)
        im = axs[i].pcolor(_t, _x, rho, cmap="jet")
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r"$t$")
        axs[i].set_ylabel(r"$x$")
        axs[i].set_title("Absolute error base")
        j += 1
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
        j += 1
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()

    if score_ol is not None:
        i = idx(r, j, total_rows)
        axs[i].plot(np.arange(T_in, T_in + len(score_ol)) * deltaT, score_ol)
        axs[i].set_xlabel(r"t")
        axs[i].set_title(r"$L_2$ error")
        j += 1
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
        j += 1
    if r == total_rows - 1:
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname=fname)
            plt.close()
        return
    j = 0
    r += 1
    if frame_olr is not None:
        i = idx(r, j, total_rows)
        rho = frame_olr
        im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r"$t$")
        axs[i].set_ylabel(r"$x$")
        axs[i].set_title(r"Open-loop observer with reset, $\hat{\rho}(x,t)$")
        axs[i].axvline(x=deltaT * T_in, c="red")
        if (frame_pred.shape[1] - T_in) / T_out <= 5:
            tvline = T_in + T_out
            while tvline < frame_pred.shape[1]:
                axs[i].axvline(x=deltaT * tvline, c="red")
                tvline += T_out
        j += 1
        i = idx(r, j, total_rows)
        _t, _x = np.meshgrid(t, x)
        rho = np.abs(frame_true - frame_olr)
        im = axs[i].pcolor(_t, _x, rho, cmap="jet")
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r"$t$")
        axs[i].set_ylabel(r"$x$")
        axs[i].set_title("Absolute error base w/ reset")
        for sensor in sensors:
            axs[i].axhline(y=sensor, c="black")
        j += 1
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
        j += 1
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
    if score_olr is not None:
        i = idx(r, j, total_rows)
        axs[i].plot(np.arange(T_in, T_in + len(score_olr)) * deltaT, score_olr)
        axs[i].set_xlabel(r"t")
        axs[i].set_title(r"$L_2$ error")
        j += 1
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
        j += 1

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname=fname)
        plt.close()
    return


def plot_unrolled(
    frame_true,
    frame_cl,
    frame_ol,
    frame_olr,
    score_cl,
    score_ol,
    score_olr,
    deltaX,
    deltaT,
    T_in,
    scaled_solution,
    fname=None,
    sensors=[],
):
    total_rows = 2
    total_columns = 2
    if scaled_solution == True:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    fig, axs = plt.subplots(
        nrows=total_rows,
        ncols=total_columns,
        sharey=False,
        figsize=(5 * total_columns, 5 * total_rows),
    )

    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (frame_true.shape[1] - 1), frame_true.shape[1])
    _t, _x = np.meshgrid(t, x)

    r = 0
    j = 0
    rho = frame_true
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r"$t$ [min]")
    axs[i].set_ylabel(r"$x$ [km]")
    axs[i].set_title(r"True $\rho(x,t)$")
    # axs[i].axvline(x=deltaT * T_in, c="red")

    for k, sensor in enumerate(sensors):
        if k == 0:
            axs[i].axhline(y=sensor, c="black", label=r"$\mathcal{M}$")
        else:
            axs[i].axhline(y=sensor, c="black")
    axs[i].legend(loc="upper right")

    j += 1
    i = idx(r, j, total_rows)
    rho = frame_ol
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r"$t$ [min]")
    axs[i].set_ylabel(r"$x$ [km]")
    axs[i].set_title(r"Open loop $\hat{\rho}(x,t)$")

    j = 0
    r += 1
    i = idx(r, j, total_rows)
    rho = frame_olr
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r"$t$ [min]")
    axs[i].set_ylabel(r"$x$ [km]")
    axs[i].set_title(r"Open loop with reset $\hat{\rho}(x,t)$")

    j += 1
    i = idx(r, j, total_rows)
    rho = frame_cl
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r"$t$ [min]")
    axs[i].set_ylabel(r"$x$ [km]")
    axs[i].set_title(r"Closed loop $\hat{\rho}(x,t)$")

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname=f"{fname}.png", format="png")

    fig, ax = plt.subplots()
    ax.plot(
        np.arange(T_in, T_in + len(score_cl)) * deltaT,
        score_cl,
        label="Closed loop",
        color="blue",
    )
    ax.plot(
        np.arange(T_in, T_in + len(score_ol)) * deltaT,
        score_ol,
        label="Open loop",
        color="red",
    )
    ax.plot(
        np.arange(T_in, T_in + len(score_olr)) * deltaT,
        score_olr,
        label="Open loop with reset",
        color="green",
    )
    ax.set_xlabel(r"t")
    ax.set_title(r"$L_2$ error")
    ax.legend()
    if fname is not None:
        fig.savefig(fname=f"{fname}_l2_error.png", format="png")
        plt.close()


def plot_pred_io(
    frame_pred,
    frame_true,
    deltaX,
    deltaT,
    T_in,
    T_out,
    scaled_solution,
    figscale=None,
    fname_params=None,
):
    if fname_params is not None:
        dir, exp_id, i = fname_params["dir"], fname_params["id"], fname_params["i"]
    if scaled_solution == True:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None

    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (frame_true.shape[1] - 1), frame_true.shape[1])
    _t, _x = np.meshgrid(t, x)

    fig, ax = plt.subplots()
    rho = frame_pred
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.set_ylabel(r"$x$")
    ax.set_yticks([])
    ax.set_xlabel(r"$t$")
    ax.set_xticks([(T_in - 1) * deltaT, (T_in + T_out - 1) * deltaT])
    ax.set_xticklabels([r"$0$", r"$n_{out}\Delta t$"])
    ax.axvline(x=deltaT * (T_in - 1), c="red")
    ax.fill_between(
        [tt for tt in t if tt <= deltaT * (T_in - 1)],
        min(x),
        max(x),
        alpha=0.4,
        color="grey",
    )

    x_limits = ax.get_xlim()
    x_domain = x_limits[1] - x_limits[0]
    current_width, current_height = fig.get_size_inches()
    current_height *= 0.8
    fig.set_size_inches(
        current_width * (x_limits[1] - x_limits[0]) / x_domain,
        current_height,
        forward=True,
    )

    if fname_params is not None:
        fig.savefig(
            fname=f"{dir}/solop_{exp_id}_experiment_{i}_components_output.svg",
            bbox_inches="tight",
            format="svg",
        )
        plt.close()

    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (T_in - 1), T_in)
    _t, _x = np.meshgrid(t, x)

    fig, ax = plt.subplots()
    rho = frame_true[:, :T_in]
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.set_ylabel(r"$x$")
    ax.set_yticks([])
    ax.set_xlabel(r"$t$")
    ax.set_xticks([deltaT * (T_in - 1)])
    ax.set_xticklabels([r"$0$"], ha="right")

    x_limits = ax.get_xlim()
    fig.set_size_inches(
        current_width * (x_limits[1] - x_limits[0]) / x_domain,
        current_height,
        forward=True,
    )
    if fname_params is not None:
        fig.savefig(
            fname=f"{dir}/solop_{exp_id}_experiment_{i}_components_input.svg",
            bbox_inches="tight",
            format="svg",
        )
        plt.close()
