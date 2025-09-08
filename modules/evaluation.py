from modules.unroll import unroll_autoregression, unroll_observers
from modules.plot import (
    plot_inspection,
    plot_pred_io,
    plot_unrolled,
    plot_test_scores,
    plot_l2_error_evolution_unformatted,
    plot_l2_error_evolution_formatted,
    plot_accuracy_robustness,
)
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass


def default_rcParams():
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 25})
    plt.rcParams.update({"ytick.labelsize": 14, "xtick.labelsize": 14})


def inspect_ol(
    model,
    loader,
    config,
    device,
    deltaX,
    deltaT,
    T_in,
    T_out,
    id="test",
    max_unroll=np.inf,
):
    """
    Produces a diagnostic view of open-loop observer (autoregression). One view (image) per sample.
    """
    dir = config["main_dir"] + config["test"]["save_dir"]
    scaled_solution = config["data"]["scaled_solution"]
    eval_name = r"$L_2$"
    results = unroll_autoregression(
        model=model,
        loader=loader,
        device=device,
        T_in=T_in,
        T_out=T_out,
        max_unroll=max_unroll,
    )
    y_pred = results["y_pred"]
    y_true = results["y_true"]
    frame_scores_pred = results["frame_scores_pred"]
    for i in range(y_pred.shape[0]):
        frame_pred = y_pred[i]
        frame_true = y_true[i]
        plot_inspection(
            frame_pred=frame_pred,
            frame_true=frame_true,
            deltaX=deltaX,
            deltaT=deltaT,
            T_in=T_in,
            T_out=T_out,
            scaled_solution=scaled_solution,
            fname=f"{dir}/{id}_experiment_{i}",
        )
    fig, ax = plt.subplots()
    ax.scatter(np.arange(frame_scores_pred.shape[0]), frame_scores_pred)
    ax.set_xlabel("Test example")
    ax.set_ylabel("Score")
    fig.suptitle(
        f"Average {eval_name} error in test set{'{:10.3f}'.format(np.mean(frame_scores_pred))}"
    )
    fig.savefig(fname=f"{dir}/{id}_scores")
    print(
        f"Average {eval_name} error in test set: {'{:10.3f}'.format(np.mean(frame_scores_pred))}"
    )
    return


def inspect_observers(
    model,
    loader,
    config,
    device,
    deltaX,
    deltaT,
    T_in,
    T_out,
    id="test",
    max_fcst=np.inf,
    gp_error=True,
    std_y=0,
):
    """
    Produces a diagnostic view of observers. One view (image) per sample.
    Includes the true and noisy data, the GP regressed measurements, and the three observers (predictions and errors).
    """
    dir = config["main_dir"] + config["test"]["save_dir"]
    scaled_solution = config["data"]["scaled_solution"]
    eval_name = r"$L_2$"
    results = unroll_observers(
        model=model,
        loader=loader,
        device=device,
        T_in=T_in,
        T_out=T_out,
        N_sensors=config["data"]["N_sensors"],
        sample=False,
        max_fcst=max_fcst,
        gp_error=gp_error,
        std_y=std_y,
    )

    y_true = results["y_true"]
    y_true_noisy = results["y_true_noisy"]
    y_cl = results["y_cl"]
    y_ol = results["y_ol"]
    y_olr = results["y_olr"]
    bcs = results["bcs"]
    frame_scores_cl = results["frame_scores_cl"]
    scores_cl = results["scores_cl"]
    scores_ol = results["scores_ol"]
    scores_olr = results["scores_olr"]

    for i in range(y_cl.shape[0]):
        Nx = y_cl[i].shape[0]
        sensor_xind = np.array(
            [int(x) for x in np.linspace(0, Nx - 1, config["data"]["N_sensors"])]
        )
        sensors = deltaX * sensor_xind
        plot_inspection(
            frame_pred=y_cl[i],
            frame_true=y_true[i],
            frame_true_noisy=y_true_noisy[i],
            frame_ol=y_ol[i],
            frame_olr=y_olr[i],
            xbcs=bcs[i],
            deltaX=deltaX,
            deltaT=deltaT,
            T_in=T_in,
            T_out=T_out,
            scaled_solution=scaled_solution,
            fname=f"{dir}/unroll_{max_fcst}_{id}_experiment_{i}",
            score_pred=scores_cl[i],
            score_ol=scores_ol[i],
            score_olr=scores_olr[i],
            sensors=sensors,
            gp_error=gp_error,
        )
    fig, ax = plt.subplots()
    ax.scatter(np.arange(frame_scores_cl.shape[0]), frame_scores_cl)
    ax.set_xlabel("Test example")
    ax.set_ylabel("Score")
    fig.suptitle(
        f"Average {eval_name} error in test set{'{:10.3f}'.format(np.mean(frame_scores_cl))}"
    )
    fig.savefig(fname=f"{dir}/{id}_scores")
    print(
        f"Average {eval_name} error in test set: {'{:10.3f}'.format(np.mean(frame_scores_cl))}"
    )
    np.save(f"{dir}/unroll_{max_fcst}_{id}_scores_pred.npy", scores_cl)
    np.save(f"{dir}/unroll_{max_fcst}_{id}_scores_base.npy", scores_ol)
    np.save(f"{dir}/unroll_{max_fcst}_{id}_full_scores_pred.npy", frame_scores_cl)
    return


def report_accuracy(
    model,
    loaders,
    config,
    device,
    deltaX,
    deltaT,
    T_in,
    T_out,
    id="test",
    max_fcst=np.inf,
):
    """
    Evaluate the observers on the test set and produce the following plots in the report
    - Fig. 8: Robustness of the observers under noise and disturbance.
    - Fig. 9: Evolution of prediction accuracy over the observation horizon.
    """
    dir = config["main_dir"] + config["test"]["save_dir"]
    scores = {}
    loader = loaders["id"]
    std_ys = config["test"]["std_ys"]
    for std_y in std_ys:
        print(f"ID sigma={std_y} pending...")
        results = unroll_observers(
            model=model,
            loader=loader,
            device=device,
            T_in=T_in,
            T_out=T_out,
            N_sensors=config["data"]["N_sensors"],
            sample=config["data"]["sample"],
            n_samples=config["data"]["n_samples"],
            max_fcst=max_fcst,
            std_y=std_y,
        )
        frame_scores_cl = results["frame_scores_cl"]
        frame_scores_olr = results["frame_scores_olr"]
        frame_scores_ol = results["frame_scores_ol"]
        scores[rf"$\sigma={std_y}$"] = {}
        scores[rf"$\sigma={std_y}$"]["Closed loop"] = frame_scores_cl
        scores[rf"$\sigma={std_y}$"]["Open loop with reset"] = frame_scores_olr
        scores[rf"$\sigma={std_y}$"]["Open loop"] = frame_scores_ol

        if std_y == 0:
            scores_cl = results["scores_cl"]
            scores_ol = results["scores_ol"]
            scores_olr = results["scores_olr"]

            plot_test_scores(frame_scores_cl, fname=f"{dir}/{id}_scores")
            plot_l2_error_evolution_unformatted(
                scores_cl,
                scores_olr,
                scores_ol,
                T_in,
                T_out,
                deltaT,
                fname=f"{dir}/{id}_batch_l2_error_distribution",
            )
            plot_l2_error_evolution_formatted(
                scores_cl,
                scores_olr,
                scores_ol,
                T_in,
                T_out,
                deltaT,
                fname=f"{dir}/{id}_batch_l2_error_distribution_formatted",
            )

            np.save(f"{dir}/unroll_{max_fcst}_{id}_scores_cl.npy", scores_cl)
            np.save(f"{dir}/unroll_{max_fcst}_{id}_scores_ol.npy", scores_ol)
            np.save(f"{dir}/unroll_{max_fcst}_{id}_scores_olr.npy", scores_olr)
            np.save(
                f"{dir}/unroll_{max_fcst}_{id}_frame_scores_cl.npy", frame_scores_cl
            )
            np.save(
                f"{dir}/unroll_{max_fcst}_{id}_frame_scores_ol.npy", frame_scores_ol
            )
            np.save(
                f"{dir}/unroll_{max_fcst}_{id}_frame_scores_olr.npy", frame_scores_olr
            )

    for key, l in loaders["ood"].items():
        print(f"OOD {key} pending...")
        results = unroll_observers(
            model=model,
            loader=l,
            device=device,
            T_in=T_in,
            T_out=T_out,
            N_sensors=config["data"]["N_sensors"],
            sample=config["data"]["sample"],
            n_samples=config["data"]["n_samples"],
            max_fcst=max_fcst,
            std_y=0,
        )
        frame_scores_cl = results["frame_scores_cl"]
        frame_scores_olr = results["frame_scores_olr"]
        frame_scores_ol = results["frame_scores_ol"]
        scores[f"{key}"] = {}
        scores[f"{key}"]["Closed loop"] = frame_scores_cl
        scores[f"{key}"]["Open loop with reset"] = frame_scores_olr
        scores[f"{key}"]["Open loop"] = frame_scores_ol

    df = pd.DataFrame.from_dict(scores, orient="columns")
    df.to_csv(f"{dir}/{id}_batch_accuracy_robustness.csv")
    df = df.reset_index().rename(columns={"index": "model"})
    df = df.melt(id_vars=["model"], var_name="dataset", value_name="errors")
    df = df.explode("errors", ignore_index=True)
    plot_accuracy_robustness(df=df, fname=f"{dir}/{id}_batch_accuracy_robustness")


def report_unrolled(
    model,
    loader,
    config,
    device,
    deltaX,
    deltaT,
    T_in,
    T_out,
    id="test",
    max_fcst=np.inf,
):
    """Evaluate the observers on the test set and produce the following plot from the report
    - Fig. 7: Traffic density estimation with observer variants."""
    dir = config["main_dir"] + config["test"]["save_dir"]
    results = unroll_observers(
        model=model,
        loader=loader,
        device=device,
        T_in=T_in,
        T_out=T_out,
        N_sensors=config["data"]["N_sensors"],
        sample=config["data"]["sample"],
        n_samples=config["data"]["n_samples"],
        max_fcst=max_fcst,
        std_y=0,
    )

    scaled_solution = config["data"]["scaled_solution"]
    y_true = results["y_true"]
    y_cl = results["y_cl"]
    y_ol = results["y_ol"]
    y_olr = results["y_olr"]
    scores_cl = results["scores_cl"]
    scores_ol = results["scores_ol"]
    scores_olr = results["scores_olr"]

    for i in range(y_cl.shape[0]):
        frame_pred = y_cl[i]
        Nx = frame_pred.shape[0]
        sensor_xind = np.array(
            [int(x) for x in np.linspace(0, Nx - 1, config["data"]["N_sensors"])]
        )
        sensors = deltaX * sensor_xind

        plot_unrolled(
            frame_true=y_true[i],
            frame_cl=y_cl[i],
            frame_ol=y_ol[i],
            frame_olr=y_olr[i],
            score_cl=scores_cl[i],
            score_ol=scores_ol[i],
            score_olr=scores_olr[i],
            deltaX=deltaX,
            deltaT=deltaT,
            T_in=T_in,
            scaled_solution=scaled_solution,
            fname=f"{dir}/observers_{max_fcst}_{id}_experiment_{i}",
            sensors=sensors,
        )


def report_openloop(
    model, loader, config, device, deltaX, deltaT, T_in, T_out, id="test"
):
    """Evaluate the observers on the test set and produce the following plot from the report
    - Fig. 2: Open-loop prediction with solution operator"""
    dir = config["main_dir"] + config["test"]["save_dir"]
    scaled_solution = config["data"]["scaled_solution"]
    results = unroll_autoregression(
        model=model, loader=loader, device=device, T_in=T_in, T_out=T_out, max_unroll=1
    )
    y_pred = results["y_pred"]
    y_true = results["y_true"]
    for i in range(y_pred.shape[0]):
        frame_pred = y_pred[i]
        frame_true = y_true[i]
        plot_pred_io(
            frame_pred=frame_pred,
            frame_true=frame_true,
            deltaX=deltaX,
            deltaT=deltaT,
            T_in=T_in,
            T_out=T_out,
            scaled_solution=scaled_solution,
            fname_params={"dir": dir, "id": id, "i": i},
        )
