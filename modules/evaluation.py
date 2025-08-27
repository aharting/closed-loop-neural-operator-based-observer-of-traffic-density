from modules.unravel import unravel, unravel_dual
from modules.plot import plot_reconstruction, plot_base_io, plot_unravel_fullstep, plot_test_scores, plot_l2_error_evolution_unformatted, plot_l2_error_evolution_formatted,plot_accuracy_robustness
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
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'ytick.labelsize': 14, 'xtick.labelsize': 14})       # Y tick font size

def evaluate(model, loader, config, device, deltaX, deltaT, T_in, T_out, id='test', max_unravel=np.inf):
    """
    Produces a diagnostic view of open-loop autoregression. One view (image) per sample. 
    """
    dir = config['main_dir'] + config['test']['save_dir']
    scaled_solution = config['data']['scaled_solution']
    eval_name=r"$L_2$"
    results = unravel(model=model, loader=loader, device=device, T_in=T_in, T_out=T_out, max_unravel=max_unravel)
    ypred = results["ypred"]
    ytrue = results["ytrue"]
    full_scores_pred = results["full_scores_pred"]
    for i in range(ypred.shape[0]):
        frame_pred = ypred[i]
        frame_true = ytrue[i]
        plot_reconstruction(frame_pred=frame_pred, frame_true=frame_true,
                            deltaX=deltaX, deltaT=deltaT, T_in=T_in, T_out=T_out,
                            scaled_solution=scaled_solution, fname=f"{dir}/{id}_experiment_{i}")
    fig, ax = plt.subplots()
    ax.scatter(np.arange(full_scores_pred.shape[0]), full_scores_pred)
    ax.set_xlabel("Test example")
    ax.set_ylabel("Score")
    fig.suptitle(
        f"Average {eval_name} error in test set{'{:10.3f}'.format(np.mean(full_scores_pred))}")
    fig.savefig(fname=f"{dir}/{id}_scores")
    print(
        f"Average {eval_name} error in test set: {'{:10.3f}'.format(np.mean(full_scores_pred))}")
    return


def evaluate_dual(model, loader, config, device, deltaX, deltaT, T_in, T_out, id='test', max_fcst=np.inf, gp_error=True, std_y=0):
    """
    Produces a diagnostic view of observers. One view (image) per sample. 
    Includes the true and noisy data, the GP regressed measurements, and the three observers (predictions and errors).
    """
    dir = config['main_dir'] + config['test']['save_dir']
    scaled_solution = config['data']['scaled_solution']
    eval_name = r"$L_2$"
    results = unravel_dual(model=model,
                           loader=loader,
                           device=device,
                           T_in=T_in,
                           T_out=T_out,
                           N_sensors=config['data']['N_sensors'],
                           sample=False,
                           max_fcst=max_fcst,
                           gp_error=gp_error,
                           std_y=std_y)

    ytrue = results["ytrue"]
    ytrue_noisy = results["ytrue_noisy"]
    ypred = results["ypred"]
    ybase = results["ybase"]
    ybase_reset = results["ybase_reset"]
    bcs = results["bcs"]
    full_scores_pred = results["full_scores_pred"]
    scores_pred = results["scores_pred"]
    scores_base = results["scores_base"]
    scores_base_reset = results["scores_base_reset"]

    for i in range(ypred.shape[0]):
        frame_pred = ypred[i]
        frame_true = ytrue[i]
        frame_true_noisy = ytrue_noisy[i]
        frame_base = ybase[i]
        frame_base_reset = ybase_reset[i]
        xbcs = bcs[i]
        score_pred = scores_pred[i]
        score_base = scores_base[i]
        score_base_reset = scores_base_reset[i]
        Nx = frame_pred.shape[0]
        sensor_xind = np.array([int(x) for x in np.linspace(0, Nx - 1, config['data']['N_sensors'])])
        sensors = deltaX * sensor_xind
        plot_reconstruction(frame_pred=frame_pred,
                            frame_true=frame_true,
                            frame_true_noisy=frame_true_noisy,
                            frame_base=frame_base,
                            frame_base_reset=frame_base_reset,
                            xbcs=xbcs,
                            deltaX=deltaX,
                            deltaT=deltaT,
                            T_in=T_in,
                            T_out=T_out,
                            scaled_solution=scaled_solution,
                            fname=f"{dir}/unravel_{max_fcst}_{id}_experiment_{i}",
                            score_pred=score_pred,
                            score_base=score_base,
                            score_base_reset=score_base_reset,
                            sensors=sensors,
                            gp_error=gp_error)
    fig, ax = plt.subplots()
    ax.scatter(np.arange(full_scores_pred.shape[0]), full_scores_pred)
    ax.set_xlabel("Test example")
    ax.set_ylabel("Score")
    fig.suptitle(
        f"Average {eval_name} error in test set{'{:10.3f}'.format(np.mean(full_scores_pred))}")
    fig.savefig(fname=f"{dir}/{id}_scores")
    print(
        f"Average {eval_name} error in test set: {'{:10.3f}'.format(np.mean(full_scores_pred))}")
    np.save(f"{dir}/unravel_{max_fcst}_{id}_scores_pred.npy", scores_pred)
    np.save(f"{dir}/unravel_{max_fcst}_{id}_scores_base.npy", scores_base)
    np.save(f"{dir}/unravel_{max_fcst}_{id}_full_scores_pred.npy", full_scores_pred)
    return

def report_accuracy(model, loaders, config, device, deltaX, deltaT, T_in, T_out, id='test', max_fcst=np.inf):
    dir = config['main_dir'] + config['test']['save_dir']
    scores = {}
    loader = loaders["id"]
    std_ys= config['test']['std_ys']
    for std_y in std_ys:
        print(f"ID sigma={std_y} pending...")
        results = unravel_dual(model=model,
                               loader=loader,
                               device=device,
                               T_in=T_in,
                               T_out=T_out,
                               N_sensors=config['data']['N_sensors'],
                               sample=config['data']['sample'],
                               n_samples=config['data']['n_samples'],
                               max_fcst=max_fcst,
                               std_y=std_y)
        full_scores_pred = results["full_scores_pred"]
        full_scores_base_reset = results["full_scores_base_reset"]
        full_scores_base = results["full_scores_base"]
        scores[fr"$\sigma={std_y}$"] = {}
        scores[fr"$\sigma={std_y}$"]["Closed loop"] = full_scores_pred
        scores[fr"$\sigma={std_y}$"]["Open loop with reset"] = full_scores_base_reset
        scores[fr"$\sigma={std_y}$"]["Open loop"] = full_scores_base
    
        if std_y == 0:
            full_scores_pred = results["full_scores_pred"]
            full_scores_base = results["full_scores_base"]
            full_scores_base_reset = results["full_scores_base_reset"]
            scores_pred = results["scores_pred"]
            scores_base = results["scores_base"]
            scores_base_reset = results["scores_base_reset"]

            plot_test_scores(full_scores_pred, fname=f"{dir}/{id}_scores")
            plot_l2_error_evolution_unformatted(scores_pred, scores_base_reset, scores_base, T_in, T_out, deltaT, fname=f"{dir}/{id}_batch_l2_error_distribution")
            plot_l2_error_evolution_formatted(scores_pred, scores_base_reset, scores_base, T_in, T_out, deltaT, fname=f"{dir}/{id}_batch_l2_error_distribution_formatted")
            
            np.save(f"{dir}/unravel_{max_fcst}_{id}_scores_pred.npy", scores_pred)
            np.save(f"{dir}/unravel_{max_fcst}_{id}_scores_base.npy", scores_base)
            np.save(f"{dir}/unravel_{max_fcst}_{id}_scores_base_reset.npy", scores_base_reset)
            np.save(f"{dir}/unravel_{max_fcst}_{id}_full_scores_pred.npy", full_scores_pred)
            np.save(f"{dir}/unravel_{max_fcst}_{id}_full_scores_base.npy", full_scores_base)
            np.save(f"{dir}/unravel_{max_fcst}_{id}_full_scores_base_reset.npy", full_scores_base_reset)

    for key, l in loaders["ood"].items():
        print(f"OOD {key} pending...")
        results = unravel_dual(model=model,
                               loader=l,
                               device=device,
                               T_in=T_in,
                               T_out=T_out,
                               N_sensors=config['data']['N_sensors'],
                               sample=config['data']['sample'],
                               n_samples=config['data']['n_samples'],
                               max_fcst=max_fcst,
                               std_y=0)
        full_scores_pred = results["full_scores_pred"]
        full_scores_base_reset = results["full_scores_base_reset"]
        full_scores_base = results["full_scores_base"]
        scores[f"{key}"] = {}
        scores[f"{key}"]["Closed loop"] = full_scores_pred
        scores[f"{key}"]["Open loop with reset"] = full_scores_base_reset
        scores[f"{key}"]["Open loop"] = full_scores_base

    df = pd.DataFrame.from_dict(scores, orient='columns')
    df.to_csv(f"{dir}/{id}_batch_accuracy_robustness.csv")
    df = df.reset_index().rename(columns={'index': 'model'})
    df = df.melt(id_vars=["model"], var_name="dataset", value_name="errors")
    df = df.explode("errors", ignore_index=True)
    plot_accuracy_robustness(df=df, fname=f"{dir}/{id}_batch_accuracy_robustness")

def report_unrolled(model, loader, config, device, deltaX, deltaT, T_in, T_out, id='test', max_fcst=np.inf):
    dir = config['main_dir'] + config['test']['save_dir']
    results = unravel_dual(model=model,
                            loader=loader,
                            device=device,
                            T_in=T_in,
                            T_out=T_out,
                            N_sensors=config['data']['N_sensors'],
                            sample=config['data']['sample'],
                            n_samples=config['data']['n_samples'],
                            max_fcst=max_fcst,
                            std_y=0)

    scaled_solution = config['data']['scaled_solution']
    ytrue = results["ytrue"]
    ypred = results["ypred"]
    ybase = results["ybase"]
    ybase_reset = results["ybase_reset"]
    scores_pred = results["scores_pred"]
    scores_base = results["scores_base"]
    scores_base_reset = results["scores_base_reset"]

    for i in range(ypred.shape[0]):
        frame_pred = ypred[i]
        frame_true = ytrue[i]
        frame_base = ybase[i]
        frame_base_reset = ybase_reset[i]
        score_pred = scores_pred[i]
        score_base = scores_base[i]
        score_base_reset = scores_base_reset[i]
        Nx = frame_pred.shape[0]
        sensor_xind = np.array([int(x) for x in np.linspace(
            0, Nx - 1, config['data']['N_sensors'])])
        sensors = deltaX * sensor_xind

        plot_unravel_fullstep(frame_true=frame_true,
                    frame_pred=frame_pred,
                    frame_base=frame_base,
                    frame_base_reset=frame_base_reset,
                    score_pred=score_pred,
                    score_base=score_base,
                    score_base_reset=score_base_reset,
                    deltaX=deltaX,
                    deltaT=deltaT,
                    T_in=T_in,
                    scaled_solution=scaled_solution,
                    fname=f"{dir}/observers_{max_fcst}_{id}_experiment_{i}",
                    sensors=sensors)

def report_openloop(model, loader, config, device, deltaX, deltaT, T_in, T_out, id='test'):
    dir = config['main_dir'] + config['test']['save_dir']
    scaled_solution = config['data']['scaled_solution']
    results = unravel(model=model, loader=loader, device=device, T_in=T_in, T_out=T_out, max_unravel=1)
    ypred = results["ypred"]
    ytrue = results["ytrue"]
    for i in range(ypred.shape[0]):
        frame_pred = ypred[i]
        frame_true = ytrue[i]
        plot_base_io(frame_pred=frame_pred, frame_true=frame_true,
                     deltaX=deltaX, deltaT=deltaT, T_in=T_in, T_out=T_out, 
                     scaled_solution=scaled_solution,
                     fname_params={"dir":dir, "id":id, "i":i})