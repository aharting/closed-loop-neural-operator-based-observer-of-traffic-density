import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass
#plt.rcParams["text.usetex"] = True
#plt.rcParams.update({'font.size': 13})

def idx(row, column, total_rows=1):
    if total_rows == 1:
        return column
    return (row, column)

def plot_reconstruction(frame_pred, frame_true, deltaX, deltaT, T_in, T_out, scaled_solution, fname, frame_true_noisy=None, xbcs=None, frame_base=None, frame_base_reset=None, frame_corrected=None, score_pred=None, score_base=None, score_base_reset=None, score_corrected=None, sensors=[], gp_error=True):
    total_rows = 2 + (frame_base is not None) * 1 + \
        (frame_base_reset is not None) * 1 + (frame_corrected is not None) * 1
    total_columns = 3
    if scaled_solution == True:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    fig, axs = plt.subplots(nrows=total_rows, ncols=total_columns, sharex=True,
                            sharey=False, figsize=(5 * total_columns, 5 * total_rows))

    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (frame_true.shape[1] - 1), frame_true.shape[1])
    _t, _x = np.meshgrid(t, x)

    r = 0
    j = 0
    rho = frame_true
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Exact $\rho(x, t)$')
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
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title(r'Noisy exact $\rho(x, t)$')
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
            im = axs[i].pcolor(_tt, _xx, rho, cmap="rainbow",
                               vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title(
            f"{'Abs of  ' if gp_error else ''}GP {'error ' if gp_error else ''}given measurements")
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
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Predicted $\hat{\rho}(x,t)$')
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
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title('Absolute error')
    for sensor in sensors:
        axs[i].axhline(y=sensor, c="black")
    j += 1
    if score_pred is not None:
        i = idx(r, j, total_rows)
        axs[i].plot(np.arange(T_in, T_in + len(score_pred))
                    * deltaT, score_pred)
        axs[i].set_xlabel(r't')
        axs[i].set_title(r'$L_2$ error')
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
    if frame_base is not None:
        i = idx(r, j, total_rows)
        rho = frame_base
        im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title(r'Base predicted $\hat{\rho}(x,t)$')
        j += 1
        i = idx(r, j, total_rows)
        _t, _x = np.meshgrid(t, x)
        rho = np.abs(frame_true - frame_base)
        im = axs[i].pcolor(_t, _x, rho, cmap="jet")
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title('Absolute error base')

        j += 1
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
        j += 1
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
    if score_base is not None:
        i = idx(r, j, total_rows)
        axs[i].plot(np.arange(T_in, T_in + len(score_base))
                    * deltaT, score_base)
        axs[i].set_xlabel(r't')
        axs[i].set_title(r'$L_2$ error')
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
    if frame_base_reset is not None:
        i = idx(r, j, total_rows)
        rho = frame_base_reset
        im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title(r'Base w/ reset predicted $\hat{\rho}(x,t)$')        
        axs[i].axvline(x=deltaT * T_in, c="red")    
        if (frame_pred.shape[1] - T_in) / T_out <= 5:
            tvline = T_in + T_out
            while tvline < frame_pred.shape[1]:
                axs[i].axvline(x=deltaT * tvline, c="red")
                tvline += T_out
        j += 1
        i = idx(r, j, total_rows)
        _t, _x = np.meshgrid(t, x)
        rho = np.abs(frame_true - frame_base_reset)
        im = axs[i].pcolor(_t, _x, rho, cmap="jet")
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title('Absolute error base w/ reset')
        for sensor in sensors:
            axs[i].axhline(y=sensor, c="black")
        j += 1
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
        j += 1
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
    if score_base_reset is not None:
        i = idx(r, j, total_rows)
        axs[i].plot(np.arange(T_in, T_in + len(score_base_reset))
                    * deltaT, score_base_reset)
        axs[i].set_xlabel(r't')
        axs[i].set_title(r'$L_2$ error')
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
    if frame_corrected is not None:
        i = idx(r, j, total_rows)
        rho = frame_corrected
        im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title(r'Corrected prediction of $\hat{\rho}(x,t)$')        
        axs[i].axvline(x=deltaT * T_in, c="red")    
        if (frame_pred.shape[1] - T_in) / T_out <= 5:
            tvline = T_in + T_out
            while tvline < frame_pred.shape[1]:
                axs[i].axvline(x=deltaT * tvline, c="red")
                tvline += T_out
        j += 1
        i = idx(r, j, total_rows)
        _t, _x = np.meshgrid(t, x)
        rho = np.abs(frame_true - frame_corrected)
        im = axs[i].pcolor(_t, _x, rho, cmap="jet")
        fig.colorbar(im, ax=axs[i])
        axs[i].set_xlabel(r'$t$')
        axs[i].set_ylabel(r'$x$')
        axs[i].set_title('Absolute error corrected')
        for sensor in sensors:
            axs[i].axhline(y=sensor, c="black")

        j += 1
    else:
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
        j += 1
        i = idx(r, j, total_rows)
        axs[i].set_axis_off()
    if score_corrected is not None:
        i = idx(r, j, total_rows)
        axs[i].plot(np.arange(T_in, T_in + len(score_corrected))
                    * deltaT, score_corrected)
        axs[i].set_xlabel(r't')
        axs[i].set_title('L2 error')
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

def plot_unravel_1step(frame_true, frame_pred, frame_base, xbcs, deltaX, deltaT, T_in, T_out, scaled_solution, score_pred=None, score_base=None, fname=None, sensors=[], gp_error=False):
    total_rows = 2
    total_columns = 3
    if scaled_solution == True:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    fig, axs = plt.subplots(nrows=total_rows, ncols=total_columns, sharex=True,
                            sharey=False, figsize=(5 * total_columns, 5 * total_rows))

    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (frame_true.shape[1] - 1), frame_true.shape[1])
    _t, _x = np.meshgrid(t, x)

    r = 0
    j = 0
    rho = frame_true
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Exact $\rho(x, t)$')
    axs[i].axvline(x=deltaT * T_in, c="red")
    axs[i].axvline(x=deltaT * (T_in + T_out), c="red")
    for sensor in sensors:
        axs[i].axhline(y=sensor, c="black")

    j += 1
    i = idx(r, j, total_rows)
    # Tout fewer measurements since only correction to IC
    xx = np.linspace(0, deltaX * (xbcs.shape[0] - 1), xbcs.shape[0])
    tt = np.linspace(0, deltaT * (xbcs.shape[1] - 1), xbcs.shape[1])
    _tt, _xx = np.meshgrid(tt, xx)
    rho = np.abs(xbcs)
    if gp_error:
        im = axs[i].pcolor(_tt, _xx, rho, cmap="jet")
    else:
        im = axs[i].pcolor(_tt, _xx, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(
        f"{'Abs of  ' if gp_error else ''}GP {'error ' if gp_error else 'posterior mean '}given measurements")
    axs[i].axvline(x=deltaT * T_in, c="red")
    axs[i].axvline(x=deltaT * (T_in + T_out), c="red")
    for sensor in sensors:
        axs[i].axhline(y=sensor, c="black", xmin=0,
                       xmax=deltaT * (T_in + T_out))
    j += 1
    i = idx(r, j, total_rows)
    axs[i].set_axis_off()

    r += 1
    j = 0
    rho = frame_base
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Open loop $\hat{\rho}(x, t)$')
    axs[i].axvline(x=deltaT * T_in, c="red")
    axs[i].axvline(x=deltaT * (T_in + T_out), c="red")
    for sensor in sensors:
        axs[i].axhline(y=sensor, c="black")

    j += 1
    rho = frame_pred
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Closed loop $\hat{\rho}(x, t)$')
    axs[i].axvline(x=deltaT * T_in, c="red")
    axs[i].axvline(x=deltaT * (T_in + T_out), c="red")
    for sensor in sensors:
        axs[i].axhline(y=sensor, c="black")

    j += 1
    i = idx(r, j, total_rows)
    axs[i].plot(np.arange(T_in, T_in + len(score_pred))
                * deltaT, score_pred, label="Closed loop")
    axs[i].plot(np.arange(T_in, T_in + len(score_base))
                * deltaT, score_base, label="Open loop")
    axs[i].set_xlabel(r't')
    axs[i].set_title(r'$L_2$ error')
    axs[i].legend()
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname=fname)
        plt.close()

def plot_unravel_fullstep(frame_true, frame_pred, frame_base, frame_base_reset, score_pred, score_base, score_base_reset, deltaX, deltaT, T_in, scaled_solution, fname=None, sensors=[]):
    total_rows = 2
    total_columns = 2
    if scaled_solution == True:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    fig, axs = plt.subplots(nrows=total_rows, ncols=total_columns,
                            sharey=False, figsize=(5 * total_columns, 5 * total_rows))

    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (frame_true.shape[1] - 1), frame_true.shape[1])
    _t, _x = np.meshgrid(t, x)

    r = 0
    j = 0
    rho = frame_true
    i = idx(r, j, total_rows)
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    #fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'True $\rho(x,t)$')
    #axs[i].axvline(x=deltaT * T_in, c="red")

    for k, sensor in enumerate(sensors):
        if k==0:
            axs[i].axhline(y=sensor, c="black", label=r"$\mathcal{M}$")
        else:
            axs[i].axhline(y=sensor, c="black")
    axs[i].legend(loc="upper right")

    j += 1
    i = idx(r, j, total_rows)
    rho = frame_base
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Open loop $\hat{\rho}(x,t)$')

    j = 0
    r += 1
    i = idx(r, j, total_rows)
    rho = frame_base_reset
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    #fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Open loop with reset $\hat{\rho}(x,t)$')

    j += 1
    i = idx(r, j, total_rows)
    rho = frame_pred
    im = axs[i].pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$x$')
    axs[i].set_title(r'Closed loop $\hat{\rho}(x,t)$')

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname=fname)

    fig, ax = plt.subplots()
    ax.plot(np.arange(T_in, T_in + len(score_pred)) * deltaT,
            score_pred, label="Closed loop", color="blue")
    ax.plot(np.arange(T_in, T_in + len(score_base)) * deltaT,
            score_base, label="Open loop", color="red")
    ax.plot(np.arange(T_in, T_in + len(score_base_reset)) * deltaT,
            score_base_reset, label="Open loop with reset", color="green")
    ax.set_xlabel(r't')
    ax.set_title(r'$L_2$ error')
    ax.legend()
    if fname is not None:
        fig.savefig(fname=f"{fname}_l2_error")
        plt.close()

def plot_base_io(frame_pred, frame_true, deltaX, deltaT, T_in, T_out, scaled_solution, figscale=None, fname_params=None):
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
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in - 1) * deltaT,  (T_in + T_out - 1) * deltaT])
    ax.set_xticklabels([r"$0$", r"$n_{out}\Delta t$"])
    ax.axvline(x=deltaT * (T_in - 1), c="red")
    ax.fill_between([tt for tt in t if tt <= deltaT * (T_in - 1)], min(x), max(x), alpha=0.4, color="grey")
    if figscale is not None:
        current_width = figscale["current_width"]
        current_height = figscale["current_height"]
        x_domain = figscale["x_domain"]
        x_limits = ax.get_xlim()
        fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
        fig.savefig(fname=f"{dir}/arch0_{exp_id}_experiment_{i}_components_base_output")
        plt.close()
    
    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (T_in - 1), T_in)
    _t, _x = np.meshgrid(t, x)
    
    fig, ax = plt.subplots()    
    rho = frame_true[:, :T_in]
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([deltaT * (T_in - 1)])
    ax.set_xticklabels([r"$0$"], ha='right')
    if figscale is not None:
        current_width = figscale["current_width"]
        current_height = figscale["current_height"]
        x_domain = figscale["x_domain"]
        x_limits = ax.get_xlim()
        fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
        fig.savefig(fname=f"{dir}/arch0_{exp_id}_experiment_{i}_components_base_input", bbox_inches='tight')
        plt.close()

def plot_architechture_io(frame_true, frame_pred, frame_corrected, xbcs, deltaX, deltaT, T_in, T_out, scaled_solution, fname_params=None, sensor_xind=[], sensors=[]):
    if fname_params is not None:
        dir, exp_id, i = fname_params["dir"], fname_params["id"], fname_params["i"]
    if scaled_solution == True:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    
    fig, ax = plt.subplots()
    x = np.linspace(0, deltaX * (frame_true.shape[0] - 1), frame_true.shape[0])
    t = np.linspace(0, deltaT * (frame_true.shape[1] - 1), frame_true.shape[1])
    _t, _x = np.meshgrid(t, x)    
    rho = np.concatenate((frame_pred[:, :T_in + T_out - T_in], frame_corrected[:, T_in + T_out - T_in:T_in + T_out], frame_pred[:, T_in + T_out:]), axis=1)
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in - 1) * deltaT,  (T_in + T_out - 1) * deltaT])
    ax.set_xticklabels([r"$T_{in}$", r"$T_{in} + T_{out}$"])
    #ax.axvline(x=deltaT * (T_in - 1), c="red")
    ax.axvline(x=deltaT * (T_in + T_out - 1), c="red")
    ax.axvline(x=deltaT * (T_in + T_out - T_in - 1), c="red", linestyle="--")
    #ax.fill_between([tt for tt in t if tt <= deltaT * (T_in - 1)], min(x), max(x), alpha=0.4, color="grey")
    ax.fill_between([tt for tt in t if tt >= deltaT * (T_out - 1) and tt <= deltaT * (T_in + T_out - 1)], min(x), max(x), alpha=0.4, color="grey")
    for sensor in sensors:
        ax.axhline(y=sensor, c="black")
    x_limits = ax.get_xlim()    
    x_domain = x_limits[1] - x_limits[0] 
    current_width, current_height = fig.get_size_inches()
    if fname_params is not None:
       fig.savefig(fname=f"{dir}/arch1_{exp_id}_experiment_{i}_components_seq")
       plt.close()

    fig, ax = plt.subplots()
    rho = frame_pred[:, :T_in + T_out]
    x = np.linspace(0, deltaX * (rho.shape[0] - 1), rho.shape[0])
    t = np.linspace(0, deltaT * (T_in + T_out - 1), T_in + T_out)
    _t, _x = np.meshgrid(t, x)
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in - 1) * deltaT,  (T_in + T_out - 1) * deltaT])
    ax.set_xticklabels([r"$T_{in}$", r"$T_{in} + T_{out}$"])
    ax.axvline(x=deltaT * (T_in - 1), c="red")
    ax.fill_between([tt for tt in t if tt <= deltaT * (T_in - 1)], min(x), max(x), alpha=0.4, color="grey")
    for sensor in sensors:
        ax.axhline(y=sensor, c="black")
    x_limits = ax.get_xlim()
    fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
       fig.savefig(fname=f"{dir}/arch1_{exp_id}_experiment_{i}_components_predict")
       plt.close()

    fig, ax = plt.subplots()
    rho = xbcs[:, T_in - 1:T_in + T_out] # Include last T_in for viz    
    x = np.linspace(0, deltaX * (rho.shape[0] - 1), rho.shape[0])
    t = np.linspace(deltaT * (T_in - 1), deltaT * (T_in + T_out - 1), T_out + 1)
    _t, _x = np.meshgrid(t, x)    
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in - 1) * deltaT,  (T_in + T_out - 1) * deltaT])
    ax.set_xticklabels([r"$T_{in}$", r"$T_{in} + T_{out}$"])
    #ax.axvline(x=deltaT * (T_in - 1), c="red")
    for sensor in sensors:
        ax.axhline(y=sensor, c="black")
    x_limits = ax.get_xlim()
    fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
       fig.savefig(fname=f"{dir}/arch1_{exp_id}_experiment_{i}_components_gp_meas")
       plt.close()
    fig, ax = plt.subplots()
    rho = frame_corrected[:, T_in - 1:T_in + T_out]    
    x = np.linspace(0, deltaX * (rho.shape[0] - 1), rho.shape[0])
    t = np.linspace(deltaT * (T_in - 1), deltaT * (T_in + T_out - 1), T_out + 1)
    _t, _x = np.meshgrid(t, x)
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in - 1) * deltaT,  (T_in + T_out - 1) * deltaT])
    ax.set_xticklabels([r"$T_{in}$", r"$T_{in} + T_{out}$"])
    #ax.axvline(x=deltaT * (T_in - 1), c="red")
    ax.axvline(x=deltaT * (T_in + T_out - T_in - 1), c="red", linestyle="--")
    #ax.fill_between([tt for tt in t if tt <= deltaT * T_in], min(x), max(x), alpha=0.4, color="grey")
    #ax.fill_between([tt for tt in t if tt >= deltaT * (T_out) and tt <= deltaT * (T_in + T_out)], min(x), max(x), alpha=0.5, color="grey")    
    for sensor in sensors:
        ax.axhline(y=sensor, c="black")
    x_limits = ax.get_xlim()
    fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
       fig.savefig(fname=f"{dir}/arch1_{exp_id}_experiment_{i}_components_corr")
       plt.close()
    fig, ax = plt.subplots()
    rho = xbcs[:, :T_in]
    x = np.linspace(0, deltaX * (rho.shape[0] - 1), rho.shape[0])
    t = np.linspace(0, deltaT * (T_in - 1), T_in)
    _t, _x = np.meshgrid(t, x)    
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=axs[i])
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in-1) * deltaT])
    ax.set_xticklabels([r"$T_{in}$"], ha='right')
    for sensor in sensors:
        ax.axhline(y=sensor, c="black")
    x_limits = ax.get_xlim()
    fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
       fig.savefig(fname=f"{dir}/arch1_{exp_id}_experiment_{i}_components_gp_ic", bbox_inches='tight')
       plt.close()
    sensor_x = sensors
    sensor_t = np.linspace(0, deltaT * (T_in - 1), T_in)
    sensor_ys = frame_true[sensor_xind, :T_in]
    _t, _x = np.meshgrid(sensor_t, sensor_x)
    fig, ax = plt.subplots()
    ax.scatter(_t, _x, c=sensor_ys, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in - 1) * deltaT])
    ax.set_xticklabels([r"$T_{in}$"], ha='right')
    x_limits = ax.get_xlim()
    ax.set_ylim(sensor_x.min().item(), sensor_x.max().item())
    fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
       fig.savefig(fname=f"{dir}/arch1_{exp_id}_experiment_{i}_components_ic", bbox_inches='tight')
       plt.close()
    sensor_x = sensors
    sensor_t = np.linspace(deltaT * (T_in - 1), deltaT * (T_in + T_out - 1), T_out + 1)
    sensor_ys = frame_true[sensor_xind, T_in - 1:T_in + T_out]
    _t, _x = np.meshgrid(sensor_t, sensor_x)
    fig, ax = plt.subplots()
    ax.scatter(_t, _x, c=sensor_ys, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.set_ylabel(r'$x$')
    ax.set_yticks([])
    ax.set_xlabel(r'$t$')
    ax.set_xticks([(T_in - 1) * deltaT,  (T_in + T_out - 1) * deltaT])
    ax.set_xticklabels([r"$T_{in}$", r"$T_{in} + T_{out}$"])
    ax.set_ylim(sensor_x.min().item(), sensor_x.max().item())
    ax.set_xlim(sensor_t.min().item(), sensor_t.max().item())
    ax.axvline(x=deltaT * (T_in - 1), c="red")
    x_limits = ax.get_xlim()
    fig.set_size_inches(current_width * (x_limits[1] - x_limits[0]) / x_domain, current_height, forward=True)
    if fname_params is not None:
       fig.savefig(fname=f"{dir}/arch1_{exp_id}_experiment_{i}_components_meas")
       plt.close()
    figscale = {"current_width":current_width, 
                "current_height":current_height, 
                "x_domain":x_domain}
    return figscale
        
def plot_seq_out(frame_pred, frame_true, frame_base, frame_base_reset, deltaX, deltaT, T_in, scaled_solution, fname_params=None, sensors=[]):
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
    rho = frame_true
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    #fig.colorbar(im, ax=ax)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    #ax.axvline(x=deltaT * T_in, c="red")
    #ax.set_xticks([])
    #ax.set_yticks([])
    for sensor in sensors:
        ax.axhline(y=sensor, c="black")
    bbox = ax.get_position()
    target_plot_width, target_plot_height = fig.get_size_inches()
    if fname_params is not None:
        fig.savefig(fname=f"{dir}/illu2_{exp_id}_components_frame_true_experiment_{i}")
        plt.close()
    fig, ax = plt.subplots()
    rho = frame_base
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    cb_ax = fig.add_axes([.91,.124,.04,.754])
    fig.colorbar(im,orientation='vertical',cax=cb_ax)
    #ax.set_xticks([])
    #ax.set_yticks([])
    if fname_params is not None:
        fig.savefig(fname=f"{dir}/illu2_{exp_id}_components_frame_base_experiment_{i}")
        plt.close()
    fig, ax = plt.subplots()
    rho = frame_base_reset
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    #fig.colorbar(im, ax=ax)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_title(r'Base with reset predicted $\hat{\rho}(x,t)$')
    if fname_params is not None:
        fig.savefig(fname=f"{dir}/illu2_{exp_id}_components_frame_base_reset_experiment_{i}")
        plt.close()
    fig, ax = plt.subplots()
    rho = frame_pred
    im = ax.pcolor(_t, _x, rho, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    cb_ax = fig.add_axes([.91,.124,.04,.754])
    fig.colorbar(im,orientation='vertical',cax=cb_ax)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_title(r'Predicted $\hat{\rho}(x,t)$')
    if fname_params is not None:
        fig.savefig(fname=f"{dir}/illu2_{exp_id}_components_frame_pred_experiment_{i}")
        plt.close()