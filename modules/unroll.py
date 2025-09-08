import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import torch
import numpy as np
import copy
from imported.losses import LpLoss
from modules.data import interpolate
from tqdm import tqdm


def build_true_frame(data, max_unroll):
    """Build the true density frame from data"""
    all_x, all_y = data
    n_unroll = 1
    # Unroll true frame
    for j in range(all_x.shape[0]):
        data_x, data_y = all_x[j], all_y[j]
        if j == 0:
            frame = torch.concat((data_x[..., :-1], data_y), axis=1)
        else:
            sub_ivl = torch.concat((data_x[..., :-1], data_y), axis=1)
            frame = torch.concat((frame, sub_ivl), axis=1)
        if n_unroll >= max_unroll:
            break
        else:
            n_unroll += 1
    return frame


def gen_initial_data(input, sensor_xind, sample, n_samples=0):
    """Generates the first input to the observer (always the data-based estimate
    Output:
        - x_grid: shape [Nx, 1]
        - xic: shape [1, n_samples, Nx, N_rho + 1]
        - frame: shape [Nx, N_rho]
    """
    x_grid = input[:, [-1]]
    sensor_x = x_grid[sensor_xind].unsqueeze(0)
    sensor_y = input[sensor_xind, :-1].unsqueeze(0)

    xic, _ = interpolate(
        x_grid.cpu(), sensor_x.cpu(), sensor_y.cpu(), sample=sample, n_samples=n_samples
    )
    xic = torch.tensor(xic, device=input.device.type, dtype=torch.float)
    frame = xic[..., :-1].squeeze(0)
    frame = torch.mean(frame, dim=0)  # display_xic[0] to get one sample
    return x_grid, xic, frame


def unroll_ol_observer(
    model,
    initial_data,
    frame_true,
    T_in,
    T_out,
    sensor_xind,
    sample,
    n_samples,
    myeval,
    T_fcst=None,
    max_fcst=np.inf,
):
    """
    Unrolls the open-loop observer.
    The observer is initialized with GP regression, after which autoregression is unrolled in increments of T_fcst.
    """
    if T_fcst == None:
        T_fcst = T_out
    # First initial condition
    x_grid, xic, frame = initial_data
    frame_ol = frame

    # The first T_in:T_in + T_out - 1 states are populated with the data-based estimate
    # (so that starting the autoregression at t=0 works, since first prediction predicts rho at t=T_in+T_out)
    y = frame_true[..., T_in : T_in + T_out - 1].unsqueeze(0)
    sensor_x = x_grid[torch.newaxis, sensor_xind, :]
    sensor_y = y[:, sensor_xind, :]
    xbc, _ = interpolate(
        x_grid.cpu(), sensor_x.cpu(), sensor_y.cpu(), sample=sample, n_samples=n_samples
    )
    xbc = torch.tensor(xbc, device=model.device, dtype=torch.float)
    pred_y = torch.mean(xbc[..., :-1], dim=1)

    score = myeval(
        torch.swapaxes(pred_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1)
    )  # compute error per time - time along batch dimension, squeeze original batch dimension
    frame_ol = torch.concatenate((frame_ol, pred_y.squeeze(0)), axis=1)

    start_score = copy.deepcopy(score)[0]
    _score = -1
    while frame_ol.shape[1] + T_fcst <= min(
        frame_true.shape[1], max_fcst + T_in + T_out
    ):
        # If process is unstable, stop the computation
        if _score > max(6 * start_score, 0.75):  # np.inf: # max(6 * start_score, 0.75):
            score = torch.concatenate((score, _score), axis=0)
            mean = torch.mean(pred_y)
            if mean > 0.5:
                proxy_value = torch.ones_like(pred_y.squeeze(0))
            else:
                proxy_value = torch.zeros_like(pred_y.squeeze(0))
            frame_ol = torch.concatenate((frame_ol, proxy_value), axis=1)
            continue
        # Recent T_in predictions serve as model input
        xic = frame_ol[..., -(T_out + T_in - T_fcst) : -(T_out - T_fcst)].unsqueeze(0)
        xic = torch.concatenate((xic, x_grid.unsqueeze(0)), axis=-1)
        pred_y = model.ic_model(xic)
        # Get the last prediction
        pred_y = pred_y[..., -T_fcst:]
        # Get the corresponding true value and compute the score
        y = frame_true[..., frame_ol.shape[1] : frame_ol.shape[1] + T_fcst].unsqueeze(0)
        _score = myeval(
            torch.swapaxes(pred_y, 0, -1).squeeze(-1),
            torch.swapaxes(y, 0, -1).squeeze(-1),
        )
        score = torch.concatenate((score, _score), axis=0)
        # Append the prediction to the frame
        frame_ol = torch.concatenate((frame_ol, pred_y.squeeze(0)), axis=1)
    return frame_ol, score


def unroll_olr_observer(
    model,
    initial_data,
    frame_true,
    T_in,
    T_out,
    sensor_xind,
    sample,
    n_samples,
    myeval,
    T_fcst=None,
    max_fcst=np.inf,
):
    """
    Unrolls the open-loop observer with reset.
    The observer is initialized with GP regression, after which the following applies:
    - At every time step, get the data-based state estimate and pass as input to the prediction operator.
    - Repeat while shifting the prediction window in increments of T_fcst
    """
    if T_fcst == None:
        T_fcst = T_out
    # First initial condition
    x_grid, xic, frame = initial_data
    frame_olr = frame

    # The first T_in:T_in + T_out - 1 states are populated with the data-based estimate for comparability with the other observers
    y = frame_true[..., T_in : T_in + T_out - 1].unsqueeze(0)
    sensor_x = x_grid[torch.newaxis, sensor_xind, :]
    sensor_y = y[:, sensor_xind, :]
    xbc, _ = interpolate(
        x_grid.cpu(), sensor_x.cpu(), sensor_y.cpu(), sample=sample, n_samples=n_samples
    )
    xbc = torch.tensor(xbc, device=model.device, dtype=torch.float)
    pred_y = torch.mean(xbc[..., :-1], dim=1)

    score = myeval(
        torch.swapaxes(pred_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1)
    )  # compute error per time - time along batch dimension, squeeze original batch dimension
    frame_olr = torch.concatenate((frame_olr, pred_y.squeeze(0)), axis=1)

    start_score = copy.deepcopy(score)[0]
    _score = -1
    while frame_olr.shape[1] + T_fcst <= min(
        frame_true.shape[1], max_fcst + T_in + T_out
    ):
        # Get data-based estimate of input state through GP regression
        y = frame_true[
            ...,
            -(T_out + T_in - T_fcst)
            + frame_olr.shape[1] : -(T_out - T_fcst)
            + frame_olr.shape[1],
        ].unsqueeze(0)
        sensor_y = y[:, sensor_xind, :]
        xic, _ = interpolate(
            x_grid.cpu(),
            sensor_x.cpu(),
            sensor_y.cpu(),
            sample=False,
            n_samples=n_samples,
        )
        xic = torch.tensor(xic, device=model.device, dtype=torch.float)
        # n_samples input samples are passed to the model, output is averaged.
        unflat_shape = xic.shape[:2]
        xic_flat = xic.flatten(start_dim=0, end_dim=1)
        pred_y = model.ic_model(xic_flat)
        pred_y = torch.unflatten(pred_y, dim=0, sizes=unflat_shape)
        pred_y = torch.mean(pred_y, dim=1)  # mean over samples
        # Get the last prediction
        pred_y = pred_y[..., -T_fcst:]
        # Get the corresponding true value and compute the score
        y = frame_true[..., frame_olr.shape[1] : frame_olr.shape[1] + T_fcst].unsqueeze(
            0
        )
        _score = myeval(
            torch.swapaxes(pred_y, 0, -1).squeeze(-1),
            torch.swapaxes(y, 0, -1).squeeze(-1),
        )
        score = torch.concatenate((score, _score), axis=0)
        # Append the prediction to the frame
        frame_olr = torch.concatenate((frame_olr, pred_y.squeeze(0)), axis=1)
    return frame_olr, score


def unroll_cl_observer(
    model,
    initial_data,
    frame_true,
    T_in,
    T_out,
    sensor_xind,
    sample,
    n_samples,
    myeval,
    gp_error,
    T_fcst=None,
    max_fcst=np.inf,
):
    """
    Unrolls the closed-loop observer with reset (main contribution).
    The observer is initialized with GP regression, after which the following applies:
    - Make an open-loop prediction. Collect new measurement. Move input window and apply correction.
    - Repeat while shifting the prediction window in increments of T_fcst
    """
    if T_fcst == None:
        T_fcst = T_out
    # First initial condition
    x_grid, xic, frame = initial_data
    frame_cl = frame
    if gp_error:
        xbcs = torch.zeros_like(frame)
    else:
        xbcs = frame
    # The first T_in:T_in + T_out - 1 states are populated with the data-based estimate
    # (so that starting the autoregression at t=0 works, since first prediction predicts rho at t=T_in+T_out)
    y = frame_true[..., T_in : T_in + T_out - 1].unsqueeze(0)
    sensor_x = x_grid[torch.newaxis, sensor_xind, :]
    sensor_y = y[:, sensor_xind, :]
    xbc, _ = interpolate(
        x_grid.cpu(), sensor_x.cpu(), sensor_y.cpu(), sample=sample, n_samples=n_samples
    )
    xbc = torch.tensor(xbc, device=model.device, dtype=torch.float)
    pred_y = torch.mean(xbc[..., :-1], dim=1)
    score = myeval(
        torch.swapaxes(pred_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1)
    )
    frame_cl = torch.concat((frame_cl, pred_y.squeeze(0)), axis=1)

    y_hist = frame_true[..., : frame_cl.shape[1]]
    assert frame_cl.shape == y_hist.shape
    start_score = copy.deepcopy(score)[0]
    _score = -1
    pred_y_corrected = frame_cl[..., :T_in].unsqueeze(0)
    while frame_cl.shape[1] + T_fcst <= min(
        frame_true.shape[1], max_fcst + T_in + T_out
    ):
        # Get the input state as the corrected predictions
        xic = pred_y_corrected[..., :T_in]
        xic = torch.concatenate((xic, x_grid.unsqueeze(0)), axis=-1)
        # Predict
        pred_y = model.ic_model(xic)
        # Get the last prediction
        pred_y = pred_y[..., -T_fcst:]
        # Get the corresponding true value and compute the score
        y_fcst = frame_true[
            ..., frame_cl.shape[1] : frame_cl.shape[1] + T_fcst
        ].unsqueeze(0)
        _score = myeval(
            torch.swapaxes(pred_y, 0, -1).squeeze(-1),
            torch.swapaxes(y_fcst, 0, -1).squeeze(-1),
        )
        score = torch.concatenate((score, _score), axis=0)
        # Append the prediction to the frame
        frame_cl = torch.concat((frame_cl, pred_y.squeeze(0)), axis=1)
        # Move forward in time and gather the newly received data
        y_hist = frame_true[..., : frame_cl.shape[1]]
        assert frame_cl.shape == y_hist.shape
        # Compute corrected frame
        y = y_hist[..., -(T_out + T_in) + T_fcst : -T_in + T_fcst].unsqueeze(
            0
        )  # correction window
        sensor_y = y[:, sensor_xind, :]
        xbc, _ = interpolate(
            x_grid.cpu(),
            sensor_x.cpu(),
            sensor_y.cpu(),
            sample=sample,
            n_samples=n_samples,
        )
        xbc = torch.tensor(xbc, device=model.device, dtype=torch.float)
        xp = frame_cl[..., -(T_out + T_in) + T_fcst : -T_in + T_fcst].unsqueeze(
            0
        )  # prediction estimate
        xp = xp.unsqueeze(1)
        xp = xp.repeat(
            (1, sample * n_samples + (1 - sample), 1, 1)
        )  # repeat for n_samples to be compatible with xbcs
        pred_y_corrected = model(
            x1=xp, x2=xbc
        )  # output prediction is averaged across input samples in fwd call
        # Store data-based estimate for visualization
        xbc = xbc[..., -T_out - 1 : -T_out + T_fcst - 1]
        if gp_error:
            xbcs = torch.concatenate(
                (xbcs, (pred_y - torch.mean(xbc, dim=1)).squeeze(0)), axis=1
            )  # display gp error
        else:
            xbcs = torch.concatenate(
                (xbcs, (torch.mean(xbc, dim=1)).squeeze(0)), axis=1
            )  # display gp
    return frame_cl, xbcs, score


def unroll_observers(
    model,
    loader,
    device,
    T_in,
    T_out,
    N_sensors,
    myeval_frame=LpLoss(d=2, p=2),
    myeval_t=LpLoss(d=1, p=2, reduce_dims=None),
    sample=False,
    n_samples=1,
    max_fcst=np.inf,
    gp_error=True,
    std_y=0,
):
    """Unrolls the observers"""
    model.eval()
    model.ic_model.eval()
    y_cl = []
    y_true = []
    y_true_noisy = []
    y_ol = []
    y_olr = []
    bcs = []
    scores_cl = []
    scores_ol = []
    scores_olr = []
    frame_scores_cl = []
    frame_scores_ol = []
    frame_scores_olr = []
    if max_fcst == np.inf:
        max_unroll = np.inf
    else:
        max_unroll = np.ceil(max_fcst / T_out) * T_out
    assert T_in <= T_out
    with torch.no_grad():
        print("Unrolling observers for each sample...")
        for i, _data in tqdm(enumerate(loader)):
            all_x, all_y = _data
            all_x, all_y = all_x.to(device), all_y.to(device)
            all_x, all_y = all_x.squeeze(0), all_y.squeeze(0)
            data = (all_x, all_y)
            # Build true frame from slices
            frame_true = build_true_frame(data=data, max_unroll=max_unroll)
            # Build noisy true frame from slices
            all_x = torch.concatenate(
                (
                    torch.clamp(
                        all_x[..., :-1] + std_y * torch.randn_like(all_x[..., :-1]),
                        0,
                        1,
                    ),
                    all_x[..., [-1]],
                ),
                axis=-1,
            )
            all_y = torch.clamp(all_y + std_y * torch.randn_like(all_y), 0, 1)
            data = (all_x, all_y)
            frame_true_noisy = build_true_frame(data=data, max_unroll=max_unroll)

            Nx = all_x.shape[1]
            sensor_xind = np.array([int(x) for x in np.linspace(0, Nx - 1, N_sensors)])

            initial_data = gen_initial_data(
                input=all_x[0], sensor_xind=sensor_xind, sample=False
            )

            frame_ol, score = unroll_ol_observer(
                model=model,
                initial_data=initial_data,
                frame_true=frame_true_noisy,
                T_in=T_in,
                T_out=T_out,
                sensor_xind=sensor_xind,
                sample=sample,
                n_samples=n_samples,
                myeval=myeval_t,
                T_fcst=1,
                max_fcst=max_fcst,
            )  # frame_true is used for score
            scores_ol.append(score.cpu().numpy())

            frame_olr, score = unroll_olr_observer(
                model=model,
                initial_data=initial_data,
                frame_true=frame_true_noisy,
                T_in=T_in,
                T_out=T_out,
                sensor_xind=sensor_xind,
                sample=sample,
                n_samples=n_samples,
                myeval=myeval_t,
                T_fcst=1,
                max_fcst=max_fcst,
            )  # frame_true is used for ic reset and score
            scores_olr.append(score.cpu().numpy())

            frame_cl, xbcs, score = unroll_cl_observer(
                model=model,
                initial_data=initial_data,
                frame_true=frame_true_noisy,
                T_in=T_in,
                T_out=T_out,
                sensor_xind=sensor_xind,
                sample=sample,
                n_samples=n_samples,
                myeval=myeval_t,
                gp_error=gp_error,
                T_fcst=1,
                max_fcst=max_fcst,
            )  # frame_true is used for correction measurements and score
            scores_cl.append(score.cpu().numpy())

            Tmax = min(
                frame_cl.shape[1], frame_true.shape[1]
            )  # if cap unroll, have mismatch in T
            frame_true = frame_true[..., :Tmax]
            frame_true_noisy = frame_true_noisy[..., :Tmax]
            frame_ol = frame_ol[..., :Tmax]
            frame_olr = frame_olr[..., :Tmax]
            frame_cl = frame_cl[..., :Tmax]
            xbcs = xbcs[..., :Tmax]

            score = myeval_frame(
                frame_cl[:, T_in:].unsqueeze(0),
                frame_true[:, T_in : frame_cl.shape[1]].unsqueeze(0),
            )
            frame_scores_cl.append(score.cpu().numpy())
            score = myeval_frame(
                frame_ol[:, T_in:].unsqueeze(0),
                frame_true[:, T_in : frame_ol.shape[1]].unsqueeze(0),
            )
            frame_scores_ol.append(score.cpu().numpy())
            score = myeval_frame(
                frame_olr[:, T_in:].unsqueeze(0),
                frame_true[:, T_in : frame_olr.shape[1]].unsqueeze(0),
            )
            frame_scores_olr.append(score.cpu().numpy())

            frame_true = frame_true.cpu().numpy()
            frame_true_noisy = frame_true_noisy.cpu().numpy()
            frame_ol = frame_ol.cpu().numpy()
            frame_olr = frame_olr.cpu().numpy()
            frame_cl = frame_cl.cpu().numpy()
            xbcs = xbcs.cpu().numpy()

            y_true.append(frame_true)
            y_true_noisy.append(frame_true_noisy)
            y_ol.append(frame_ol)
            y_olr.append(frame_olr)
            y_cl.append(frame_cl)
            bcs.append(xbcs)

    results = {
        "y_true": np.array(y_true),
        "y_true_noisy": np.array(y_true_noisy),
        "y_cl": np.array(y_cl),
        "y_ol": np.array(y_ol),
        "y_olr": np.array(y_olr),
        "bcs": np.array(bcs),
        "frame_scores_cl": np.array(frame_scores_cl),
        "frame_scores_ol": np.array(frame_scores_ol),
        "frame_scores_olr": np.array(frame_scores_olr),
        "scores_cl": np.array(scores_cl),
        "scores_ol": np.array(scores_ol),
        "scores_olr": np.array(scores_olr),
    }
    return results


def unroll_autoregression(
    model, loader, device, T_in, T_out, myeval_frame=LpLoss(d=2, p=2), max_unroll=np.inf
):
    """Open-loop unrolling (autoregression)"""
    N = len(loader)
    model.eval()
    frame_scores_pred = []
    y_pred = []
    y_true = []

    assert T_in <= T_out
    with torch.no_grad():
        for i, data in enumerate(loader):
            all_x, all_y = data
            all_x, all_y = all_x.to(device), all_y.to(device)
            all_x, all_y = all_x.squeeze(0), all_y.squeeze(0)
            ic = all_x[0]
            n_unroll = 1
            for j in range(all_x.shape[0]):
                data_x, data_y = all_x[j], all_y[j]
                if j == 0:
                    frame_true = torch.concat((data_x[:, :-1], data_y), axis=1)
                else:
                    sub_ivl = torch.concat((data_x[:, :-1], data_y), axis=1)
                    frame_true = torch.concat((frame_true, sub_ivl), axis=1)
                if n_unroll >= max_unroll:
                    break
                else:
                    n_unroll += 1
            frame_pred = ic[:, :-1]
            while frame_pred.shape[1] + T_out <= frame_true.shape[1]:
                pred_y = model(ic.unsqueeze(0)).squeeze(0)
                frame_pred = torch.concat((frame_pred, pred_y), axis=1)
                ic = torch.concat((pred_y[:, -T_in:], ic[:, [-1]]), axis=1)

            Tmax = min(
                frame_pred.shape[1], frame_true.shape[1]
            )  # if cap unroll, have mismatch in T
            frame_true = frame_true[..., :Tmax]
            frame_pred = frame_pred[..., :Tmax]
            if myeval_frame is not None:
                frame_scores_pred.append(
                    myeval_frame(
                        frame_pred[:, T_in:].unsqueeze(0),
                        frame_true[:, T_in:].unsqueeze(0),
                    ).item()
                )
            frame_pred = frame_pred.cpu().numpy()
            frame_true = frame_true.cpu().numpy()
            y_pred.append(frame_pred)
            y_true.append(frame_true)
    results = {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "frame_scores_pred": np.array(frame_scores_pred),
    }
    return results
