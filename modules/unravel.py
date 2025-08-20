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
from modules.data import gpr_bcs, gpr_ics
from tqdm import tqdm

def unravel(model, loader, device, T_in, T_out, myeval_frame=LpLoss(d=2, p=2), max_unravel=np.inf):
    """Open-loop unrolling (autoregression)"""
    N = len(loader)
    model.eval()
    full_scores_pred = []
    ypred= []
    ytrue = []

    assert T_in <= T_out
    with torch.no_grad():
        for i, data in enumerate(loader):
            all_x, all_y = data
            all_x, all_y = all_x.to(device), all_y.to(device)   
            all_x, all_y = all_x.squeeze(0), all_y.squeeze(0)
            ic = all_x[0]
            n_unravel = 1
            for j in range(all_x.shape[0]):
                data_x, data_y = all_x[j], all_y[j]
                if j == 0:
                    frame_true = torch.concat((data_x[:, :-1], data_y), axis=1)
                else:
                    sub_ivl = torch.concat((data_x[:, :-1], data_y), axis=1)
                    frame_true = torch.concat((frame_true, sub_ivl), axis=1)
                if n_unravel >= max_unravel:
                    break
                else:
                    n_unravel += 1
            frame_pred = ic[:, :-1]
            while frame_pred.shape[1] + T_out <= frame_true.shape[1]:
                pred_y = model(ic.unsqueeze(0)).squeeze(0)
                frame_pred = torch.concat((frame_pred, pred_y), axis=1)
                ic = torch.concat((pred_y[:, -T_in:], ic[:, [-1]]), axis=1)
            
            Tmax = min(frame_pred.shape[1], frame_true.shape[1]) # if cap unravel, have mismatch in T
            frame_true = frame_true[..., :Tmax]
            frame_pred = frame_pred[..., :Tmax]
            if myeval_frame is not None:
                full_scores_pred.append(myeval_frame(frame_pred[:, T_in:].unsqueeze(0), frame_true[:, T_in:].unsqueeze(0)).item())
            frame_pred = frame_pred.cpu().numpy()
            frame_true = frame_true.cpu().numpy()
            ypred.append(frame_pred)
            ytrue.append(frame_true)
    results = {
        "ytrue":np.array(ytrue),
        "ypred":np.array(ypred),
        "full_scores_pred":np.array(full_scores_pred)}
    return results

def unravel_frame_true(data, max_unravel):
    all_x, all_y = data
    n_unravel = 1
    # Unravel true frame
    for j in range(all_x.shape[0]):
        data_x, data_y = all_x[j], all_y[j]
        if j == 0:
            frame_true = torch.concat((data_x[..., :-1], data_y), axis=1)
        else:
            sub_ivl = torch.concat((data_x[..., :-1], data_y), axis=1)
            frame_true = torch.concat((frame_true, sub_ivl), axis=1)
        if n_unravel >= max_unravel:
            break
        else:
            n_unravel += 1
    return frame_true

def _first_xic(data, N_sensors, sample, n_samples=0):
    all_x, all_y = data
    # Generate initial condition
    _xic = all_x[[0]] # deterministic, used to get the x values       
    first_xic, _ = gpr_ics(_xic.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
    first_xic = torch.tensor(first_xic, device=all_x.device.type, dtype=torch.float)
    display_xic = first_xic[..., :-1].squeeze(0)
    display_xic = torch.mean(display_xic, dim=0) # display_xic[0] to get one sample
    return _xic, first_xic, display_xic

def unravel_frame_base(model, ic_data, frame_true, T_in, T_out, N_sensors, sample, n_samples, myeval, T_fcst=None, max_fcst=np.inf):
    if T_fcst == None:
        T_fcst=T_out
    # Unravel base model
    # For the first initial condition: sample ic and average base prediction output
    _xic, first_xic, display_xic = ic_data
    frame_base = display_xic
    y = frame_true[..., T_in:T_in + T_out - 1].unsqueeze(0)
    xbc, _ = gpr_bcs(_xic.cpu(), y.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
    xbc = torch.tensor(xbc, device=first_xic.device.type, dtype=torch.float)
    base_y = torch.mean(xbc[..., :-1], dim=1)
    score = myeval(torch.swapaxes(base_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1)) # compute error per time - time along batch dimension, squeeze original batch dimension
    frame_base = torch.concatenate((frame_base, base_y.squeeze(0)), axis=1)
    start_score = copy.deepcopy(score)[0]
    _score = -1
    while frame_base.shape[1] + T_fcst <= min(frame_true.shape[1], max_fcst + T_in + T_out):
        if _score > max(6 * start_score, 0.75):
            score = torch.concatenate((score, _score), axis=0)
            mean = torch.mean(base_y)
            if mean > 0.5:
                instab = torch.ones_like(base_y.squeeze(0))
            else:
                instab = torch.zeros_like(base_y.squeeze(0))
            frame_base = torch.concatenate((frame_base, instab), axis=1)
            continue
        xic = frame_base[..., -(T_out + T_in - T_fcst):-(T_out - T_fcst)].unsqueeze(0)
        xic = torch.concatenate((xic, _xic[..., [-1]]), axis=-1)
        base_y = model.ic_model(xic)
        base_y = base_y[..., -T_fcst:]
        y = frame_true[..., frame_base.shape[1]:frame_base.shape[1] + T_fcst].unsqueeze(0)
        _score = myeval(torch.swapaxes(base_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1))
        score = torch.concatenate((score, _score), axis=0)
        frame_base = torch.concatenate((frame_base, base_y.squeeze(0)), axis=1)
    return frame_base, score

def unravel_frame_base_reset(model, ic_data, frame_true, T_in, T_out, N_sensors, sample, n_samples, myeval, T_fcst=None, max_fcst=np.inf):
    if T_fcst == None:
        T_fcst=T_out
    # Unravel base model
    # For the first initial condition: sample ic and average base prediction output
    _xic, first_xic, display_xic = ic_data
    frame_base = display_xic
    y = frame_true[..., T_in:T_in + T_out - 1].unsqueeze(0)
    xbc, _ = gpr_bcs(_xic.cpu(), y.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
    xbc = torch.tensor(xbc, device=first_xic.device.type, dtype=torch.float)
    base_y = torch.mean(xbc[..., :-1], dim=1)
    score = myeval(torch.swapaxes(base_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1)) # compute error per time - time along batch dimension, squeeze original batch dimension
    frame_base = torch.concatenate((frame_base, base_y.squeeze(0)), axis=1)
    start_score = copy.deepcopy(score)[0]
    _score = -1
    while frame_base.shape[1] + T_fcst <= min(frame_true.shape[1], max_fcst + T_in + T_out):
        # Sample GP ICs
        xic = frame_true[..., -(T_out + T_in - T_fcst) + frame_base.shape[1]:-(T_out - T_fcst) + frame_base.shape[1]].unsqueeze(0)
        xic = torch.concatenate((xic, _xic[..., [-1]]), axis=-1)
        xic, _ = gpr_ics(xic.cpu(), N_sensors=N_sensors, sample=False)
        xic = torch.tensor(xic, device=first_xic.device.type, dtype=torch.float)
        unflat_shape = xic.shape[:2]
        xic_flat = xic.flatten(start_dim=0, end_dim=1)
        base_y = model.ic_model(xic_flat)
        base_y = torch.unflatten(base_y, dim=0, sizes=unflat_shape)
        base_y = torch.mean(base_y, dim=1) # mean over samples
        base_y = base_y[..., -T_fcst:]
        y = frame_true[..., frame_base.shape[1]:frame_base.shape[1] + T_fcst].unsqueeze(0)
        _score = myeval(torch.swapaxes(base_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1))
        score = torch.concatenate((score, _score), axis=0)
        frame_base = torch.concatenate((frame_base, base_y.squeeze(0)), axis=1)
    return frame_base, score


def unravel_frame_corrected(model, ic_data, frame_true, T_in, T_out, N_sensors, sample, n_samples, myeval, gp_error, T_fcst=None, max_fcst=np.inf):
    if T_fcst == None:
        T_fcst=T_out
    # Unravel sequential model
    # For the first initial condition: sample ic and average base prediction output
    _xic, first_xic, display_xic = ic_data
    frame_pred = display_xic # mean of samples
    if gp_error:
        xbcs = display_xic - display_xic
    else:
        xbcs = display_xic
    y = frame_true[..., T_in:T_in + T_out].unsqueeze(0)
    xbc, _ = gpr_bcs(_xic.cpu(), y.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
    xbc = torch.tensor(xbc, device=first_xic.device.type, dtype=torch.float)
    pred_y_corrected = torch.mean(xbc[..., :-1], dim=1)
    if gp_error:
        xbcs = torch.concatenate((xbcs, (pred_y_corrected - torch.mean(xbc[..., :-1], dim=1)).squeeze(0)), axis=1) # display gp error
    else:
        xbcs = torch.concatenate((xbcs, (torch.mean(xbc[..., :-1], dim=1)).squeeze(0)), axis=1) # display gp
    score = myeval(torch.swapaxes(pred_y_corrected, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1))
    frame_pred = torch.concat((frame_pred, pred_y_corrected.squeeze(0)), axis=1)
    start_score = copy.deepcopy(score)[0]
    _score = - 1
    while frame_pred.shape[1] + T_fcst <= min(frame_true.shape[1], max_fcst + T_in + T_out):
        # Compute corrected frame
        xic = frame_pred[..., -(T_out + T_in - T_fcst):-(T_out - T_fcst)].unsqueeze(0)
        xic = torch.concatenate((xic, _xic[..., [-1]]), axis=-1)
        xic = xic.unsqueeze(1)
        xic = xic.repeat((1, sample * n_samples + (1 - sample), 1, 1)) # repeat for n_samples to be compatible with xbcs
        y = frame_true[..., frame_pred.shape[1] - T_out + T_fcst:frame_pred.shape[1] + T_fcst].unsqueeze(0)
        xbc, _ = gpr_bcs(_xic.cpu(), y.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
        xbc = torch.tensor(xbc, device=first_xic.device.type, dtype=torch.float)
        ic = torch.concatenate((xic, xbc), axis=-1)
        pred_y_corrected = model(ic) # output prediction is averaged across input samples in fwd call
        pred_y_corrected = pred_y_corrected[..., -T_fcst:]
        xbc = xbc[..., -T_fcst-1:-1]
        y = y[..., -T_fcst:]
        if gp_error:
            xbcs = torch.concatenate((xbcs, (pred_y_corrected - torch.mean(xbc, dim=1)).squeeze(0)), axis=1) # display gp error
        else:
            xbcs = torch.concatenate((xbcs, (torch.mean(xbc, dim=1)).squeeze(0)), axis=1) # display gp
        _score = myeval(torch.swapaxes(pred_y_corrected, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1))
        score = torch.concatenate((score, _score), axis=0)
        frame_pred = torch.concat((frame_pred, pred_y_corrected.squeeze(0)), axis=1)        
    return frame_pred, xbcs, score

def unravel_frame_uncorrected(model, ic_data, frame_true, T_in, T_out, N_sensors, sample, n_samples, myeval, gp_error, T_fcst=None, max_fcst=np.inf):
    if T_fcst == None:
        T_fcst=T_out
    # Unravel sequential model
    # For the first initial condition: sample ic and average base prediction output
    _xic, first_xic, display_xic = ic_data
    frame_pred = display_xic # mean of samples
    if gp_error:
        xbcs = display_xic - display_xic
    else:
        xbcs = display_xic
    y = frame_true[..., T_in:T_in + T_out - 1].unsqueeze(0)
    xbc, _ = gpr_bcs(_xic.cpu(), y.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
    xbc = torch.tensor(xbc, device=first_xic.device.type, dtype=torch.float)
    pred_y = torch.mean(xbc[..., :-1], dim=1)
    score = myeval(torch.swapaxes(pred_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1))
    frame_pred = torch.concat((frame_pred, pred_y.squeeze(0)), axis=1)
    y_hist = frame_true[..., :frame_pred.shape[1]] 
    assert frame_pred.shape == y_hist.shape
    start_score = copy.deepcopy(score)[0]
    _score = -1
    pred_y_corrected = frame_pred[..., :T_in].unsqueeze(0)
    while frame_pred.shape[1] + T_fcst <= min(frame_true.shape[1], max_fcst + T_in + T_out):
        xic = pred_y_corrected[..., :T_in]
        xic = torch.concatenate((xic, _xic[..., [-1]]), axis=-1)
        pred_y = model.ic_model(xic)
        pred_y = pred_y[..., -T_fcst:]
        y_fcst = frame_true[..., frame_pred.shape[1]:frame_pred.shape[1] + T_fcst].unsqueeze(0)
        _score = myeval(torch.swapaxes(pred_y, 0, -1).squeeze(-1), torch.swapaxes(y_fcst, 0, -1).squeeze(-1))
        score = torch.concatenate((score, _score), axis=0)
        frame_pred = torch.concat((frame_pred, pred_y.squeeze(0)), axis=1)
        y_hist = frame_true[..., :frame_pred.shape[1]] 
        assert frame_pred.shape == y_hist.shape        
        # Compute corrected frame
        y = y_hist[..., - (T_out + T_in)+T_fcst:- T_in+T_fcst].unsqueeze(0)
        xbc, _ = gpr_bcs(_xic.cpu(), y.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
        xbc = torch.tensor(xbc, device=first_xic.device.type, dtype=torch.float)
        x1 = frame_pred[..., -(T_out + T_in)+T_fcst:-T_in+T_fcst].unsqueeze(0)
        x1 = x1.unsqueeze(1)
        x1 = x1.repeat((1, sample * n_samples + (1 - sample), 1, 1)) # repeat for n_samples to be compatible with xbcs
        pred_y_corrected = model(x1=x1, x2=xbc) # output prediction is averaged across input samples in fwd call
        xbc = xbc[..., -T_out-1:-T_out+T_fcst-1]
        if gp_error:
            xbcs = torch.concatenate((xbcs, (pred_y - torch.mean(xbc, dim=1)).squeeze(0)), axis=1) # display gp error
        else:
            xbcs = torch.concatenate((xbcs, (torch.mean(xbc, dim=1)).squeeze(0)), axis=1) # display gp
    return frame_pred, xbcs, score
    
def unravel_dual(model, model_corr, loader, device, T_in, T_out, N_sensors, myeval_frame=LpLoss(d=2, p=2), myeval_t=LpLoss(d=1, p=2, reduce_dims=None), sample=False, n_samples=1, max_fcst=np.inf, gp_error=True, std_y=0):
    model.eval()
    model.ic_model.eval()
    ypred= []
    ytrue = []
    ytrue_noisy = []
    ybase = []
    ybase_reset = []
    ycorrected = []
    bcs = []
    scores_pred = []
    scores_base = []
    scores_base_reset = []
    scores_corrected = []
    full_scores_pred = []
    full_scores_base = []
    full_scores_base_reset = []
    full_scores_corrected = []
    if max_fcst == np.inf:
        max_unravel = np.inf
    else:
        max_unravel = np.ceil(max_fcst / T_out) * T_out
    assert T_in <= T_out
    with torch.no_grad():
        for i, _data in tqdm(enumerate(loader)):
            all_x, all_y = _data
            all_x, all_y = all_x.to(device), all_y.to(device)   
            all_x, all_y = all_x.squeeze(0), all_y.squeeze(0)
            data = (all_x, all_y)
            frame_true = unravel_frame_true(data=data, max_unravel=max_unravel)
            # Perturb data
            all_x = torch.concatenate((
                torch.clamp(all_x[..., :-1] + std_y*torch.randn_like(all_x[..., :-1]), 0, 1),
                all_x[..., [-1]]), axis=-1)
            all_y = torch.clamp(all_y + std_y*torch.randn_like(all_y), 0, 1)
            data = (all_x, all_y)
            frame_true_noisy = unravel_frame_true(data=data, max_unravel=max_unravel)

            ic_data = _first_xic(data=data, N_sensors=N_sensors, sample=False) #_first_xic(data=data, N_sensors=N_sensors, sample=sample, n_samples=n_samples)
            frame_base, score = unravel_frame_base(model=model, ic_data=ic_data, frame_true=frame_true_noisy, T_in=T_in, T_out=T_out, N_sensors=N_sensors, sample=sample, n_samples=n_samples, myeval=myeval_t, T_fcst=1, max_fcst=max_fcst) # frame_true is used for score
            scores_base.append(score.cpu().numpy())

            frame_base_reset, score = unravel_frame_base_reset(model=model, ic_data=ic_data, frame_true=frame_true_noisy, T_in=T_in, T_out=T_out, N_sensors=N_sensors, sample=sample, n_samples=n_samples, myeval=myeval_t, T_fcst=1, max_fcst=max_fcst) # frame_true is used for ic reset and score
            scores_base_reset.append(score.cpu().numpy())

            frame_corrected, _, score = torch.zeros_like(frame_base), None, torch.zeros_like(score) # unravel_frame_corrected(model=model, ic_data=ic_data, frame_true=frame_true_noisy, T_in=T_in, T_out=T_out, N_sensors=N_sensors, sample=sample, n_samples=n_samples, myeval=myeval_t, gp_error=gp_error, T_fcst=1, max_fcst=max_fcst) # frame_true is used for correction measurements and score
            scores_corrected.append(score.cpu().numpy())
            
            frame_pred, xbcs, score = unravel_frame_uncorrected(model=model_corr, ic_data=ic_data, frame_true=frame_true_noisy, T_in=T_in, T_out=T_out, N_sensors=N_sensors, sample=sample, n_samples=n_samples, myeval=myeval_t, gp_error=gp_error, T_fcst=1, max_fcst=max_fcst) # frame_true is used for correction measurements and score
            scores_pred.append(score.cpu().numpy())
            
            
            Tmax = min(frame_pred.shape[1], frame_true.shape[1]) # if cap unravel, have mismatch in T
            frame_true = frame_true[..., :Tmax]
            frame_true_noisy = frame_true_noisy[..., :Tmax]
            frame_base = frame_base[..., :Tmax]
            frame_base_reset = frame_base_reset[..., :Tmax]
            frame_corrected = frame_corrected[..., :Tmax]
            frame_pred = frame_pred[..., :Tmax]
            xbcs = xbcs[..., :Tmax]
            
            score = myeval_frame(frame_pred[:, T_in:].unsqueeze(0), frame_true[:, T_in:frame_pred.shape[1]].unsqueeze(0))
            full_scores_pred.append(score.cpu().numpy())
            score = myeval_frame(frame_base[:, T_in:].unsqueeze(0), frame_true[:, T_in:frame_base.shape[1]].unsqueeze(0))
            full_scores_base.append(score.cpu().numpy())
            score = myeval_frame(frame_base_reset[:, T_in:].unsqueeze(0), frame_true[:, T_in:frame_base_reset.shape[1]].unsqueeze(0))
            full_scores_base_reset.append(score.cpu().numpy())
            score = myeval_frame(frame_corrected[:, T_in:].unsqueeze(0), frame_true[:, T_in:frame_corrected.shape[1]].unsqueeze(0))
            full_scores_corrected.append(score.cpu().numpy())
            
            frame_true = frame_true.cpu().numpy()
            frame_true_noisy = frame_true_noisy.cpu().numpy()
            frame_base = frame_base.cpu().numpy()
            frame_base_reset = frame_base_reset.cpu().numpy()
            frame_corrected = frame_corrected.cpu().numpy()
            frame_pred = frame_pred.cpu().numpy()
            xbcs = xbcs.cpu().numpy()
            ytrue.append(frame_true)
            ytrue_noisy.append(frame_true_noisy)
            ybase.append(frame_base)
            ybase_reset.append(frame_base_reset)
            ycorrected.append(frame_corrected)
            ypred.append(frame_pred)
            bcs.append(xbcs)
    
    results = {
        "ytrue":np.array(ytrue),
        "ytrue_noisy":np.array(ytrue_noisy),
        "ypred":np.array(ypred),
        "ybase":np.array(ybase),
        "ybase_reset":np.array(ybase_reset),
        "ycorrected":np.array(ycorrected),
        "bcs":np.array(bcs),
        "full_scores_pred":np.array(full_scores_pred),
        "full_scores_base":np.array(full_scores_base),
        "full_scores_base_reset":np.array(full_scores_base_reset),
        "full_scores_corrected":np.array(full_scores_corrected),
        "scores_pred":np.array(scores_pred),
        "scores_base":np.array(scores_base),
        "scores_base_reset":np.array(scores_base_reset),
        "scores_corrected":np.array(scores_corrected)
    }
    return results

def unravel_frame_1step(model, ic_data, frame_true, T_in, T_out, N_sensors, sample, n_samples, myeval, gp_error):
    # Unravel sequential model
    # For the first initial condition: sample ic and average base prediction output
    _xic, first_xic, display_xic = ic_data
    frame_pred = display_xic # mean of samples
    if gp_error:
        xbcs = display_xic - display_xic
    else:
        xbcs = display_xic
    xic = copy.deepcopy(first_xic) # Model expects sampled ic
    unflat_shape = xic.shape[:2]
    xic_flat = xic.flatten(start_dim=0, end_dim=1)
    pred_y = model.ic_model(xic_flat)
    pred_y = torch.unflatten(pred_y, dim=0, sizes=unflat_shape)
    pred_y = torch.mean(pred_y, dim=1) # mean over samples
    # Compute score for uncorrected
    y = frame_true[..., frame_pred.shape[1]:frame_pred.shape[1] + T_out].unsqueeze(0)
    score = myeval(torch.swapaxes(pred_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1))
    # Correction
    xbc, _ = gpr_bcs(_xic.cpu(), y.cpu(), N_sensors=N_sensors, sample=sample, n_samples=n_samples)
    xbc = torch.tensor(xbc, device=first_xic.device.type, dtype=torch.float)
    ic = torch.concatenate((xic, xbc), axis=-1)
    # Display corrected
    pred_y_corrected = model(ic) # output prediction is averaged across input samples in fwd call
    frame_pred = torch.concat((frame_pred, pred_y_corrected.squeeze(0)), axis=1)
    if gp_error:
        xbcs = torch.concatenate((xbcs, (pred_y - torch.mean(xbc[..., :-1], dim=1)).squeeze(0)), axis=1) # display gp error
    else:
        xbcs = torch.concatenate((xbcs, (torch.mean(xbc[..., :-1], dim=1)).squeeze(0)), axis=1) # display gp
    xic = pred_y_corrected[..., -T_in:]
    xic = torch.concatenate((xic, _xic[..., [-1]]), axis=-1)
    pred_y = model.ic_model(xic)
    # Compute score for uncorrected
    y = frame_true[..., frame_pred.shape[1]:frame_pred.shape[1] + T_out].unsqueeze(0)
    _score = myeval(torch.swapaxes(pred_y, 0, -1).squeeze(-1), torch.swapaxes(y, 0, -1).squeeze(-1))
    score = torch.concatenate((score, _score), axis=0)
    # Display uncorrected 
    frame_pred = torch.concat((frame_pred, pred_y.squeeze(0)), axis=1)
    return frame_pred, xbcs, score

def unravel_dual_1step(model, loader, device, T_in, T_out, N_sensors, myeval_t=LpLoss(d=1, p=2, reduce_dims=None), sample=False, n_samples=1, gp_error=True, std_y=0, display_noisy=False):
    N = len(loader)
    model.eval()
    model.ic_model.eval()
    ypred= []
    ytrue = []
    ybase = []
    bcs = []
    score_base = []
    score_pred = []
    assert T_in <= T_out
    with torch.no_grad():
        for i, _data in enumerate(loader):
            all_x, all_y = _data
            all_x, all_y = all_x.to(device), all_y.to(device)   
            all_x, all_y = all_x.squeeze(0), all_y.squeeze(0)
            data = (all_x, all_y)
            if display_noisy:
                # Perturb data
                all_x = torch.concatenate((
                    torch.clamp(all_x[..., :-1] + std_y*torch.randn_like(all_x[..., :-1]), 0, 1),
                    all_x[..., [-1]]), axis=-1)
                all_y = torch.clamp(all_y + std_y*torch.randn_like(all_y), 0, 1)
                data = (all_x, all_y)
            frame_true = unravel_frame_true(data, max_unravel=2)
                 
            ic_data = _first_xic(data=data, N_sensors=N_sensors, sample=False) #_first_xic(data=data, N_sensors=N_sensors, sample=sample, n_samples=n_samples)
            frame_pred, xbcs, score = unravel_frame_1step(model, ic_data, frame_true, T_in, T_out, N_sensors, sample, n_samples, myeval_t, gp_error)
            score_pred.append(score.detach().cpu().numpy())
            frame_base, score = unravel_frame_base(model, ic_data, frame_true, T_in, T_out, myeval_t)
            score_base.append(score.detach().cpu().numpy())
            
            Tmax = min(frame_pred.shape[1], frame_base.shape[1], frame_true.shape[1]) # if cap unravel, have mismatch in T
            frame_true = frame_true[..., :Tmax]
            frame_base = frame_base[..., :Tmax]
            frame_pred = frame_pred[..., :Tmax]
            xbcs = xbcs[..., :Tmax]
            frame_true = frame_true.cpu().numpy()
            frame_base = frame_base.detach().cpu().numpy()
            frame_pred = frame_pred.detach().cpu().numpy()
            xbcs = xbcs.cpu().numpy()

            ytrue.append(frame_true)
            ybase.append(frame_base)
            ypred.append(frame_pred)
            bcs.append(xbcs)
    results = {
        "ytrue":np.array(ytrue),
        "ypred":np.array(ypred),
        "ybase":np.array(ybase),
        "bcs":np.array(bcs),
        "scores_pred":np.array(score_pred),
        "scores_base":np.array(score_base)
        }
    return results