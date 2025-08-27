import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import matplotlib.pyplot as plt

def read_density(name):
    with open(f"{name}.csv", "r", newline="") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Extract L and Tmax from the first row
    L, Tmax = map(float, data[0])  # Convert to float if necessary
    density = [list(map(float, row)) for row in data[1:]]  # Convert remaining rows to float

    return L, Tmax, np.array(density)

def load_config(file):
    if file is None:
        return None
    with open(file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config

def get_fnames(dir):
    fnames = []
    directory = os.fsencode(dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            fnames.append(os.path.join(dir, filename))
    return fnames

def gen_data_train(dir, T_in=1, T_out=1, max_dset=None, only_first=False):
    if max_dset is None:
        max_dset = np.inf
    fnames = get_fnames(dir)
    print("Loading training data")
    for i in tqdm(range(len(fnames))):
        filename = fnames[i]
        L, Tmax, rho = read_density(filename[:-4])
        if i == 0:            
            Nx = rho.shape[0]
            Xs = np.empty(shape=(0, Nx, T_in + 1))
            ys = np.empty(shape=(0, Nx, T_out))
        x, deltaX = np.linspace(0, L, Nx, retstep=True)  
        start = 0
        while start + T_in + T_out <= rho.shape[1]:
            ic = rho[:, start:start + T_in]
            ipt = np.concatenate((ic, x.reshape(-1, 1)), axis=1)
            opt = rho[:, start + T_in:start + T_in + T_out]
            Xs = np.append(Xs, np.expand_dims(ipt, axis=0), axis=0)
            ys = np.append(ys, np.expand_dims(opt, axis=0), axis=0)
            start = start + T_in + T_out
            if only_first:
                break
        if i >= max_dset - 1:
            break

    Xs = torch.Tensor(Xs)
    ys = torch.Tensor(ys)

    deltaT = Tmax / (rho.shape[1] - 1)
    return Xs, ys, deltaT, deltaX

def gen_data_test(dir, T_in=1, T_out=1, max_dset=None, only_first=False):
    if max_dset is None:
        max_dset = np.inf
    fnames = get_fnames(dir)
    Xss = []
    yss = []
    print("Loading test data...")
    for i in tqdm(range(len(fnames))):
        filename = fnames[i]
        L, Tmax, rho = read_density(filename[:-4])
        if i == 0:            
            Nx = rho.shape[0]
        x, deltaX = np.linspace(0, L, Nx, retstep=True)  
        start = 0
        Xs = np.empty(shape=(0, Nx, T_in + 1))
        ys = np.empty(shape=(0, Nx, T_out))
        while start + T_in + T_out <= rho.shape[1]:
            ic = rho[:, start:start + T_in]
            ipt = np.concatenate((ic, x.reshape(-1, 1)), axis=1)
            opt = rho[:, start + T_in:start + T_in + T_out]
            Xs = np.append(Xs, np.expand_dims(ipt, axis=0), axis=0)
            ys = np.append(ys, np.expand_dims(opt, axis=0), axis=0)
            start = start + T_in + T_out
            if only_first:
                break
        Xss.append(Xs)
        yss.append(ys)
        
        if i >= max_dset - 1:
            break

    Xss = torch.Tensor(np.array(Xss))
    yss = torch.Tensor(np.array(yss))

    deltaT = Tmax / (rho.shape[1] - 1)
    return Xss, yss, deltaT, deltaX
    
def gpr(sensor_x, sensor_ys, all_x, sample=False, n_samples=1):
    sensor_x = sensor_x.cpu().numpy()
    sensor_ys = sensor_ys.cpu().numpy()
    all_x = all_x.cpu().numpy()
    kernel = None # ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed") + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(sensor_x, sensor_ys)
    if sample:
        pred = gpr.sample_y(all_x, n_samples=n_samples)
    else:
        pred, std = gpr.predict(all_x, return_std=True)
        pred, std = np.expand_dims(pred, axis=-1), np.expand_dims(std, axis=-1)
    if sample:
        pred = np.transpose(pred, (2, 0, 1))
        return pred, None
    else:
        pred = np.transpose(pred, (2, 0, 1))
        std = np.transpose(std, (2, 0, 1))
        return pred, std
        
def gpr_ics(Xs, N_sensors, sample=False, n_samples=1, disable=True):
    # GP sample interpolation of Xs
    Nx = Xs.shape[1]
    Xics = np.empty((0, n_samples * sample + (1- sample), Nx, Xs.shape[-1])) # only for 1d currently
    if sample is False:
        Xics_stds = np.empty((0, n_samples * sample + (1- sample), Nx, Xs.shape[-1] - 1))
    else:
        Xics_stds = None
    
    for datapoint in tqdm(range(Xs.shape[0]), disable=disable):
        sensor_xind = np.array([int(x) for x in np.linspace(0, Nx - 1, N_sensors)])
        all_x = Xs[datapoint, :, [-1]]
        sensor_x = Xs[datapoint, sensor_xind][:, [-1]]
        sensor_ys = Xs[datapoint, sensor_xind, :-1]
        pred, stds = gpr(sensor_x, sensor_ys, all_x, sample=sample, n_samples=n_samples)
        #print(pred.shape, sensor_ys.shape)
        #pred = sensor_ys.unsqueeze(0)
        all_x_repeated = np.repeat(all_x, axis=-1, repeats=(n_samples * sample + (1- sample)))
        all_x_repeated = np.transpose(all_x_repeated, axes=(1, 0)).unsqueeze(-1)
        xic = np.concatenate((pred, all_x_repeated.cpu().numpy()), axis=-1)
        Xics = np.concatenate((Xics, np.expand_dims(xic, axis=0)), axis=0)
        if sample is False:
            Xics_stds = np.concatenate((Xics_stds, np.expand_dims(stds, axis=0)), axis=0)
    return Xics, Xics_stds

def gpr_bcs(Xs, ys, N_sensors, sample=False, n_samples=1, disable=True):
    # GP sample interpolation of Xs
    Nx = Xs.shape[1]
    Xbcs = np.empty((0, n_samples * sample + (1- sample), Nx, ys.shape[-1] + 1)) # only for 1d currently
    if sample is False:
        Xbcs_stds = np.empty((0, n_samples * sample + (1- sample), Nx, ys.shape[-1]))
    else:
        Xbcs_stds = None
    
    for datapoint in tqdm(range(Xs.shape[0]), disable=disable):
        sensor_xind = np.array([int(x) for x in np.linspace(0, Nx - 1, N_sensors)])
        all_x = Xs[datapoint, :, [-1]]
        sensor_x = Xs[datapoint, sensor_xind, -1:]
        sensor_ys = ys[datapoint, sensor_xind, ...]
        pred, stds = gpr(sensor_x, sensor_ys, all_x, sample=sample, n_samples=n_samples)
        #_pred, stds = gpr(sensor_x, sensor_ys, all_x, sample=sample, n_samples=n_samples) # test debug
        #pred, _ = sensor_ys, sensor_ys # test debug
        #pred = pred.reshape(_pred.shape) # test debug
#        print(pred.shape, sensor_ys.shape)
#        pred = sensor_ys.unsqueeze(0)
        all_x_repeated = np.repeat(all_x, axis=-1, repeats=(n_samples * sample + (1- sample)))
        all_x_repeated = np.transpose(all_x_repeated, axes=(1, 0)).unsqueeze(-1)
        xbc = np.concatenate((pred, all_x_repeated.cpu().numpy()), axis=-1)
        Xbcs = np.concatenate((Xbcs, np.expand_dims(xbc, axis=0)), axis=0)
        if sample is False:
            Xbcs_stds = np.concatenate((Xbcs_stds, np.expand_dims(stds, axis=0)), axis=0)
    return Xbcs, Xbcs_stds


def plot_gpr_bcs(Xbcs, Xbcs_stds, N_sensors, max_output_dim_show=None, only_first_datapoint=True):
    """
    Plot GPR results
    """
    output_dim = Xbcs_stds.shape[-1]
    if max_output_dim_show is None:
        max_output_dim_show = output_dim
    Nx = Xbcs.shape[1]
    sensor_xind = np.array([int(x) for x in np.linspace(0, Nx - 1, N_sensors)])
    for datapoint in range(Xbcs.shape[0]):
        nrows = min(max_output_dim_show, output_dim) // 2 + 1
        fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(25, 3*nrows))
        for i in range(min(max_output_dim_show, output_dim)):
            ax[i//2, i%2].plot(Xbcs[datapoint, :, -1], Xbcs[datapoint, :, i])
            ax[i//2, i%2].scatter(Xbcs[datapoint, sensor_xind, -1], Xbcs[datapoint, sensor_xind, i], color="black")
            ax[i//2, i%2].fill_between(Xbcs[datapoint, :, -1], Xbcs[datapoint, :, i] - Xbcs_stds[datapoint, :, i], Xbcs[datapoint, :, i] + Xbcs_stds[datapoint, :, i])
        if only_first_datapoint:
            break