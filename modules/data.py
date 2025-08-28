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
    """
    Generate training data by splitting each sample into subsequence input-output pairs of length T_in, T_out
    Return one batch with all slice pairs (this will be shuffled)
    """
    if max_dset is None:
        max_dset = np.inf
    fnames = get_fnames(dir)
    print("Loading training data...")
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
    """
    Generate test data by splitting each sample into subsequence input-output pairs of length T_in, T_out
    Return one batch of slice pairs per sample
    """
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
    
def gpr(x_train, y_train, x_infer, sample=False, n_samples=1):
    x_train = x_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    x_infer = x_infer.cpu().numpy()
    kernel = None # ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed") + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_train, y_train)
    if sample:
        pred = gpr.sample_y(x_infer, n_samples=n_samples)
    else:
        pred, std = gpr.predict(x_infer, return_std=True)
        pred, std = np.expand_dims(pred, axis=-1), np.expand_dims(std, axis=-1)
    if sample:
        pred = np.transpose(pred, (2, 0, 1))
        return pred, None
    else:
        pred = np.transpose(pred, (2, 0, 1))
        std = np.transpose(std, (2, 0, 1))
        return pred, std

def interpolate(x_grid, sensor_x, sensor_y, sample=False, n_samples=1, disable=True):
    """Interpolate measurements to get data-based estimate
    Input:
        x_grid: shape [Nx, 1]
        sensor_x: shape [batch, N_sensors, 1]
        sensor_y: shape [batch, N_sensors, N_rho]
    Output:
        Xipts: shape [batch, n_samples, Nx, N_rho + 1]
    """
    Nx = x_grid.shape[0] # xgrid has shape [123, 1]
    Xipts = np.empty((0, n_samples * sample + (1- sample), Nx, sensor_y.shape[-1] + 1)) # only for 1d currently
    if sample is False:
        Xipts_stds = np.empty((0, n_samples * sample + (1 - sample), Nx, sensor_y.shape[-1]))
    else:
        Xipts_stds = None
    for i in tqdm(range(sensor_y.shape[0]), disable=disable):
        pred, stds = gpr(sensor_x[i], sensor_y[i], x_grid, sample=sample, n_samples=n_samples)
        x_grid_repeated = np.repeat(x_grid, axis=-1, repeats=(n_samples * sample + (1- sample)))
        x_grid_repeated = np.transpose(x_grid_repeated, axes=(1, 0)).unsqueeze(-1)
        xipt = np.concatenate((pred, x_grid_repeated.cpu().numpy()), axis=-1) # model input includes function values and grid values
        Xipts = np.concatenate((Xipts, np.expand_dims(xipt, axis=0)), axis=0)
        if sample is False:
            Xipts_stds = np.concatenate((Xipts_stds, np.expand_dims(stds, axis=0)), axis=0)
    return Xipts, Xipts_stds