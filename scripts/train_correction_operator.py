import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import argparse
import torch
import numpy as np
from modules.models import Prediction, Correction
from modules.data import gen_data_train, load_config, interpolate
from modules.train import train

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="Combined operator config file path",
    default="configs/closedloop.yaml",
    required=False,
)


class TripleTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, X, Z, y):
        assert len(X) == len(Z) == len(y)
        self.X = X
        self.Z = Z
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx], self.y[idx]


def run(config):
    directory = Path("models")
    directory.mkdir(parents=True, exist_ok=True)

    if config["train"]["device"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = config["train"]["device"]
    print("Using device:", device)

    train_data_dir = config["data"]["train_data_dir"]
    T_in = config["data"]["T_in"]
    T_out = config["data"]["T_out"]
    max_dset = config["data"]["max_dset"]
    only_first = config["data"]["only_first"]

    N_sensors = config["data"]["N_sensors"]
    sample = config["data"]["sample"]
    n_samples = config["data"]["n_samples"]
    std_y = config["data"]["std_y"]  # measurement noise - Gaussian variance
    train_noisy = config["data"]["train_noisy"]  # train on noisy data

    Xs, ys, deltaT, deltaX = gen_data_train(
        train_data_dir, T_in=T_in, T_out=T_out, max_dset=max_dset, only_first=only_first
    )

    if train_noisy:
        Xs = torch.concatenate(
            (
                torch.clamp(
                    Xs[..., :-1] + std_y * torch.randn_like(Xs[..., :-1]), 0, 1
                ),
                Xs[..., [-1]],
            ),
            axis=-1,
        )
        ys = torch.clamp(ys + std_y * torch.randn_like(ys), 0, 1)

    Nx = Xs.shape[1]
    sensor_xind = np.array([int(x) for x in np.linspace(0, Nx - 1, N_sensors)])
    x_grid = Xs[0, :, [-1]]

    sensor_x = x_grid[torch.newaxis, sensor_xind, :]
    sensor_x = sensor_x.repeat(Xs.shape[0], 1, 1)

    sensor_y = Xs[:, sensor_xind, :-1]
    Xics, _ = interpolate(x_grid, sensor_x, sensor_y, sample=sample, n_samples=1)

    sensor_y = ys[:, sensor_xind, ...]
    Xbcs, _ = interpolate(x_grid, sensor_x, sensor_y, sample=sample, n_samples=1)

    Xics, Xbcs, ys = (
        torch.from_numpy(Xics).float(),
        torch.from_numpy(Xbcs).float(),
        ys.float(),
    )

    train_dataset = TripleTensorDataset(Xics, Xbcs, ys)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["train"]["batchsize"], shuffle=True
    )

    config_ic = load_config(config["model"]["base_model_config"])
    ic_model = Prediction(
        modes1=config_ic["model"]["modes1"],
        fc_dim=config_ic["model"]["fc_dim"],
        layers=config_ic["model"]["layers"],
        activation=config_ic["model"]["activation"],
        output_activation=config_ic["model"]["output_activation"],
        input_codim=T_in,
        output_codim=T_out,
    )

    ic_model.load_state_dict(
        torch.load("models/" + config_ic["train"]["fname"], weights_only=True)
    )
    model = Correction(
        ic_model=ic_model,
        deltaT=deltaT,
        T_out=T_out,
        device=device,
        modes1=config["model"]["modes1"],
        modes2=config["model"]["modes2"],
        fc_dim=config["model"]["fc_dim"],
        layers=config["model"]["layers"],
        activation=config["model"]["activation"],
        output_activation=config["model"]["output_activation"],
    )

    model = train(
        model=model,
        config=config,
        train_loader=train_loader,
        device=device,
        operator="correction",
    )
    torch.save(model.state_dict(), "models/" + config["train"]["fname"])


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    run(config=config)
