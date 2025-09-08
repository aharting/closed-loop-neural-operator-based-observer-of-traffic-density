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
from tqdm import tqdm
import numpy as np
from imported.losses import LpLoss
import matplotlib.pyplot as plt


def mse(pred, y):
    return torch.mean(torch.square(pred - y))


def train(model, config, train_loader, device, operator="prediction"):
    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=config["train"]["base_lr"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["train"]["milestones"],
        gamma=config["train"]["scheduler_gamma"],
    )

    model.to(device)
    if operator == "correction":
        model.ic_model.to(device)
    model.train()
    myloss = LpLoss(
        d=1, p=2, reduce_dims=[0, 1], reductions=["mean", "sum"]
    )  # sum over output dimensions, mean over batch. Need to input y=[batch, output, x]
    pbar = range(config["train"]["epochs"])
    if config["train"]["use_tqdm"] == True:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    train_loss_saved = np.zeros(config["train"]["epochs"])

    for e in pbar:
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if operator == "prediction":
                # Prediction operator
                x, y = data
                x, y = x.to(device), y.to(device)
                out = model(x).reshape(y.shape)
            elif operator == "correction":
                # Correction operator
                x, z, y = data
                x, z, y = x.to(device), z.to(device), y.to(device)
                unflat_shape = x.shape[:2]
                x = x.flatten(start_dim=0, end_dim=1)
                pred = model.ic_model(x)
                pred = pred.unflatten(0, unflat_shape)
                out = model(pred, z).reshape(y.shape)
            else:
                raise ValueError("Undefined operator")
            data_loss = myloss(
                torch.transpose(out, 1, 2), torch.transpose(y, 1, 2)
            )  # see comment wrt myLoss, y has shape (batch, x, t)
            total_loss = data_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        if config["train"]["use_tqdm"] is True:
            pbar.set_description((f"Epoch {e}, train loss: {train_loss:.5f} "))
        train_loss_saved[e] = train_loss
    print("Done!")

    fig, ax = plt.subplots()
    fig.suptitle("Loss evolution")
    ax.plot(np.arange(config["train"]["epochs"]), train_loss_saved)
    ax.set_title("Train loss")
    dir = config["main_dir"] + config["test"]["save_dir"]
    fig.savefig(fname=f"{dir}/loss")
    return model
