import torch
import copy
import numpy as np
from imported.models import FNN1d, FNN2d


class Prediction(FNN1d):
    pass


class Correction(FNN2d):
    def __init__(
        self,
        ic_model,
        modes1,
        modes2,
        deltaT,
        T_out,
        device,
        width=64,
        fc_dim=128,
        layers=None,
        activation="gelu",
        output_activation=None,
    ):
        super(Correction, self).__init__(
            modes1=modes1,
            modes2=modes2,
            width=width,
            fc_dim=fc_dim,
            layers=layers,
            in_codim=2,
            out_codim=1,
            activation=activation,
            output_activation=output_activation,
        )
        self.ic_model = ic_model
        # Freeze IC model
        for param in self.ic_model.parameters():
            param.requires_grad = False
        self.gridt = np.linspace(0, deltaT * (T_out - 1), num=T_out, dtype=np.float32)
        self.device = device

    def forward(self, x1, x2):
        unflat_shape = x1.shape[:2]
        x1 = x1.flatten(start_dim=0, end_dim=1)
        x2 = x2.flatten(start_dim=0, end_dim=1)
        e = x1 - x2[..., :-1]  # Luenberger-like observer
        x = torch.cat((x1.unsqueeze(-1), e.unsqueeze(-1)), dim=-1)
        # Create grid
        gridx = copy.deepcopy(x2[0, ..., -1]).to(self.device)
        gridt = torch.tensor(self.gridt).to(self.device)
        nb = x.shape[0]
        nx = len(gridx)
        nt = len(gridt)
        grid = gridx.expand(nb, nt, nx).permute(0, 2, 1).unsqueeze(-1)
        grid = torch.cat((grid, gridt.expand(nb, nx, nt).unsqueeze(-1)), dim=-1)
        x = torch.cat((x, grid), dim=-1)
        x = super().forward(x)
        x = torch.unflatten(x, dim=0, sizes=unflat_shape)
        x = torch.mean(x, dim=1)  # mean over samples
        return x.squeeze(-1)
