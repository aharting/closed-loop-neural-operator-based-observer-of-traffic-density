"""
Imported from https://github.com/Plasma-FNO/FNO_Isothermal_Blob/blob/main/FNO.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)

def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=-1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)

        return x
    
class FNN1d(nn.Module):
    def __init__(self, modes1, 
                 width=64, fc_dim=128,
                 layers=None, input_codim=1, output_codim=1,
                 activation='gelu',
                 output_activation=None
                ):
        super(FNN1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: evaluation + 1 locations (u_1(x), ..., u_N(x), x) (for example, u_1=u(t1, x), u2=u(t2, x), ...)
        input shape: (batchsize, x=x_discretisation, c=in_codim + 1)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, c=out_codim)
        """

        self.modes1 = modes1
        self.width = width
        self.in_dim = input_codim  + 1
        self.out_dim = output_codim

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(input_codim + 1, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, mode1_num)
            for in_size, out_size, mode1_num
            in zip(self.layers, self.layers[1:], self.modes1)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.out_dim)
        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

        if output_activation =='tanh':
            self.output_activation = F.tanh
        elif output_activation == 'gelu':
            self.output_activation = F.gelu
        elif output_activation == 'relu':
            self.output_activation == F.relu
        elif output_activation == 'sigmoid':
            self.output_activation = F.sigmoid
        elif output_activation is None:
            self.output_activation = None
        else:
            raise ValueError(f'{output_activation} is not supported')

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, input_codim + 1)
        Returns:
            - x: (batch size, x_grid, output_codim)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx = x.shape[1] # original shape
        size_x = x.shape[1]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            # x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = x.reshape(batchsize, size_x, self.out_dim)
        x = x[:, :nx, :]
        return x
    
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                            dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None, in_codim=1, out_codim=1,
                 activation='gelu',
                 output_activation=None):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: evaluation + 2 locations (u_1(x, y), ..., u_N(x, y),  x, y)
        input shape: (batchsize, x=x_discretisation, y=y_discretisation, c=in_codim + 2)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisation, c=out_codim)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_codim + 2
        self.out_dim = out_codim

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
                                       for in_size, out_size, mode1_num, mode2_num
                                       in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv2d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.out_dim)
        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

        if output_activation =='tanh':
            self.output_activation = F.tanh
        elif output_activation == 'gelu':
            self.output_activation = F.gelu
        elif output_activation == 'relu':
            self.output_activation == F.relu
        elif output_activation == 'sigmoid':
            self.output_activation = F.sigmoid
        elif output_activation == 'softplus':
            self.output_activation = F.softplus            
        elif output_activation is None:
            self.output_activation = None
        else:
            raise ValueError(f'{output_activation} is not supported')

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, in_codim + 2)
        Returns:
            - x: (batch size, x_grid, y_grid, out_codim)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx = x.shape[1] # original shape
        ny = x.shape[2]
        size_x = nx
        size_y = ny
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            # x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = x.reshape(batchsize, size_x, size_y, self.out_dim)
        x = x[:, :nx, :]
        return x