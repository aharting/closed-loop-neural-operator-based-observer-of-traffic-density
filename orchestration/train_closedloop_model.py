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
from modules.fourier import FNN1d, Sequential, Dummy
from modules.data import gen_data_train, load_config, gpr_ics, gpr_bcs
from modules.pdeno import train
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--config", 
                    type=str, 
                    help="Base operator config file path", 
                    required=False)

parser.add_argument("--load_Xs_ys", 
                    type=str2bool, 
                    help="", 
                    required=False,
                    default=False)
        
def run(config, load_Xs_ys):
    if config['train']['device'] is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = config['train']['device']
    print('Using device:', device)
    
    train_data_dir = config['data']['train_data_dir']
    T_in = config['data']['T_in']
    T_out = config['data']['T_out']
    max_dset = config['data']['max_dset']
    only_first = config['data']['only_first']
    Xs_ys_path = config['data']['Xs_ys_path']
    N_sensors=config['data']['N_sensors']
    sample=config['data']['sample']
    n_samples=config['data']['n_samples']
    std_y=config['data']['std_y'] # measurement noise - Gaussian variance
    train_noisy=config['data']['train_noisy'] # train on noisy data
    print("Train on noisy data:", train_noisy)
    if load_Xs_ys:
        Xs = torch.load(f"{Xs_ys_path}/{config['data']['name']}_Nsensors_{N_sensors}_nsamples_{n_samples if sample else '0'}_std_y_{str(std_y).replace('.', '') if train_noisy else '0'}_Xs.pt")
        ys = torch.load(f"{Xs_ys_path}/{config['data']['name']}_Nsensors_{N_sensors}_nsamples_{n_samples if sample else '0'}_std_y_{str(std_y).replace('.', '') if train_noisy else '0'}_ys.pt")
    else:
        _Xics, ys, deltaT, deltaX = gen_data_train(train_data_dir, T_in=T_in, T_out=T_out, max_dset=max_dset, only_first=only_first)
        if train_noisy:
            _Xics = torch.concatenate((torch.clamp(_Xics[..., :-1] + std_y*torch.randn_like(_Xics[..., :-1]), 0, 1),_Xics[..., [-1]]), axis=-1)
            ys = torch.clamp(ys + std_y*torch.randn_like(ys), 0, 1)
        Xics, _ = gpr_ics(_Xics, N_sensors, sample=sample, n_samples=n_samples, disable=False)
        Xbcs, _  = gpr_bcs(_Xics, ys, N_sensors, sample=sample, n_samples=n_samples, disable=False)
        Xs = torch.concatenate((torch.Tensor(Xics), torch.Tensor(Xbcs)), axis=-1)
        torch.save(Xs, f"{Xs_ys_path}/{config['data']['name']}_Nsensors_{N_sensors}_nsamples_{n_samples if sample else '0'}_std_y_{str(std_y).replace('.', '') if train_noisy else '0'}_Xs.pt")
        torch.save(ys, f"{Xs_ys_path}/{config['data']['name']}_Nsensors_{N_sensors}_nsamples_{n_samples if sample else '0'}_std_y_{str(std_y).replace('.', '') if train_noisy else '0'}_ys.pt")
    train_dataset = torch.utils.data.TensorDataset(Xs, ys)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)

    config_ic = load_config(config['model']['base_model_config'])
    ic_model = FNN1d(modes1=config_ic['model']['modes1'],
                     fc_dim=config_ic['model']['fc_dim'],
                     layers=config_ic['model']['layers'],
                     activation=config_ic['model']['activation'],
                     output_activation=config_ic['model']['output_activation'],
                     input_codim=T_in,
                     output_codim=T_out
                     ).to(device)
    
    ic_model.load_state_dict(torch.load(config_ic['train']['save_path'], weights_only=True))
    model = Sequential(ic_model=ic_model,
                       deltaT=deltaT,
                       T_out=T_out,
                       device=device,
                       modes1=config['model']['modes1'],
                       modes2=config['model']['modes2'],
                       fc_dim=config['model']['fc_dim'],
                       layers=config['model']['layers'],
                       activation=config['model']['activation'],
                       output_activation=config['model']['output_activation']).to(device)
    
    model = train(model=model, config=config, train_loader=train_loader, device=device)
    torch.save(model.state_dict(), config['main_dir'] + config['train']['save_path'])

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    load_Xs_ys = args.load_Xs_ys
    run(config=config, load_Xs_ys=load_Xs_ys)

