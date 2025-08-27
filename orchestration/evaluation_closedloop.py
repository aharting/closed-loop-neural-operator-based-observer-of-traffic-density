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
from modules.fourier import FNN1d, Correction
from modules.data import gen_data_test, load_config
from modules.evaluation import inspect_observers

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
                    help="config", 
                    required=True)
parser.add_argument("--max_fcst", 
                    type=int, 
                    help="max_fcst", 
                    required=False,
                    default=np.inf)
parser.add_argument("--gp_error", 
                    type=str2bool, 
                    help="", 
                    required=False,
                    default=False)
def run(config, max_fcst=np.inf, gp_error=True):
    if config['train']['device'] is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = config['train']['device']
    print('Using device:', device)

    T_in = config['data']['T_in']
    T_out = config['data']['T_out']
    max_dset = config['data']['max_dset']
    std_y = config['data']['std_y']
    
    test_data_dir = config["data"]["test_data_dir"]
    Xss, yss, deltaT, deltaX = gen_data_test(test_data_dir, T_in=T_in, T_out=T_out, max_dset=max_dset)
    test_dataset = torch.utils.data.TensorDataset(Xss, yss)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)  
    loader = test_loader

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
    model = Correction(ic_model=ic_model,
                        deltaT=deltaT,
                        T_out=T_out,
                        device=device,
                        modes1=config['model']['modes1'],
                        modes2=config['model']['modes2'],
                        fc_dim=config['model']['fc_dim'],
                        layers=config['model']['layers'],
                        activation=config['model']['activation'],
                        output_activation=config['model']['output_activation']).to(device)
    model.load_state_dict(torch.load(config['train']['save_path'], weights_only=True))
    
    inspect_observers(model=model,
                  loader=loader, 
                  config=config, 
                  device=device, 
                  deltaX=deltaX, 
                  deltaT=deltaT, 
                  T_in=T_in, 
                  T_out=T_out, 
                  id='test', 
                  max_fcst=max_fcst, 
                  gp_error=gp_error, 
                  std_y=std_y)

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    max_fcst = args.max_fcst
    gp_error = args.gp_error
    run(config=config, max_fcst=max_fcst, gp_error=gp_error)

