import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass
    
from modules.data import gen_data_test
import argparse
import torch
import numpy as np
from modules.fourier import FNN1d, FNN2d
from modules.data import load_config
from modules.evaluation import inspect_ol

parser = argparse.ArgumentParser()
parser.add_argument("--config", 
                    type=str, 
                    help="Base operator config file path", 
                    required=True)
parser.add_argument("--max_unroll", 
                    type=int, 
                    help="max unroll", 
                    required=False,
                    default=np.inf)
def run(config, max_unroll=np.inf):
    if config['train']['device'] is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = config['train']['device']
    print('Using device:', device)

    T_in = config['data']['T_in']
    T_out = config['data']['T_out']
    max_dset = config['data']['max_dset']
    only_first = config['data']['only_first']

    test_data_dir = config["data"]["test_data_dir"]
    Xss, yss, deltaT, deltaX = gen_data_test(test_data_dir, T_in=T_in, T_out=T_out, max_dset=max_dset)

    test_dataset = torch.utils.data.TensorDataset(Xss, yss)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)  
    loader = test_loader

    model = FNN1d(modes1=config['model']['modes1'],
                    fc_dim=config['model']['fc_dim'],
                    layers=config['model']['layers'],
                    activation=config['model']['activation'],
                    output_activation=config['model']['output_activation'],
                    input_codim=T_in,
                    output_codim=T_out
                    ).to(device)
    model.load_state_dict(torch.load(config['train']['save_path'], weights_only=True, map_location=device))
    model.eval()

    inspect_ol(model=model, loader=loader, config=config, device=device, deltaX=deltaX, deltaT=deltaT, T_in=T_in, T_out=T_out, id='test', max_unroll=max_unroll)

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    max_unroll = args.max_unroll
    run(config=config, max_unroll=max_unroll)

