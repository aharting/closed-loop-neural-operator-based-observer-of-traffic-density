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
from modules.models import Prediction
from modules.data import gen_data_train, load_config
from modules.train import train


parser = argparse.ArgumentParser()
parser.add_argument("--config", 
                    type=str,
                    default="configs/openloop.yaml", 
                    help="Prediction operator config file path", 
                    required=False)

def run(config):
    directory = Path("models")
    directory.mkdir(parents=True, exist_ok=True)

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

    Xs, ys, deltaT, deltaX = gen_data_train(train_data_dir, T_in=T_in, T_out=T_out, max_dset=max_dset, only_first=only_first)
    
    train_dataset = torch.utils.data.TensorDataset(Xs, ys)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    
    model = Prediction(modes1=config['model']['modes1'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  activation=config['model']['activation'],
                  output_activation=config['model']['output_activation'],
                  input_codim=T_in,
                  output_codim=T_out
                  ).to(device)
    
    model = train(model=model, config=config, train_loader=train_loader, device=device, operator="prediction")
    torch.save(model.state_dict(), "models/" + config['train']['fname'])

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    run(config=config)

