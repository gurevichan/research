import torch
from training_utils import train_model, test
from models import BBandFChead, SideNet, SideNetGroups
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from torch.utils.data import DataLoader
from caltech101 import load_caltech101
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import argparse
import os
import sys


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nt', '--net_type', nargs="+", help="resnet, side_net, side_net_group, all")
    parser.add_argument('-ls', '--limb_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('--data_len', type=int, default=None)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--debug', action="store_true")
    
    args = parser.parse_args()
    if args.net_type == ['all']:
        args.net_type = ["resnet", "side_net", "side_net_group"]
    print(args.net_type)
    return args

    
if __name__ == "__main__":
    args = parse()
    output_dir = "logs/finetune"
    num_files_in_out_dir = len(os.listdir(output_dir))
    out_path = os.path.join(output_dir, f"log_{str(num_files_in_out_dir).zfill(4)}.log")
    print(f'printing to {out_path}')
    num_samples = 10 if args.debug else None
    with open(out_path, 'w') as sys.stdout:
        print(args)
        train_dataset, test_dataset = load_caltech101(data_path="/home/me.docker/work/data", train_dataset_len=args.data_len)
        for net in args.net_type:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            resnet.fc = nn.Identity()
            if net == "resnet":
                model = BBandFChead(resnet, hidden_dim=0, output_dim=101, train_bb=False) 
                # creating a model with frozen bb and only a linear eval head.
            elif net == "side_net":
                model = SideNet(resnet, num_classes=101, limb_size=args.limb_size).to(device)
            elif net == "side_net_group":
                model = SideNetGroups(resnet, num_classes=101, limb_size=args.limb_size).to(device)
            print(f"*****Tuning {net}*****\n"*3)
            print(f'Total params: {sum(p.numel() for p in model.parameters()):,d}')
            print(model)
            _ = train_model(model, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=args.batch_size,
                            lr=args.learning_rate, epochs=args.epochs, 
                            samples_per_epoch=num_samples, test_max_samples=num_samples, 
                            silent=False, plot=False)

            print('done')