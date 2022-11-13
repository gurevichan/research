from torchsummary import summary
import torch
from utils import BBandFChead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_vicregl_resnet50 = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p9')
model_vicregl_resnet50.name = "ResNet-50"

model_vicregl_large = torch.hub.load('facebookresearch/vicregl:main', 'convnext_xlarge_alpha0p75')
model_vicregl_large.name = "ConvNeXt-XL"


import sys
sys.path.append("/mobileye/ALGO_VAST/mobileye-team-angie/andreyg/data-mining-research/data_mining_research/momo/")

import torch.optim as optim
from torch.utils.data import DataLoader
from utils import train_model

from src.data_utils import get_momo_data
from src.sample_policy import get_sampler
from src.sample_policy import RandomSampler


data_path = "/root/tal/repos/data-mining-research/data_mining_research/momo/data/"
train_dataset = get_momo_data(train=True, data_path=data_path, load_clip=False, load_images=True, small=False)
test_dataset = get_momo_data(train=False, data_path=data_path, load_clip=False, load_images=True)
input_dim = train_dataset[0][0].shape
output_dim = 2
print(len(train_dataset), len(test_dataset))



with open('logs/log.txt', 'w') as sys.stdout:
    
    for train_bb in [False, True]:
        for model in [model_vicregl_large, model_vicregl_resnet50]:
            for hidden_dim in [128, 256, 512, 1024]:
                for lr in [0.001, 0.0001]:
                    curr_model = BBandFChead(model, hidden_dim=hidden_dim, train_bb=train_bb)
                    print(f'{model.name}: train_bb {train_bb}, hidden_dim {hidden_dim}, lr={lr}')
                    train_model(curr_model, train_dataset=train_dataset, test_dataset=test_dataset, lr=lr, epochs=1)