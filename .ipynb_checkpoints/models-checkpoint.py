import torch.nn.functional as F
from torch import nn
import torch
from torchvision.models import ResNet

CHANNELS_DIM = 1


class BBandFChead(nn.Module):
    def __init__(self, bb_model, train_bb=False, bb_output_dim=2048, hidden_dim=512, output_dim=2):
        super().__init__()
        self.train_bb = train_bb
        self.bb_model = bb_model
        self.bb_model.fc = nn.Identity()
        self.flatten = nn.Flatten()
        if hidden_dim > 0:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(bb_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.linear_relu_stack = nn.Linear(bb_output_dim, output_dim)

    def forward(self, x):
        if not self.train_bb:
            with torch.no_grad():
                x = self.bb_model(x)
        else:
            x = self.bb_model(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    
class SideNet(nn.Module):
    def __init__(self, bb_model, train_bb=False, limb_size=32, num_classes=101, verbose=False, 
                 constant_block_size=False):
        super().__init__()
        self.train_bb = train_bb
        self.bb_model = bb_model
        self.bb_model.fc = nn.Identity()
        self.verbose = verbose
        resnet_blocks_output_shape = {"l1": 256, "l2": 512, "l3": 1024, "l4": 2048}
        mul = 2
        if constant_block_size:
            my_blocks_output_shape = {"l1": limb_size,
                                      "l2": limb_size,
                                      "l3": limb_size,
                                      "l4": limb_size}
        else:
            my_blocks_output_shape = {"l1": limb_size,
                          "l2": int(limb_size * mul),
                          "l3": int(limb_size * 2 * mul),
                          "l4": int(limb_size * 4 * mul)}
        my_blocks_input_shape = {"l1": resnet_blocks_output_shape['l1'], 
                                 "l2": resnet_blocks_output_shape["l2"] + my_blocks_output_shape["l1"], 
                                 "l3": resnet_blocks_output_shape["l3"] + my_blocks_output_shape["l2"], 
                                 "l4": resnet_blocks_output_shape["l4"] + my_blocks_output_shape["l3"]}
        self.l1_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l1"], 
                                          out_channels=my_blocks_output_shape['l1'],
                                          kernel=3, stride=2, padding=1)
        self.l2_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l2"], 
                                          out_channels=my_blocks_output_shape['l2'],
                                          kernel=3, stride=2, padding=1)
        self.l3_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l3"], 
                                          out_channels=my_blocks_output_shape['l3'],
                                          kernel=3, stride=2, padding=1)
        self.l4_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l4"], 
                                          out_channels=my_blocks_output_shape['l4'],
                                          kernel=3, stride=2, padding=1)
        self.fc = nn.Linear(2048 + my_blocks_output_shape["l4"], num_classes)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.bb_model.conv1(x)
            x = self.bb_model.bn1(x)
            x = self.bb_model.relu(x)
            x = self.bb_model.maxpool(x)

            l1 = x = self.bb_model.layer1(x)
            l2 = x = self.bb_model.layer2(x)
            l3 = x = self.bb_model.layer3(x)
            l4 = x = self.bb_model.layer4(x)

            x = self.bb_model.avgpool(x)
            x_bb = torch.flatten(x, 1)
        
        x = self.l1_layers(l1)
        if self.verbose:
            print(f'l1: in={l1.shape}, my={x.shape}, next={l2.shape}') 
        x = torch.concat((x, l2), dim=CHANNELS_DIM)
        if self.verbose:
            print(f'concat shape {x.shape}')
        x = self.l2_layers(x)
        if self.verbose:
            print(f'l2: in={l2.shape}, my={x.shape}, next={l3.shape}') 
        x = torch.concat((x, l3), dim=CHANNELS_DIM)
        x = self.l3_layers(x)
        if self.verbose:
            print(f'l3: in={l3.shape}, my={x.shape}, next={l4.shape}') 
        x = torch.concat((x, l4), dim=CHANNELS_DIM)
        x = self.l4_layers(x)
        x = self.bb_model.avgpool(x)
        x_finetuning = torch.flatten(x, 1)
        if self.verbose:
            print(f'final before concat: x_bb={x_bb.shape}, x_finetuning={x_finetuning.shape}') 
        x = torch.concat((x_bb, x_finetuning), dim=CHANNELS_DIM)
        x = self.fc(x)
        if self.verbose:
            print(f'final: {x.shape}')
        return x
    
    
class SideNetGroups(nn.Module):
    def __init__(self, bb_model, train_bb=False, limb_size=32, num_classes=101, verbose=False,
                 constant_block_size=False):
        super().__init__()
        self.train_bb = train_bb
        self.bb_model = bb_model
        self.bb_model.fc = nn.Identity()
        self.verbose = verbose
        resnet_blocks_output_shape = {"l1": 256, "l2": 512, "l3": 1024, "l4": 2048}
        mul = 2

        if constant_block_size:
            my_blocks_output_shape = {"l1": limb_size,
                                      "l2": limb_size,
                                      "l3": limb_size,
                                      "l4": limb_size}
        else:
            my_blocks_output_shape = {"l1": limb_size,
                          "l2": int(limb_size * mul),
                          "l3": int(limb_size * 2 * mul),
                          "l4": int(limb_size * 4 * mul)}
                
        my_blocks_input_shape = {"l1": resnet_blocks_output_shape['l1'], 
                                 "l2": 2 * my_blocks_output_shape["l1"], 
                                 "l3": 2 * my_blocks_output_shape["l2"], 
                                 "l4": 2 * my_blocks_output_shape["l3"]}
        
        
        
        self.l2_conv = nn.Conv2d(in_channels=resnet_blocks_output_shape['l2'], 
                                 out_channels=my_blocks_output_shape["l1"], kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.l3_conv = nn.Conv2d(in_channels=resnet_blocks_output_shape['l3'], 
                                 out_channels=my_blocks_output_shape["l2"], kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.l4_conv = nn.Conv2d(in_channels=resnet_blocks_output_shape['l4'], 
                                 out_channels=my_blocks_output_shape["l3"], kernel_size=1, stride=1, padding=0, groups=16, bias=False)
        self.l1_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l1"], 
                                          out_channels=my_blocks_output_shape['l1'],
                                          kernel=3, stride=2, padding=1)
        self.l2_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l2"], 
                                          out_channels=my_blocks_output_shape['l2'],
                                          kernel=3, stride=2, padding=1)
        self.l3_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l3"], 
                                          out_channels=my_blocks_output_shape['l3'],
                                          kernel=3, stride=2, padding=1)
        self.l4_layers = get_conv_bn_relu(in_channels=my_blocks_input_shape["l4"], 
                                          out_channels=my_blocks_output_shape['l4'],
                                          kernel=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048 + my_blocks_output_shape["l4"], num_classes)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.bb_model.conv1(x)
            x = self.bb_model.bn1(x)
            x = self.bb_model.relu(x)
            x = self.bb_model.maxpool(x)

            l1 = x = self.bb_model.layer1(x)
            l2 = x = self.bb_model.layer2(x)
            l3 = x = self.bb_model.layer3(x)
            l4 = x = self.bb_model.layer4(x)

            x = self.bb_model.avgpool(x)
            x_bb = torch.flatten(x, 1)
        
        x = self.l1_layers(l1)
        l2 = self.l2_conv(l2)
        if self.verbose:
            print(f'l1: in={l1.shape}, my={x.shape}, next={l2.shape}') 
        x = torch.concat((x, l2), dim=CHANNELS_DIM)
        if self.verbose:
            print(f'concat shape {x.shape}')
        x = self.l2_layers(x)
        l3 = self.l3_conv(l3)
        if self.verbose:
            print(f'l2: in={l2.shape}, my={x.shape}, next={l3.shape}') 
        x = torch.concat((x, l3), dim=CHANNELS_DIM)
        x = self.l3_layers(x)
        l4 = self.l4_conv(l4)
        if self.verbose:
            print(f'l3: in={l3.shape}, my={x.shape}, next={l4.shape}') 
        x = torch.concat((x, l4), dim=CHANNELS_DIM)
        x = self.l4_layers(x)
        x = self.avgpool(x)
        x_finetuning = self.flatten(x)
        if self.verbose:
            print(f'final before concat: x_bb={x_bb.shape}, x_finetuning={x_finetuning.shape}') 
        x = torch.concat((x_bb, x_finetuning), dim=CHANNELS_DIM)
        x = self.fc(x)
        if self.verbose:
            print(f'final: {x.shape}')
        return x
    
    
def get_conv_bn_relu(in_channels, out_channels, stride=1, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )