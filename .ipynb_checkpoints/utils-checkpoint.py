from tqdm import tqdm
import numpy as np
import glob

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time


class Metrics:
    def __init__():
        self.reset()
        
    def reset():    
        self.correct = 0
        self.count = 0
        self.count_pos = 0
        self.count_neg = 0
        self.true_pos = 0
        self.true_neg = 0

    def update(target, pred, test_loss):
        target_ = target.view_as(pred)
        self.test_loss += test_loss
        self.correct += pred.eq(target_).sum().item()
        self.true_pos += ((pred == 1) & (target_ == 1)).sum()
        self.count_pos += (target_ == 1).sum()
        self.true_neg += ((pred == 0) & (target_ == 0)).sum()
        self.count_neg += (target_ == 0).sum()
        self.count += len(data)
        
    def result():
        test_loss = self.test_loss / count
        tpr = self.true_pos / self.count_pos
        tnr = self.true_neg / self.count_neg
        acc = self.correct / self.count
        tpr_tnr_2 = (self.tpr + self.tnr) / 2
        return tpr, tnr, acc, tpr_tnr_2

    
# def test(model, test_dataloader, experiment=None, silent=False, max_samples=None, header_str="Test"):
#     model.eval()
#     metrics = Metrics()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     with torch.no_grad():
#         for data, _, target, _, _ in test_dataloader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             metrics.update(pred, target, test_loss)
#             if max_samples is not None and count > max_samples:
#                 break

#     test_loss, tpr, tnr, acc, tpr_tnr_2 = metrics.result()
#     if experiment is not None:
#         experiment.log_metric("test_loss", test_loss)
#         experiment.log_metric("test_acc", acc)
#         experiment.log_metric("test_tpr", tpr)
#         experiment.log_metric("test_tnr", tnr)
#         experiment.log_metric("test_tpr+tnr/2", tpr_tnr_2)

#     if not silent:
#         print(f'{header_str}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{count} ({100. * correct / count:.2f}%), ' + \
#               f'TPR: {100. * tpr:.2f}%, TNR: {100. * tnr:.2f}%, TNR+TPR/2: {100*((tpr+tnr)/2):.2f}%')
    
#     return tpr, tnr

    
    
    
    
def test(model, test_dataloader, experiment=None, silent=False, max_samples=None, header_str="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    count_pos = 0
    count_neg = 0
    true_pos = 0
    true_neg = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for single_sample in test_dataloader:
            if len(single_sample) ==5:
                data, _, target, _, _ = sample
                data, target = data.to(device), target.to(device)
            else:
                data, target = single_sample[0].to(device), single_sample[1].to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target_ = target.view_as(pred)
            correct += pred.eq(target_).sum().item()
            true_pos += ((pred == 1) & (target_ == 1)).sum()
            count_pos += (target_ == 1).sum()
            true_neg += ((pred == 0) & (target_ == 0)).sum()
            count_neg += (target_ == 0).sum()
            count += len(data)
            if max_samples is not None and count > max_samples:
                break

    test_loss /= count
    tpr = true_pos / count_pos
    tnr = true_neg / count_neg
    acc = correct / count
    tpr_tnr_2 = (tpr + tnr) / 2
    if experiment is not None:
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("test_acc", acc)
        experiment.log_metric("test_tpr", tpr)
        experiment.log_metric("test_tnr", tnr)
        experiment.log_metric("test_tpr+tnr/2", tpr_tnr_2)

    if not silent:
        print(f'{header_str}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{count} ({100. * correct / count:.2f}%), ' + \
              f'TPR: {100. * tpr:.2f}%, TNR: {100. * tnr:.2f}%, TNR+TPR/2: {100*((tpr+tnr)/2):.2f}%')
    
    return tpr, tnr


def train_epoch(model, train_dataloader, optimizer, max_samples=None, epoch=0, log_interval=100, silent=False):
    model.train()
    step = 1
    total_loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for single_sample in train_dataloader:
        if len(single_sample) ==5:
            data, _, target, sample_weight, _ = sample
            data, target, sample_weight = data.to(device), target.to(device), sample_weight.to(device)
        else:
            data, target = single_sample[0].to(device), single_sample[1].to(device)
            sample_weight = 1.
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = loss * sample_weight
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        if log_interval > 1 and step % log_interval == 0:
            _print_train(silent, epoch, step, batch_size=train_dataloader.batch_size, 
                         dataset_length=len(train_dataloader) if max_samples is None else max_samples,
                         loss=loss)

        step += 1
        total_loss += loss.item()
        if (max_samples is not None) and (step * train_dataloader.batch_size > max_samples):
            _print_train(silent, epoch, step, batch_size=train_dataloader.batch_size, 
                         dataset_length=len(train_dataloader) if max_samples is None else max_samples,
                         loss=loss)
            break

def _print_train(silent, epoch, step, batch_size, dataset_length, loss):
    if not silent:
        print(f'Train Epoch: {epoch} [{step * batch_size}/{dataset_length} ' + \
              f'({100. * step * batch_size / dataset_length:.0f}%)]\tLoss: {loss.item():.6f}')
        
        

class BBandFChead(nn.Module):
    def __init__(self, bb_model, train_bb=False, hidden_dim=512, output_dim=2):
        super().__init__()
        self.train_bb = train_bb
        self.bb_model = bb_model
        self.flatten = nn.Flatten()
        if hidden_dim > 0:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(2048, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.linear_relu_stack = nn.Linear(2048, output_dim)

    def forward(self, x):
        if not self.train_bb:
            with torch.no_grad():
                x = self.bb_model(x)
        else:
            x = self.bb_model(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_model(model, train_dataset, test_dataset, epochs=5, samples_per_epoch=10_000, log_interval=100,
                batch_size=64, lr=0.0001):
    num_workers = 1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size * num_workers, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * num_workers, shuffle=False, num_workers=num_workers)

    for e in range(epochs):
        start = time.time()
        train_epoch(model, train_dataloader, optimizer=optimizer, 
                    max_samples=samples_per_epoch, log_interval=log_interval, epoch=e, silent=True)
        test(model, test_dataloader, max_samples=1000, header_str=f'Step {(e+1)*samples_per_epoch}: Test ({time.time() - start:.1f}s)')

        
def load_momo_dataset(data_path="/root/tal/repos/data-mining-research/data_mining_research/momo/data/"):
    
    from src.data_utils import get_momo_data
    from src.sample_policy import get_sampler
    from src.sample_policy import RandomSampler


    train_dataset = get_momo_data(train=True, data_path=data_path, load_clip=False, load_images=True, small=False)
    test_dataset = get_momo_data(train=False, data_path=data_path, load_clip=False, load_images=True)
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset