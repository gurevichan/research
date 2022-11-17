from tqdm import tqdm
import numpy as np
import glob

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time


ZFILL_LENGTH = 8


class Metrics:
    def __init__(self):
        self.reset()
        
    def reset(self):    
        self.correct = 0
        self.count = 0
        self.count_pos = 0
        self.count_neg = 0
        self.true_pos = 0
        self.true_neg = 0
        self.test_loss = 0

    def update(self, target, pred, test_loss, count):
        target_ = target.view_as(pred)
        self.test_loss += test_loss
        self.correct += pred.eq(target_).sum().item()
        self.true_pos += ((pred == 1) & (target_ == 1)).sum()
        self.count_pos += (target_ == 1).sum()
        self.true_neg += ((pred == 0) & (target_ == 0)).sum()
        self.count_neg += (target_ == 0).sum()
        self.count += count
        
    def result(self):
        test_loss = self.test_loss / self.count
        tpr = self.true_pos / self.count_pos
        tnr = self.true_neg / self.count_neg
        acc = self.correct / self.count
        tpr_tnr_2 = (tpr + tnr) / 2
        return test_loss, tpr, tnr, acc, tpr_tnr_2

    def print(self, header_str='', tnrs=False):
        test_loss, tpr, tnr, acc, tpr_tnr_2 = self.result()
        msg = f'{header_str.ljust(40)}: loss: {test_loss:.4f}, Accuracy: {self.correct}/{self.count} ({100. * acc:.2f}%)'
        if tnrs:
            msg += f'TPR: {100. * tpr:.2f}%, TNR: {100. * tnr:.2f}%, TNR+TPR/2: {100*((tpr+tnr)/2):.2f}%'
        print(msg, end='\r')
    

    
def test(model, test_dataloader, experiment=None, silent=False, max_samples=None, header_str="Test"):
    model.eval()
    metrics = Metrics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for single_sample in test_dataloader:
            data, target, _ = get_imgs_and_labels(single_sample, device)
            output = model(data)
            curr_test_loss = F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            metrics.update(pred, target, curr_test_loss, len(data))
            if max_samples is not None and metrics.count > max_samples:
                break

    test_loss, tpr, tnr, acc, tpr_tnr_2 = metrics.result()
    if experiment is not None:
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("test_acc", acc)
        experiment.log_metric("test_tpr", tpr)
        experiment.log_metric("test_tnr", tnr)
        experiment.log_metric("test_tpr+tnr/2", tpr_tnr_2)

    if not silent:
        metrics.print(header_str)
    
    return test_loss, acc, tpr, tnr

def get_imgs_and_labels(single_sample, device):
    """
    Handles momo data and caltech101 datasets
    """
    if len(single_sample) ==5:
        # this is for momo data
        # imgs, clip_representations, labels, sample_weight, ???
        imgs, _, labels, sample_weight, _ = sample
        imgs, labels, sample_weight = imgs.to(device), labels.to(device), sample_weight.to(device)
    else:
        # this is for caltech101 data
        # imgs, labels
        imgs, labels = single_sample[0].to(device), single_sample[1].to(device)
        sample_weight = 1.
    return imgs, labels, sample_weight
    

def train_epoch(model, train_dataloader, optimizer, max_samples=None, epoch=0, log_interval=100, silent=False):
    model.train()
    step = 1
    metrics = Metrics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    samples_per_train_epoch = max_samples if max_samples is not None else len(train_dataloader)
    for single_sample in train_dataloader:
        data, target, sample_weight = get_imgs_and_labels(single_sample, device)
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = loss * sample_weight
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        # if log_interval > 1 and step % log_interval == 0:
        #     _print_train(silent, epoch, step, batch_size=train_dataloader.batch_size, 
        #                  dataset_length=len(train_dataloader) if max_samples is None else max_samples,
        #                  loss=loss)

        step += 1
        pred = output.argmax(dim=1, keepdim=True)
        metrics.update(pred, target, loss, len(data))
        if (max_samples is not None) and (step * train_dataloader.batch_size > max_samples):
            # _print_train(silent, epoch, step, batch_size=train_dataloader.batch_size, 
            #              dataset_length=len(train_dataloader) if max_samples is None else max_samples,
            #              loss=loss)
            break
    if not silent:
        msg = f'Train; Epoch {epoch+1}, Step ' \
              f'{str((epoch+1) * (samples_per_train_epoch - 1) * train_dataloader.batch_size + len(data)).zfill(ZFILL_LENGTH)}'
        metrics.print(msg)
    train_loss, tpr, tnr, acc, tpr_tnr_2 = metrics.result()
    return train_loss.cpu().detach().numpy(), acc, tpr, tnr

    
def _print_train(silent, epoch, step, batch_size, dataset_length, loss):
    if not silent:
        print(f'Train Epoch: {epoch} [{step * batch_size}/{dataset_length} ' + \
              f'({100. * step * batch_size / dataset_length:.0f}%)]\tLoss: {loss.item():.6f}')
        

def train_model(model, train_dataset, test_dataset, epochs=5, samples_per_epoch=None, test_max_samples=None, log_interval=100,
                batch_size=64, lr=0.0001, silent=False, plot=True):
    num_workers = 1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size * num_workers, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * num_workers, shuffle=False, num_workers=num_workers)
    samples_per_train_epoch = samples_per_epoch if samples_per_epoch is not None else len(train_dataset)
    
    max_acc = 0
    best_epoch = -1
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for e in range(epochs):
        start = time.time()
        train_loss, train_acc, _, _ = train_epoch(model, train_dataloader, optimizer=optimizer, 
                                                  max_samples=samples_per_epoch, log_interval=log_interval, epoch=e, silent=silent)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss, test_acc, _, _ = test(model, test_dataloader, max_samples=test_max_samples, 
                         header_str=f'Test;  Epoch {e+1}, Step {str((e+1)*samples_per_train_epoch).zfill(ZFILL_LENGTH)} ' \
                         f'({time.time() - start:.1f}s)', silent=silent)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if max_acc < test_acc:
            max_acc = test_acc
            best_epoch = e
    print(f'\nBest accuracy of {100 * max_acc:.2f} achieved at epoch {best_epoch}')
    if plot:
        plot_results(train_loss_list, test_loss_list, train_acc_list, test_acc_list)
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list

        
def load_momo_dataset(data_path="/root/tal/repos/data-mining-research/data_mining_research/momo/data/"):
    
    from src.data_utils import get_momo_data
    from src.sample_policy import get_sampler
    from src.sample_policy import RandomSampler


    train_dataset = get_momo_data(train=True, data_path=data_path, load_clip=False, load_images=True, small=False)
    test_dataset = get_momo_data(train=False, data_path=data_path, load_clip=False, load_images=True)
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset


def plot_results(train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    from matplotlib import pyplot as plt

    f, ax = plt.subplots(2)
    ax[0].plot(range(len(train_loss_list)), train_loss_list)
    ax[0].plot(range(len(train_loss_list)), test_loss_list)
    ax[0].legend(['train loss', "test loss"])
    ax[0].grid()

    ax[1].plot(range(len(train_acc_list)), train_acc_list)
    ax[1].plot(range(len(train_loss_list)), test_acc_list)
    ax[1].legend(['train acc', "test acc"])
    ax[1].grid()