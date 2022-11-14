from torchvision.datasets import Caltech101
from torchvision import transforms
import torch 


def load_caltech101(data_path="/root/data/", train_fruction=0.8, train_dataset_len: int=None):
    """
    train_dataset_len: is passed will ignore "train_fruction", and use only "train_dataset_len" samples
    """
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
        transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    dataset = Caltech101(data_path, transform=data_transforms)
    if train_dataset_len is not None:
        train_size = train_dataset_len
    else:
        train_size = int(train_fruction*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset