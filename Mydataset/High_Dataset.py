"""Load separate npy samples/labels, converting the whole array once."""
import numpy as np
import torch
from torch.utils.data import Dataset


def data_transforms(data):
    return torch.from_numpy(data)


class MyDataset(Dataset):
    def __init__(self, data_filename, label_filename, pre_transform=True, to_cuda=False):
        self.datas = np.load(data_filename)
        self.labels = np.load(label_filename)
        if len(self.datas) != len(self.labels):
            raise ValueError('Sample and label counts differ')
        self.pre_transform = pre_transform
        if pre_transform:
            self.datas = torch.from_numpy(self.datas.astype(np.float32, copy=False))
            self.labels = torch.from_numpy(self.labels.astype(np.int64, copy=False))
            if to_cuda:
                self.datas = self.datas.cuda()
                self.labels = self.labels.cuda()

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return self.datas.shape[0]
