#定义一个专门用来读取图片和标签分离的npy数据集的Dataset：
import torch
import numpy as np
from torchvision import  transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,data_filename,label_filename) :
        #将数据和标签分别进行读入,读到的是numpy数据
        self.datas=np.load(data_filename)
        self.labels=np.load(label_filename)
        super().__init__()

    def __getitem__(self, index):
        #先对datas和labels进行转换
        datas=self.data_transforms(self.datas)
        labels=self.label_transforms(self.labels)
        #取出对应位置的元素返回
        return datas[index],labels[index]

    def data_transforms(self,data):
        #NHWC转换为NCHW
        data=data.transpose((0,3,1,2))
        data=torch.from_numpy(data)
        #print(type(data))
        return data
    def label_transforms(self,label):
        label=torch.from_numpy(label)
        return label

    def __len__(self):
        return self.datas.shape[0]


#测试一下
train_dataset=IndiaPinesDataset("XtrainWindowSize15.npy","ytrainWindowSize15.npy")
# # imag,target=train_dataset[1]
# # print(type(imag))
# # print(imag.shape)
# # print(target)
train_loader=DataLoader(datasets=train_dataset,batch_size=100,shuffle=True,drop_last=True)
print(train_loader.__len__())
for data in train_loader:
    imgs,targets=data
    print(type(imgs[0][0][0][0]))
    print(imgs[0].shape)
    print(targets[0])