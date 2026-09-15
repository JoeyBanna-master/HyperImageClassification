#定义一个专门用来读取图片和标签分离的npy数据集的Dataset：
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
'''
说白了，之前的dataloader是在读取数据的时候再进行数据的处理的，
而改进就是在dataset初始化的时候就将数据全部处理好,以此达到更快的dataloder读取速度，
速度是之前的几十倍
'''

def data_transforms(data):
    # data必须转为float形式，要不然进不了模型
    data = torch.from_numpy(data)
    # print(type(data))
    return data

#现在不管怎么样先试一下
class MyDataset(Dataset):
    def __init__(self,data_filename,label_filename,pre_transform: bool = True,to_cuda: bool = False) :
        #将数据和标签分别进行读入,读到的是numpy数据
        self.datas = np.load(data_filename)
        # print("Loaded data shape:", self.datas.shape)
        self.labels = np.load(label_filename)
        self.pre_transform = pre_transform
        super().__init__()
        if pre_transform is not False:
            self.datas = self.datas.astype("float32")
            self.labels = self.labels.astype("int64")
            for index in range(len(self)):
                self.datas[index] = data_transforms(self.datas[index])
                self.labels[index] = torch.Tensor([self.labels[index]]).squeeze_()
                if to_cuda:
                    self.labels[index] = self.labels[index].cuda()
            if to_cuda:
                self.datas = self.datas.cuda()
        # NHWC转换为NCHW
        #最新的数据处理代码已经转过了，就不用再转了
        #self.datas = self.datas.transpose((0, 3, 1, 2))

    def __getitem__(self, index):
        #取出对应位置的元素返回
        return self.datas[index],self.labels[index]

    def __len__(self):
        return self.datas.shape[0]


#测试一下
# train_dataset=MyDataset(r"E:\HSILearning\Refrain\datasets\IP\1%XtestWindowSize11.npy",r"E:\HSILearning\Refrain\datasets\IP\1%ytestWindowSize11.npy")
# imag,target=train_dataset[1]
# print(type(imag))
# print(imag.shape)
# print(target)
# train_loader=DataLoader(dataset=train_dataset,batch_size=100,shuffle=True,drop_last=True)
# print(train_loader.__len__())
# for data in train_loader:
#     imgs,targets=data
#     print(type(imgs[0][0][0][0]))
#     print(imgs[0].shape)
#     print(targets[0])