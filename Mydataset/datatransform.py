'''
dataset一开始初始化的时候是看不见位于其内部的方法的，所以要将数据处理的函数单独拎出来
'''
import torch
import numpy as np

def data_transforms(data):
    # data必须转为float形式，要不然进不了模型
    data = torch.from_numpy(data)
    # print(type(data))
    return data
