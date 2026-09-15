
#这个文件总和所有画图的代码，所有的featuremap在这里画了，并提供可调用的接口，想办法画成位图形式
import os
from scipy.io import loadmat
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from tifffile import imread



# 对单个像素周围提取 patch 时，边缘像素就无法取y了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

#注意，这里是重新提出pathces来进行预测
#要特别注意的是，这里的数据处理方式要和我们的数据处理代码要对的上
def draw_featuremap(device, dataset, model, windows, exp_name):










    # 每个像素周围提取 patch 的尺寸
    patch_size = windows
    X = None
    y = None
    if dataset == 'IP':
        X = loadmat('datasets/IP/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = loadmat('datasets/IP/indian_pines_gt.mat')['indian_pines_gt']
    elif dataset == 'HUST2013':
        X = loadmat('datasets/Houston/Houston.mat')['Houston']
        y = loadmat('datasets/Houston/Houston_gt.mat')['Houston_gt']
    elif dataset == 'Loukia':
        X = imread('datasets/Loukia/Loukia.tif')
        y = imread('datasets/Loukia/Loukia_GT.tif')
    elif dataset == 'BOS':
        X = loadmat('datasets/BOS/Botswana.mat')['Botswana']
        y = loadmat('datasets/BOS/Botswana_gt.mat')['Botswana_gt']
    elif dataset == 'PU':
        X = loadmat('datasets/PU/PaviaU.mat')['paviaU']
        y = loadmat('datasets/PU/PaviaU_gt.mat')['Data_gt']
    elif dataset == 'SV':
        X = loadmat('datasets/SV/salinas.mat')['HSI_original']
        y = loadmat('datasets/SV/Salinas_gt.mat')['Data_gt']
    elif dataset == 'KSC':
        X = loadmat('datasets/KSC/KSC.mat')['KSC']
        y = loadmat('datasets/KSC/KSC_gt.mat')['KSC_gt']
    # print(X.shape)
    height = y.shape[0]
    width = y.shape[1]

    MAX = X.max()
    X = X - np.mean(X, axis=(0, 1), keepdims=True)
    X = X / MAX

    X = padWithZeros(X, patch_size // 2)
    # print(X.shape)
    # 逐像素预测类别
    outputs = np.zeros((height, width))
    #这句话加不加影响不大，但是加了更加保险
    model.to(device)

    model.eval()  # 因为BN要求不少于一个sample，我们需要将模型的模式设置为测试模式才可以只通过一个sample
    for i in range(height):
        for j in range(width):
            if int(y[i, j]) == 0:
                continue
            else:
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                X_test_image = torch.FloatTensor(image_patch.transpose(2, 0, 1)).to(device)
                X_test_image = X_test_image.unsqueeze(0)
                # 将数据的
                # image_patch = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1)
                # X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2))#.to(device)
                # print(X_test_image.shape)

                prediction = model(X_test_image)
                # print(prediction.shape)
                # SSRN这里改为0
                prediction = prediction.argmax(1)
                # prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1
                #print(prediction + 1)
        # if i % 20 == 0:
        #     print('... ... row ', i, ' handling ... ...')
    #print(outputs)

    #这里得弄16的，我们得数据集最多就是16个分类
    cmap = ListedColormap(['#000000',  # 0类别的颜色，就是底板的颜色
                           '#7716DE', '#911D2B', '#7BAEE0', '#947B3A', '#E4B12B',
                           '#E06940', '#941C91', '#E3A8E1', '#539475', '#52E49E',
                           '#E0011C', '#B3E066', '#7A3A53', '#004891', '#804A02',
                           '#E0C069'
                           ])

    plt.matshow(outputs, cmap=cmap)
    # 去掉坐标轴及刻度
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    #按照论文要求，需要保存为eps格式，省得后面再来弄
    if os.path.isdir('Feature_maps/'+dataset) is False:
        os.makedirs('Feature_maps/'+dataset)
    path = 'Feature_maps/'+dataset+'/'+exp_name+'_'+dataset+'.png'
    # 这样的设置可以在保存的时候去除白边，现在只差将颜色弄好了
    #本来只需要300dpi就行了，还是弄高一点吧，省得麻烦
    plt.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0.0)
    plt.show()


