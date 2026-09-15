
percent_BOS = 0.5
patch_size_BOS = 11

percent_IP = 3
patch_size_IP = 11

percent_KSC = 5
patch_size_KSC = 11

percent_HUST2013 = 10
patch_size_HUST2013 = 11

percent_Loukia = 0.5
patch_size_Loukia = 13

percent_HUST2018 = 1
patch_size_HUST2018 = 11

percent_PU = 3
patch_size_PU = 11

percent_SV = 3
patch_size_SV = 11

percent_PC = 0.5
patch_size_PC = 13

percent_LongKou = 10
patch_size_LongKou = 11

percent_HanChuan = 0.1
patch_size_HanChuan = 9

percent_HongHu = 10
patch_size_HongHu = 7

percent_XU = 0.5
patch_size_XU = 9




CONFIG_HSIC = {
    'BOS' :{
        'bands' :145,
        'num_classes' : 14,
        'k' : 70, #SSRN特有的参数
        'train_data_path' : 'datasets/BOS/{}%XtrainWindowSize{}.npy'.format(percent_BOS, patch_size_BOS),
        'train_label_path' : 'datasets/BOS/{}%ytrainWindowSize{}.npy'.format(percent_BOS, patch_size_BOS),
        'val_data_path' : 'datasets/BOS/{}%XvalWindowSize{}.npy'.format(percent_BOS, patch_size_BOS),
        'val_label_path' : 'datasets/BOS/{}%yvalWindowSize{}.npy'.format(percent_BOS, patch_size_BOS),
        'test_data_path' : 'datasets/BOS/{}%XtestWindowSize{}.npy'.format(percent_BOS, patch_size_BOS),
        'test_label_path' : 'datasets/BOS/{}%ytestWindowSize{}.npy'.format(percent_BOS, patch_size_BOS)
    },
    'IP' :{
        'bands' :200,
        'num_classes' : 16,
        'k' : 97, #SSRN特有的参数
        'train_data_path' : 'datasets/IP/{}%XtrainWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'train_label_path' : 'datasets/IP/{}%ytrainWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'val_data_path' : 'datasets/IP/{}%XvalWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'val_label_path' : 'datasets/IP/{}%yvalWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'test_data_path' : 'datasets/IP/{}%XtestWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'test_label_path' : 'datasets/IP/{}%ytestWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'only_train_data_path': 'datasets/IP/Onlytrtain{}XtrainWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'only_train_label_path': 'datasets/IP/Onlytrtain{}ytrainWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'only_test_data_path': 'datasets/IP/Onlytrtain{}XtestWindowSize{}.npy'.format(percent_IP, patch_size_IP),
        'only_test_label_path': 'datasets/IP/Onlytrtain{}ytestWindowSize{}.npy'.format(percent_IP, patch_size_IP)

    },
    'KSC': {
        'bands': 176,
        'num_classes': 13,
        'k' : 85, #SSRN特有的参数
        'train_data_path': 'datasets/KSC/{}%XtrainWindowSize{}.npy'.format(percent_KSC, patch_size_KSC),
        'train_label_path': 'datasets/KSC/{}%ytrainWindowSize{}.npy'.format(percent_KSC, patch_size_KSC),
        'val_data_path': 'datasets/KSC/{}%XvalWindowSize{}.npy'.format(percent_KSC, patch_size_KSC),
        'val_label_path': 'datasets/KSC/{}%yvalWindowSize{}.npy'.format(percent_KSC, patch_size_KSC),
        'test_data_path': 'datasets/KSC/{}%XtestWindowSize{}.npy'.format(percent_KSC, patch_size_KSC),
        'test_label_path': 'datasets/KSC/{}%ytestWindowSize{}.npy'.format(percent_KSC, patch_size_KSC)
    },
    'HUST2013': {
        'bands': 144,
        'num_classes': 15,
        'k' : 69, #SSRN特有的参数
        'train_data_path': 'datasets/Houston/{}%XtrainWindowSize{}.npy'.format(percent_HUST2013, patch_size_HUST2013),
        'train_label_path': 'datasets/Houston/{}%ytrainWindowSize{}.npy'.format(percent_HUST2013, patch_size_HUST2013),
        'val_data_path': 'datasets/Houston/{}%XvalWindowSize{}.npy'.format(percent_HUST2013, patch_size_HUST2013),
        'val_label_path': 'datasets/Houston/{}%yvalWindowSize{}.npy'.format(percent_HUST2013, patch_size_HUST2013),
        'test_data_path': 'datasets/Houston/{}%XtestWindowSize{}.npy'.format(percent_HUST2013, patch_size_HUST2013),
        'test_label_path': 'datasets/Houston/{}%ytestWindowSize{}.npy'.format(percent_HUST2013, patch_size_HUST2013)
    },
    'Loukia':{
        'bands': 176,
        'num_classes': 14,
        'k' : 85,
        'train_data_path': 'datasets/Loukia/{}%XtrainWindowSize{}.npy'.format(percent_Loukia, patch_size_Loukia),
        'train_label_path': 'datasets/Loukia/{}%ytrainWindowSize{}.npy'.format(percent_Loukia, patch_size_Loukia),
        'val_data_path': 'datasets/Loukia/{}%XvalWindowSize{}.npy'.format(percent_Loukia, patch_size_Loukia),
        'val_label_path': 'datasets/Loukia/{}%yvalWindowSize{}.npy'.format(percent_Loukia, patch_size_Loukia),
        'test_data_path': 'datasets/Loukia/{}%XtestWindowSize{}.npy'.format(percent_Loukia, patch_size_Loukia),
        'test_label_path': 'datasets/Loukia/{}%ytestWindowSize{}.npy'.format(percent_Loukia, patch_size_Loukia)
    },
    'HUST2018': {
        'bands': 50,
        'num_classes': 20,
        'k': 69,  # SSRN特有的参数
        'train_data_path': 'datasets/HoustonU/{}%XtrainWindowSize{}.npy'.format(percent_HUST2018, patch_size_HUST2018),
        'train_label_path': 'datasets/HoustonU/{}%ytrainWindowSize{}.npy'.format(percent_HUST2018, patch_size_HUST2018),
        'val_data_path': 'datasets/HoustonU/{}%XvalWindowSize{}.npy'.format(percent_HUST2018, patch_size_HUST2018),
        'val_label_path': 'datasets/HoustonU/{}%yvalWindowSize{}.npy'.format(percent_HUST2018, patch_size_HUST2018),
        'test_data_path': 'datasets/HoustonU/{}%XtestWindowSize{}.npy'.format(percent_HUST2018, patch_size_HUST2018),
        'test_label_path': 'datasets/HoustonU/{}%ytestWindowSize{}.npy'.format(percent_HUST2018, patch_size_HUST2018)
    },
    'PU': {
        'bands': 103,
        'num_classes': 9,
        'k': 49,  # SSRN特有的参数
        'train_data_path': 'datasets/PU/{}%XtrainWindowSize{}.npy'.format(percent_PU, patch_size_PU),
        'train_label_path': 'datasets/PU/{}%ytrainWindowSize{}.npy'.format(percent_PU, patch_size_PU),
        'val_data_path': 'datasets/PU/{}%XvalWindowSize{}.npy'.format(percent_PU, patch_size_PU),
        'val_label_path': 'datasets/PU/{}%yvalWindowSize{}.npy'.format(percent_PU, patch_size_PU),
        'test_data_path': 'datasets/PU/{}%XtestWindowSize{}.npy'.format(percent_PU, patch_size_PU),
        'test_label_path': 'datasets/PU/{}%ytestWindowSize{}.npy'.format(percent_PU, patch_size_PU)
    },
    'SV': {
        'bands': 204,
        'num_classes': 16,
         'k': 99,  # SSRN特有的参数
        'train_data_path': 'datasets/SV/{}%XtrainWindowSize{}.npy'.format(percent_SV, patch_size_SV),
        'train_label_path': 'datasets/SV/{}%ytrainWindowSize{}.npy'.format(percent_SV, patch_size_SV),
        'val_data_path': 'datasets/SV/{}%XvalWindowSize{}.npy'.format(percent_SV, patch_size_SV),
        'val_label_path': 'datasets/SV/{}%yvalWindowSize{}.npy'.format(percent_SV, patch_size_SV),
        'test_data_path': 'datasets/SV/{}%XtestWindowSize{}.npy'.format(percent_SV, patch_size_SV),
        'test_label_path': 'datasets/SV/{}%ytestWindowSize{}.npy'.format(percent_SV, patch_size_SV)
    },
    'PC': {
        'bands': 102,
        'num_classes': 9,
        'k': 48,  # SSRN特有的参数
        'train_data_path': 'datasets/PC/{}%XtrainWindowSize{}.npy'.format(percent_PC, patch_size_PC),
        'train_label_path': 'datasets/PC/{}%ytrainWindowSize{}.npy'.format(percent_PC, patch_size_PC),
        'val_data_path': 'datasets/PC/{}%XvalWindowSize{}.npy'.format(percent_PC, patch_size_PC),
        'val_label_path': 'datasets/PC/{}%yvalWindowSize{}.npy'.format(percent_PC, patch_size_PC),
        'test_data_path': 'datasets/PC/{}%XtestWindowSize{}.npy'.format(percent_PC, patch_size_PC),
        'test_label_path': 'datasets/PC/{}%ytestWindowSize{}.npy'.format(percent_PC, patch_size_PC)
    },
    'LongKou': {
        'bands': 270,
        'num_classes': 9,
        'k': 132,  # SSRN特有的参数
        'train_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-LongKou/{}%XtrainWindowSize{}.npy'.format(percent_LongKou, patch_size_LongKou),
        'train_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-LongKou/{}%ytrainWindowSize{}.npy'.format(percent_LongKou, patch_size_LongKou),
        'val_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-LongKou/{}%XvalWindowSize{}.npy'.format(percent_LongKou, patch_size_LongKou),
        'val_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-LongKou/{}%yvalWindowSize{}.npy'.format(percent_LongKou, patch_size_LongKou),
        'test_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-LongKou/{}%XtestWindowSize{}.npy'.format(percent_LongKou, patch_size_LongKou),
        'test_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-LongKou/{}%ytestWindowSize{}.npy'.format(percent_LongKou, patch_size_LongKou)
    },
    'HanChuan': {
        'bands': 274,
        'num_classes': 16,
        'k': 132,  # SSRN特有的参数
        'train_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/{}%XtrainWindowSize{}.npy'.format(
            percent_HanChuan, patch_size_HanChuan),
        'train_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/{}%ytrainWindowSize{}.npy'.format(
            percent_HanChuan, patch_size_HanChuan),
        'val_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/{}%XvalWindowSize{}.npy'.format(
            percent_HanChuan, patch_size_HanChuan),
        'val_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/{}%yvalWindowSize{}.npy'.format(
            percent_HanChuan, patch_size_HanChuan),
        'test_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/{}%XtestWindowSize{}.npy'.format(
            percent_HanChuan, patch_size_HanChuan),
        'test_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/{}%ytestWindowSize{}.npy'.format(
            percent_HanChuan, patch_size_HanChuan)
    },
    'HongHu': {
        'bands': 270,
        'num_classes': 22,
        'train_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HongHu/{}%XtrainWindowSize{}.npy'.format(percent_HongHu, patch_size_HongHu),
        'train_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HongHu/{}%ytrainWindowSize{}.npy'.format(percent_HongHu, patch_size_HongHu),
        'val_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HongHu/{}%XvalWindowSize{}.npy'.format(percent_HongHu, patch_size_HongHu),
        'val_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HongHu/{}%yvalWindowSize{}.npy'.format(percent_HongHu, patch_size_HongHu),
        'test_data_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HongHu/{}%XtestWindowSize{}.npy'.format(percent_HongHu, patch_size_HongHu),
        'test_label_path': 'datasets/WHU-Hi/Matlab_data_format/WHU-Hi-HongHu/{}%ytestWindowSize{}.npy'.format(percent_HongHu, patch_size_HongHu)
    },
    'XU': {
        'bands': 436,
        'num_classes': 9,
        'k': 215,  # SSRN特有的参数
        'train_data_path': 'datasets/XU/{}%XtrainWindowSize{}.npy'.format(percent_XU, patch_size_XU),
        'train_label_path': 'datasets/XU/{}%ytrainWindowSize{}.npy'.format(percent_XU, patch_size_XU),
        'val_data_path': 'datasets/XU/{}%XvalWindowSize{}.npy'.format(percent_XU, patch_size_XU),
        'val_label_path': 'datasets/XU/{}%yvalWindowSize{}.npy'.format(percent_XU, patch_size_XU),
        'test_data_path': 'datasets/XU/{}%XtestWindowSize{}.npy'.format(percent_XU, patch_size_XU),
        'test_label_path': 'datasets/XU/{}%ytestWindowSize{}.npy'.format(percent_XU, patch_size_XU)
    },
}