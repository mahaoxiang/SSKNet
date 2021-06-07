import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
import Cut_Data

def Generate_Init(Data_Border, kernel_pos, HSI_CUT_SIZE, KERNEL_CUT_SIZE, KERNEL_BATCH_SIZE):
    kernel_cut = Cut_Data.Cut_kernel_data(Data_Border, kernel_pos, HSI_CUT_SIZE, KERNEL_CUT_SIZE)
    kernel_loder = DataLoader(kernel_cut, batch_size=KERNEL_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    for i, kernel_weight_1 in enumerate(iter(kernel_loder)):
        print(i, kernel_weight_1.shape)
    kernel_weight_1 = torch.as_tensor(kernel_weight_1, dtype=torch.float32).cuda()
    kernel_weight_1 = nn.Parameter(data=kernel_weight_1, requires_grad=False)
    return kernel_weight_1


def Generate_Weights(Data_Border, HSI_CUT_SIZE, KERNEL_CUT_SIZE, KERNEL_BATCH_SIZE, kernel_weight, kernel_pos):

    top_size, bottom_size, left_size, right_size = (int((HSI_CUT_SIZE - 1) / 2), int((HSI_CUT_SIZE - 1) / 2),
                                                    int((HSI_CUT_SIZE - 1) / 2), int((HSI_CUT_SIZE - 1) / 2))
    if len(Data_Border.shape) == 3:
        Data_Border = torch.tensor(Data_Border, dtype=torch.float32).cuda()
        Data_Border = Data_Border.unsqueeze(0)
        # print('data border:', Data_Border.shape)
    Data_Border_feature = F.conv2d(Data_Border, weight=kernel_weight, padding=2)
    Data_Border_feature = F.leaky_relu(Data_Border_feature)
    Features_Border = F.pad(Data_Border_feature, pad=[top_size, bottom_size, left_size, right_size], mode='constant',
                            value=0)
    Features_Border = Features_Border.squeeze()
    Features_Border = Features_Border.data.cpu().numpy()
    pix_max2 = np.max(Features_Border)
    pix_min2 = np.min(Features_Border)
    Features_Border = (Features_Border - pix_min2) / (pix_max2 - pix_min2)
    # print('Features_Border', Features_Border)
    kernel_cut = Cut_Data.Cut_features_data(Features_Border, kernel_pos, HSI_CUT_SIZE, KERNEL_CUT_SIZE)
    kernel_loder = DataLoader(kernel_cut, batch_size=KERNEL_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    for i, kernel_weight in enumerate(iter(kernel_loder)):
        print('Spatial kernel size：', kernel_weight.shape)
    kernel_weight = torch.as_tensor(kernel_weight, dtype=torch.float32).cuda()
    kernel_weight = nn.Parameter(data=kernel_weight, requires_grad=False)
    return kernel_weight, Data_Border_feature
