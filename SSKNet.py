import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class SeKG_Module(nn.Module):

    def __init__(self, channels, cut_size, reduction=16):
        super(SeKG_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                              padding=0)
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_2 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_3 = nn.Conv2d(1, 1, kernel_size=(7, 7), stride=1, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                              padding=0)
        self.sigmoid = nn.Sigmoid()
        self.cut_size = int((cut_size - 1) / 2)

    def forward(self, x):
        global spectrum_kernel_all, spectrum
        original = x
        x = self.avg_pool(x)
        spe_f = torch.transpose(x, 1, 3)
        # print(x.shape)
        spe_f3 = self.conv_1(spe_f)
        spe_f5 = self.conv_2(spe_f)
        spe_f7 = self.conv_3(spe_f)

        x1_3 = torch.transpose(spe_f3, 3, 1)
        x1_5 = torch.transpose(spe_f5, 3, 1)
        x1_7 = torch.transpose(spe_f7, 3, 1)
        x = x + x1_3  + x1_5 + x1_7
        # print(x.shape)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        original = original * x
        original = torch.unsqueeze(original, dim=0).expand((original.shape[0], original.shape[0], original.shape[1],
                                                            original.shape[2], original.shape[3]))
        for i in range(original.shape[1]):
            for k in range(5):
                idx = np.random.randint(0, original.shape[2], 100)
                spectrum_kernel_s = original[i:i+1, k:k+1, idx, self.cut_size-1:self.cut_size+2,
                                                                self.cut_size-1:self.cut_size+2]
                if k == 0:
                    spectrum_kernel_all = spectrum_kernel_s
                else:
                    spectrum_kernel_all = torch.cat((spectrum_kernel_s, spectrum_kernel_all), dim=1)
                    # print(spectrum_kernel_all.shape)
            if i == 0:
                spectrum = spectrum_kernel_all
            else:
                spectrum = torch.cat((spectrum, spectrum_kernel_all), dim=0)
                # print('spectrum:', spectrum.shape)
        return spectrum



class Layer(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(Layer, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in or stride != 1:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class SSKNet(nn.Module):
    def __init__(self):
        super(SSKNet, self).__init__()
        self.sekg = SeKG_Module(200, cut_size=11, reduction=16)    # I_200
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 50, kernel_size=3, stride=1, padding=1),  # I_200
            nn.BatchNorm2d(50)
        )
        self.layer1 = Layer(50, 100, stride=1)
        self.layer2 = Layer(100, 100, stride=1)
        self.layer3 = Layer(100, 100, stride=1)
        self.layer4 = Layer(100, 100, stride=1)
        self.layer5 = Layer(100, 100, stride=1)

        self.outlayer = nn.Linear(100 * 1 * 1, 16)
        self.extra = nn.Conv2d(200, 5, kernel_size=1, stride=1)     # I_200
        self.extra100 = nn.Conv2d(200, 100, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(100)
        self.batch_norm2 = nn.BatchNorm2d(5)

    def forward(self, x, kernel_weight1, kernel_weight2, kernel_weight3, kernel_weight4, HSI_CUT_SIZE):
        spectrum_kernel = self.sekg(x)
        # print(spectrum_kernel)
        origin_data = x
        origin_data = self.extra(origin_data)
        origin_data = self.batch_norm2(origin_data)
        spact = self.extra100(x)
        spact = F.leaky_relu(self.batch_norm(spact))
        x = F.conv2d(x, weight=kernel_weight1, padding=2)
        # print(x.shape)
        x = F.leaky_relu(self.batch_norm(x))
        x = F.conv2d(x, weight=kernel_weight2, padding=2)
        x = F.leaky_relu(self.batch_norm(x))
        x = F.conv2d(x, weight=kernel_weight3, padding=2)
        x = F.leaky_relu(self.batch_norm(x))
        x = F.conv2d(x, weight=kernel_weight4, padding=2)
        x = F.leaky_relu(self.batch_norm(x))
        x = x + spact
        x = F.leaky_relu(x)
        # print(x)
        for t in range(x.shape[0]):
            x_feature = x[t:t+1, :, :, :]
            # x_feature = x_feature_1.unsqueeze(0)
            # print('p_num:', x_feature.shape)
            p_kernel = spectrum_kernel[t:t + 1, :, :, :, :]
            p_kernel = torch.tensor(p_kernel, requires_grad=False)
            # print('p_kernel:', p_kernel.shape)
            p_kernel = torch.squeeze(p_kernel, dim=0)
            # print('p_kernel', p_kernel.shape)
            out = F.conv2d(x_feature, weight=p_kernel, stride=1, padding=1)
            # print(x)
            if t == 0:
                x_all = out
            else:
                x_all = torch.cat((x_all, out), dim=0)
            # x_all[t] = x
        x_all = self.batch_norm2(x_all)
        x_all = x_all + origin_data
        x_all = self.batch_norm2(x_all)
        x = F.relu(x_all)

        # print('x:', x.shape)
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x
