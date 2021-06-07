import numpy as np
import torch.utils.data


class Cutdata(torch.utils.data.Dataset):                           # Train data cut
    def __init__(self, image, pos_lab, cut_size, transform=None, target_transform=None):
        super(Cutdata, self).__init__()
        self.cut_size = int((cut_size - 1) / 2)                    # (B,C,H,W)
        self.pos_lab = pos_lab
        self.image = image
        self.transform = transform
        # self.t = 0

    def __getitem__(self, index):
        i, j, label = self.pos_lab[index]
        i = i + self.cut_size
        j = j + self.cut_size
        image_cut = self.image[:, i - self.cut_size: i + self.cut_size + 1,
                    j - self.cut_size: j + self.cut_size + 1]
        # self.t += 1
        # print(self.t)
        # print('image_cut shape',image_cut.shape)
        self.label = label
        return image_cut, self.label

    def __len__(self):
        return len(self.pos_lab)


class Cutalldata(torch.utils.data.Dataset):                        # All data cut
    def __init__(self, image, pos_lab, cut_size, transform=None, target_transform=None):
        super(Cutalldata, self).__init__()
        self.cut_size = int((cut_size - 1) / 2)
        self.pos_lab = pos_lab
        self.image = image
        self.transform = transform

    def __getitem__(self, index):
        i, j = self.pos_lab[index]
        i = i + self.cut_size
        j = j + self.cut_size
        image_cut = self.image[:, i - self.cut_size: i + self.cut_size + 1,
                    j - self.cut_size: j + self.cut_size + 1]
        return image_cut

    def __len__(self):
        return len(self.pos_lab)


class Cut_kernel_data(torch.utils.data.Dataset):                    # Spectral kernel cut
    def __init__(self, image, pos_lab, cut_size, kernel_size, transform=None, target_transform=None):
        super(Cut_kernel_data, self).__init__()
        self.cut_size = int((cut_size - 1) / 2)
        self.kernel_size = int((kernel_size - 1) / 2)
        self.pos_lab = pos_lab
        self.image = image
        self.transform = transform

    def __getitem__(self, index):
        i, j = self.pos_lab[index]
        i = i + self.cut_size
        j = j + self.cut_size
        image_cut = self.image[:, i - self.kernel_size: i + self.kernel_size + 1,
                    j - self.kernel_size: j + self.kernel_size + 1]

        return image_cut

    def __len__(self):
        return len(self.pos_lab)


class Cut_features_data(torch.utils.data.Dataset):
    def __init__(self, image, pos_lab, cut_size, kernel_size, transform=None, target_transform=None):
        super(Cut_features_data, self).__init__()
        self.cut_size = int((cut_size - 1) / 2)
        self.kernel_size = int((kernel_size - 1) / 2)
        self.pos_lab = pos_lab
        self.image = image
        self.transform = transform

    def __getitem__(self, index):
        i, j = self.pos_lab[index]
        i = i + self.cut_size
        j = j + self.cut_size
        image_cut = self.image[:, i - self.kernel_size: i + self.kernel_size + 1,
                    j - self.kernel_size: j + self.kernel_size + 1]

        return image_cut

    def __len__(self):
        return len(self.pos_lab)