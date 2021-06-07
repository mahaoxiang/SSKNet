import torch.nn as nn
from torch import optim
import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA
import cv2
from skimage import color, io
import Region_Segmentation
import Region_Clustering
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import torch.utils.data
import Cut_Data
import SSKNet
from Generate_Kernel import Generate_Weights, Generate_Init




HSI_CUT_SIZE = 11                                                               # Parameter initialization
KERNEL_CUT_SIZE = 5
TRAIN_RATE = [0.55,0.035,0.056,0.125,0.074,0.05,0.5,0.064,0.5,0.05,0.024,0.058,0.145,0.041,0.078,0.5]
EPOCH = 300
BATCH_SIZE = 32
KERNEL_BATCH_SIZE = 100
LR = 0.003
BorderInter = cv2.BORDER_REFLECT_101
Data_Hsi = scio.loadmat('Indian_pines_corrected.mat')                           # Load data
Label_Hsi = scio.loadmat('Indian_pines_gt.mat')                                 # Load label
# print(Data_Hsi)
# print(Label_Hsi)
Data = np.array(Data_Hsi['indian_pines_corrected'])
Label = np.array(Label_Hsi['indian_pines_gt'])
top_size, bottom_size, left_size, right_size = (int((HSI_CUT_SIZE - 1) / 2), int((HSI_CUT_SIZE - 1) / 2),
                                                int((HSI_CUT_SIZE - 1) / 2), int((HSI_CUT_SIZE - 1) / 2))
Data_Border = cv2.copyMakeBorder(Data, top_size, bottom_size, left_size, right_size, BorderInter)
all_Label = np.max(Label)
# Data = np.array(Data).transpose((2,0,1))
Data_Border = np.array(Data_Border).transpose((2, 0, 1))
pix_max = np.max(Data_Border)                                                   # Data normalization
pix_min = np.min(Data_Border)
Data_Border = (Data_Border - pix_min) / (pix_max - pix_min)

array_1 = Data.reshape(np.prod(Data.shape[:2]), np.prod(Data.shape[2:]))
pca = PCA(n_components=3)                                                       # Principal components
array_2 = pca.fit_transform(array_1)                                            # Parameter fitting
# print(array_2.shape)
Data_PCA = array_2.reshape(Data.shape[0], Data.shape[1], array_2.shape[1])      # Recovery dimension
cv2.imwrite('image_pca1.png', Data_PCA)
img = io.imread('image_pca1.png')

S_centers, S_clusters = Region_Segmentation.region_sege()                       # Homogenous region segmentation
idx_hash_final = Region_Clustering.Clustering(img, S_centers, S_clusters)       # Rough region clustering
hash_label = np.array(idx_hash_final[:, [4, 3, 5]], dtype='int64')
# idx_keenel = np.lexsort(hash_label.T)
# hash_label = hash_label[idx_keenel]
kernel_pos = np.array(hash_label[:, [0, 1]], dtype='int64')
np.random.shuffle(kernel_pos)
# kernel_pos = kernel_pos[idx_num]
kernel_weight_1 = Generate_Init(Data_Border, kernel_pos, HSI_CUT_SIZE, KERNEL_CUT_SIZE, KERNEL_BATCH_SIZE)
Data = np.array(Data).transpose((2, 0, 1))


global train_label, train_data_pos, valid_label, valid_data_pos, all_label, all_label_pos
index_all = np.array(np.where(Label != -1))
index_all = index_all.transpose()
len_all = len(index_all)
index_a = np.arange(len_all, dtype='int64')
np.random.shuffle(index_a)
all_pos = index_all[index_a]
# print('all data',all_data_pos.shape)
# all_data_pos = np.vstack((empty_data_pos, train_data_pos, valid_data_pos, test_data_pos))
np.random.shuffle(all_pos)

for i in range(1, all_Label+1):
    index_label_i = np.array(np.where(Label == i))
    index_label_i = index_label_i.transpose()
    # print(index_label_i.shape)
    len_label_i = len(index_label_i)
    # all_num += len_label_i
    len_train_i = int(len_label_i * TRAIN_RATE[i-1])
    len_valid_i = int((len_label_i - len_train_i))
    len_label_i = int(len_label_i )
    index_i = np.arange(len_label_i, dtype='int64')
    np.random.shuffle(index_i)
    train_label_i = i * np.ones((len_train_i, 1), dtype='int64')
    valid_label_i = i * np.ones((len_valid_i, 1), dtype='int64')
    all_label_i = i * np.ones((len_label_i, 1), dtype='int64')

    if i == 1:
        train_label = train_label_i
        train_data_pos = index_label_i[index_i[:len_train_i]]
        valid_label = valid_label_i
        valid_data_pos = index_label_i[index_i[len_train_i:len_train_i + len_valid_i]]
        all_label = all_label_i
        all_label_pos = index_label_i[index_i[:len_label_i]]
    else:
        train_label = np.append(train_label, train_label_i, axis=0)
        train_data_pos = np.append(train_data_pos, index_label_i[index_i[:len_train_i]], axis=0)
        valid_label = np.append(valid_label, valid_label_i, axis=0)
        valid_data_pos = np.append(valid_data_pos, index_label_i[index_i[len_train_i:len_train_i + len_valid_i]], axis=0)
        all_label = np.append(all_label, all_label_i, axis=0)
        all_label_pos = np.append(all_label_pos, index_label_i[index_i[:len_label_i]], axis=0)

train_label -= 1
valid_label -= 1
all_label -= 1
train_data = np.hstack((train_data_pos, train_label))
np.random.shuffle(train_data)
valid_data = np.hstack((valid_data_pos, valid_label))
np.random.shuffle(valid_data)
all_label_data1 = np.hstack((all_label_pos, all_label))
np.random.shuffle(all_label_data1)
# all_label_data2 = np.vstack((train_data, valid_data))
# np.random.shuffle(all_label_data2)
#print( len(all_label_data1))
#print(len(valid_data))
true_label = all_label_data1[:, 2]
pred_label = np.zeros(len(true_label), dtype=int)
pix_max = np.max(Data_Border)
pix_min = np.min(Data_Border)
Data_Border = (Data_Border - pix_min) / (pix_max - pix_min)

train_cut = Cut_Data.Cutdata(Data_Border, train_data, HSI_CUT_SIZE)
train_loder = DataLoader(train_cut, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
# print(train_cut[0][0])
# data1 , label1 = train_cut[5000]
# print(label1)
# print(data1.shape)
valid_cut = Cut_Data.Cutdata(Data_Border, valid_data, HSI_CUT_SIZE)
valid_loder = DataLoader(valid_cut, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
all_label_cut = Cut_Data.Cutdata(Data_Border, all_label_data1, HSI_CUT_SIZE)
all_label_loder = DataLoader(all_label_cut, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_cut = Cut_Data.Cutalldata(Data_Border, all_pos, HSI_CUT_SIZE)
all_loder = DataLoader(all_cut, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)


device = torch.device('cuda')
model = SSKNet.SSKNet().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
schedulr = MultiStepLR(optimizer, milestones=[60, 90, 120, 150, 180, 200, 220, 240], gamma=0.9)

# Generate spatial kernels
kernel_weight_2, Data_Border_feature = Generate_Weights(Data_Border, HSI_CUT_SIZE, KERNEL_CUT_SIZE, KERNEL_BATCH_SIZE,
                                                        kernel_weight_1, kernel_pos)
kernel_weight_3, Data_Border_feature = Generate_Weights(Data_Border_feature, HSI_CUT_SIZE, KERNEL_CUT_SIZE,
                                                        KERNEL_BATCH_SIZE, kernel_weight_2, kernel_pos)
kernel_weight_4, Data_Border_feature = Generate_Weights(Data_Border_feature, HSI_CUT_SIZE, KERNEL_CUT_SIZE,
                                                        KERNEL_BATCH_SIZE, kernel_weight_3, kernel_pos)

# Train
for epoch in range(EPOCH):
    model.train()
    schedulr.step()
    valid_idx = iter(valid_loder)
    for batchidx, (x, y) in enumerate(iter(train_loder)):
        x = torch.as_tensor(x, dtype= torch.float32).to(device)                  # weight-float32 Data-double64
        y = torch.as_tensor(y, dtype= torch.long).to(device)
        # print(y.dtype)
        # print(batchidx)
        output = model(x, kernel_weight_1, kernel_weight_2, kernel_weight_3, kernel_weight_4, HSI_CUT_SIZE)
        loss = criteon(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchidx % 40 == 0:
            loss = loss.data.cpu().numpy()
            x_v, y_v = next(valid_idx)
            x_v = torch.as_tensor(x_v, dtype= torch.float32).to(device)
            y_v = torch.as_tensor(y_v, dtype= torch.long).to(device)
            output_v = model(x_v, kernel_weight_1, kernel_weight_2, kernel_weight_3, kernel_weight_4, HSI_CUT_SIZE)
            output_v = output_v.data.cpu().numpy()
            kind_v = np.argmax(output_v, axis= 1)
            accuracy_v = 0
            for idx_v, i in enumerate(kind_v):
                if i == y_v[idx_v]:
                    accuracy_v += 1
            accuracy_v = accuracy_v / (len(kind_v))
            print('Epoch:', epoch, '  Batch:', batchidx, '  Loss:%.4f' % loss, '  Accuracy:%.2f' % (accuracy_v * 100))

num_t = 0
all_color = np.zeros((145, 145, 3))
all_result_label = np.zeros((145, 145))
for k, (x_t, y_t) in enumerate(iter(all_label_loder)):
    x_t = torch.as_tensor(x_t, dtype=torch.float32).to(device)
    y_t = torch.as_tensor(y_t, dtype=torch.long).to(device)
    output_t = model(x_t, kernel_weight_1, kernel_weight_2, kernel_weight_3, kernel_weight_4, HSI_CUT_SIZE)
    output_t = output_t.data.cpu().numpy()
    kind_t = np.argmax(output_t, axis=1)
    # print('kindt',kind_t.shape)
    for i, j in enumerate(kind_t):
        pred_label[num_t] = j
        num_t += 1


conf_mat = confusion_matrix(true_label, pred_label, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(conf_mat)
kappa_value = cohen_kappa_score(true_label, pred_label, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print('kappa:', kappa_value)
dsa = 0
pe = 0
pe_rows = np.sum(conf_mat, axis=0)
pe_cols = np.sum(conf_mat, axis=1)
for i, diag in enumerate(conf_mat):
    pred = diag[i]
    dsa += diag[i]
    pe += diag[i] * diag[i]
    acr = pred / pe_cols[i]
    rep = pred / pe_rows[i]
    f1 = 2*acr*rep / (acr + rep)
    print(" Class %d:  Accuracy: %.5f  Recall: %.5f  F1: %.5f" % (i, acr, rep, f1))
all_conf_mat = np.sum(conf_mat)
p = dsa / all_conf_mat
pe = pe / (all_conf_mat * all_conf_mat)
kappa = (p - pe) / (1 - pe)
print("OA: %.5f  Kappa: %.5f" % (p, kappa))


for k_a, (x_a, y_a) in enumerate(iter(all_label_loder)):
    x_a = torch.as_tensor(x_a, dtype=torch.float32).to(device)
    output_t = model(x_a, kernel_weight_1, kernel_weight_2, kernel_weight_3, kernel_weight_4, HSI_CUT_SIZE)
    output_t = output_t.data.cpu().numpy()
    kind_t = np.argmax(output_t, axis=1)
    for i, j in enumerate(kind_t):
        xt, yt, lab = all_label_data1[k_a * x_a.shape[0] + i]                     # all_pos/ all_label_pos
        label_t = kind_t[i]
        if label_t == 0:
            all_color[xt][yt] = [255,255,102]
            all_result_label[xt][yt] = label_t
        if label_t == 1:
            all_color[xt][yt] = [0,48,205]
            all_result_label[xt][yt] = label_t
        if label_t == 2:
            all_color[xt][yt] = [255,102,0]
            all_result_label[xt][yt] = label_t
        if label_t == 3:
            all_color[xt][yt] = [0,255,104]
            all_result_label[xt][yt] = label_t
        if label_t == 4:
            all_color[xt][yt] = [255,48,205]
            all_result_label[xt][yt] = label_t
        if label_t == 5:
            all_color[xt][yt] = [102,0,255]
            all_result_label[xt][yt] = label_t
        if label_t == 6:
            all_color[xt][yt] = [0,154,255]
            all_result_label[xt][yt] = label_t
        if label_t == 7:
            all_color[xt][yt] = [0,255,0]
            all_result_label[xt][yt] = label_t
        if label_t == 8:
            all_color[xt][yt] = [128,128,0]
            all_result_label[xt][yt] = label_t
        if label_t == 9:
            all_color[xt][yt] = [128,0,128]
            all_result_label[xt][yt] = label_t
        if label_t == 10:
            all_color[xt][yt] = [47,205,205]
            all_result_label[xt][yt] = label_t
        if label_t == 11:
            all_color[xt][yt] = [0,102,102]
            all_result_label[xt][yt] = label_t
        if label_t == 12:
            all_color[xt][yt] = [47,205,48]
            all_result_label[xt][yt] = label_t
        if label_t == 13:
            all_color[xt][yt] = [102,48,0]
            all_result_label[xt][yt] = label_t
        if label_t == 14:
            all_color[xt][yt] = [102,255,255]
            all_result_label[xt][yt] = label_t
        if label_t == 15:
            all_color[xt][yt] = [255,255,0]
            all_result_label[xt][yt] = label_t
io.imsave('./out_picture.png',all_color.astype(np.uint8))

