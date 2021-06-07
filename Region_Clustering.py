import numpy as np
from collections import  Counter
from skimage import color,io
from tqdm import tqdm


def Clustering(img,S_centers, S_clusters):
    global flag, idx_hash_final
    pix_size = 5                                                   # superpixel size
    d_th = 4                                                       # Hash similarity threshold
    idx_num = 1                                                    # Final label index
    sub_num = 0
    idx_entropy = 0
    label = np.zeros((len(S_centers), pix_size, pix_size))         # Hash fingerprint tag
    kind_label = np.arange((len(S_centers)))                       # Cluster initial label
    diff_label = 500 * np.ones(shape=(len(S_centers),1))           # Superpixel difference table
    final_label = -2 * np.ones(shape=(len(kind_label)))            # Final centroid cluster label
    final_pix_label = -1 * np.ones(shape=(S_clusters.shape[0] * S_clusters.shape[1], 1))
    #final_pix_label = -1 * np.ones(shape=(S_clusters.shape[0], S_clusters.shape[1]))
    for i in range(len(S_centers)):
        idx = (S_clusters == i)
        colornp = img[idx]
        colornp = color.rgb2grey(colornp)
        # print(colornp)
        array_gray = np.resize(colornp,(pix_size,pix_size))        # Superpixel grayscale
        supix_mean = np.sum(array_gray) / pix_size**2              # Superpixel gray average
        # print(supix_mean.shape)
        for j in range(array_gray.shape[0]):
            for p in range(array_gray.shape[1]):
                if(array_gray[j][p] >= supix_mean):
                    label[i][j][p] = 1
                else:
                    label[i][j][p] = 0

         # hist = list(array_gray)
         # print(array_gray.shape)

    label = label.reshape(len(S_centers), pix_size**2)
    # print('Superpixel hash label',label,label.shape)

    for x in tqdm(range(len(S_centers))):
        # if(len(label[i] == len(label[j]))):
        for y in range(x,len(S_centers)):
            if x == y and y < len(S_centers)-1:
                y = y + 1
            diff = label[x] - label[y]
            diff_num =len(label[x][np.nonzero(diff)])
            # print(x,y,diff_num,)
            if (diff_num <= d_th) and (diff_num < diff_label[y]):
                if (kind_label[x] != x ) :
                    kind_label[y] = kind_label[x]
                    diff_label[y] = diff_num
                else:
                    kind_label[y] = x
                    diff_label[y] = diff_num
    # print(kind_label)
    # print(Counter(kind_label))

    idx_label = np.unique(kind_label)                              # Returns the sorted category index array
    print('idx_label',idx_label)
    for t in idx_label:
        for q in range(len(kind_label)):
            if kind_label[q] == t and Counter(kind_label)[t] > 10:
                final_label[q] = idx_num
                flag = 1
            if kind_label[q] == t and Counter(kind_label)[t] <= 10:
                final_label[q] = -1
                flag = 0
        idx_num += flag
    for a in range(len(kind_label)):
        if final_label[a] == -1:
            final_label[a] = idx_num
    # print(final_label)
    # final_label = final_label[:len(final_label) - z + 1]

    # print('Final number in each category:',Counter(final_label))
    print('finlabel',final_label.shape)
    print('centers',S_centers.shape)
    '''flag_label = S_clusters.reshape(S_clusters.shape[0] * S_clusters.shape[1],1)
    for q in range(len(final_label)):
        for p in range(len(final_pix_label)):
                if flag_label[p] == q:
                    final_pix_label[p] = final_label[q]
    final_pix_label.reshape(S_clusters.shape[0], S_clusters.shape[1])
    print('finall pix label',final_pix_label)'''
    hash_slic_label = np.column_stack((S_centers, final_label))
    # print('hash slic label:',hash_slic_label.shape)

    ''' Information Entropy Mapping '''
    entropy = np.zeros(len(np.unique(final_label)))
    stander = len(np.unique(final_label))
    # print('np.uniqule',np.unique(final_label))
    for i, k in enumerate(np.unique(final_label)):
        idx1 = (final_label == k)
        idx2 = S_centers[idx1]
        idxnum = np.random.randint(0, Counter(final_label)[k], [1])
        idxy, idxx = int(idx2[idxnum, 3]), int(idx2[idxnum, 4])
        # print(idxx, idxy)
        idx_img = img[idxx - 2:idxx + 3, idxy - 2:idxy + 3, :]
        idx_img_grey = color.rgb2grey(idx_img).reshape(25)
        # print('np.uniqule_supergrey', np.unique(idx_img_grey))
        for c, j in enumerate(np.unique(idx_img_grey)):
            idx_num = Counter(idx_img_grey)[j]
            idx_p = idx_num / 25
            idx_entropy -= idx_p * np.log2(idx_p)
            # print('idx_entropy', idx_entropy)
        entropy[i] = idx_entropy
        idx_entropy = 0
    percent = (entropy / np.sum(entropy) * 100).astype(np.int)
    if np.sum(percent) < 100:
        reduce_cut = 100 - np.sum(percent)
        for e in range(reduce_cut):
            if e == stander - 1:
                e = 0
            percent[e] += 1
    print('Each Group Entropy:', entropy)
    print('Each Group Mapping:', percent)
    for t, q in enumerate(np.unique(final_label)):
        idx3 = (hash_slic_label[:, 5] == q)
        idx_hash = hash_slic_label[idx3]
        idx_rand = np.random.randint(0, len(idx_hash), [percent[t]])
        idx_hash_t = idx_hash[idx_rand]
        if t == 0:
            idx_hash_final = idx_hash_t
        else:
            idx_hash_final = np.vstack((idx_hash_final, idx_hash_t))
    print('Kernel Total Number:', idx_hash_final.shape)
    return idx_hash_final
    























