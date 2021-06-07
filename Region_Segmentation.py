import numpy as np
from skimage import io, color
import tqdm

def generate_pixels():
    indnp = np.mgrid[0:S_height,0:S_width].swapaxes(0,2).swapaxes(0,1)
    # print(indnp)
    for i in tqdm.tqdm(range(S_ITERATIONS)):
        SLIC_distances = 1 * np.ones(img.shape[:2])
        for j in range(S_centers.shape[0]):                                     # Cycle based on each centroid
            # Search close pixels in the 2S*2S area
            x_low, x_high = int(S_centers[j][3] - step), int(S_centers[j][3] + step)
            y_low, y_high = int(S_centers[j][4] - step), int(S_centers[j][4] + step)

            if x_low <= 0:
                x_low = 0

            if x_high > S_width:
                x_high = S_width

            if y_low <=0:
                y_low = 0

            if y_high > S_height:
                y_high = S_height

            cropimg = S_labimg[y_low : y_high , x_low : x_high]
            # Calculate distance between each point
            color_diff = cropimg - S_labimg[int(S_centers[j][4]), int(S_centers[j][3])]
            # color_distance = np.sum(np.square(color_diff), axis=2)
            color_distance = np.sqrt(np.sum(np.square(color_diff), axis=2))
            yy, xx = np.ogrid[y_low : y_high, x_low : x_high]                   # Generate 2D space coordinates
            # pixdist = (yy - S_centers[j][4]) ** 2 + (xx - S_centers[j][3]) ** 2
            pixdist = ((yy-S_centers[j][4])**2 + (xx-S_centers[j][3])**2)**0.5

            # S_m is "m" in the paper, (m/S)*dxy
            dist = ((color_distance / S_m) ** 2 + (pixdist / step) ** 2)        # Calculate 5D total distance
            # dist = ((color_distance/S_m)**2 + (pixdist/step)**2)**0.5

            distance_crop = SLIC_distances[y_low : y_high, x_low : x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]                                      # Update distance
            SLIC_distances[y_low : y_high, x_low : x_high] = distance_crop      # Optimal distance in 2S * 2S
            S_clusters[y_low : y_high, x_low : x_high][idx] = j

        for k in range(len(S_centers)):                                         # SLIC_centers=[L,A,B,X,Y]
            idx = (S_clusters == k)
            # print('idx:',idx.shape)
            colornp = S_labimg[idx]
            # print(colornp.shape)
            distnp = indnp[idx]
            S_centers[k][0:3] = np.sum(colornp, axis=0)                         # Sum LAB values
            sumy, sumx = np.sum(distnp, axis=0)                                 # Sum space coordinates
            S_centers[k][3:] = sumx, sumy
            S_centers[k] /= np.sum(idx)

def display_contours(color):
    rgb_img = img.copy()
    is_taken = np.zeros(img.shape[:2], np.bool)
    contours = []

    for i in range(S_width):
        for j in range(S_height):
            nr_p = 0
            for dx, dy in [(-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1)]:
                x = i + dx
                y = j + dy
                if x>=0 and x < S_width and y>=0 and y < S_height:
                    if is_taken[y, x] == False and S_clusters[j, i] != S_clusters[y, x]:
                        nr_p += 1

            if nr_p >= 2:
                is_taken[j, i] = True
                contours.append([j, i])
    for i in range(len(contours)):
        rgb_img[contours[i][0], contours[i][1]] = color
    # for k in range(S_centers.shape[0]):
    #     i,j = S_centers[k][-2:]
    #     img[int(i),int(j)] = (0,0,0)
    # io.imsave("S_contours.jpg", rgb_img)

    return rgb_img

def display_center():
    lab_img = np.zeros([S_height,S_width,3]).astype(np.float64)
    for i in range(S_width):
        for j in range(S_height):
            k = int(S_clusters[j, i])
            lab_img[j,i] = S_centers[k][0:3]
    rgb_img = color.lab2rgb(lab_img)
    # io.imsave("S_centers.jpg",rgb_img)
    return (rgb_img*255).astype(np.uint8)



def find_local_minimum(center):
    min_grad = 1
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):                        # (i,j)
        for j in range(center[1] - 1, center[1] + 2):                    # Draw initial spreading point
            c1 = S_labimg[j+1, i]
            c2 = S_labimg[j, i+1]
            c3 = S_labimg[j, i]
            if ((c1[0] - c3[0])**2) + ((c2[0] - c3[0])**2) < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                loc_min = [i, j]                                         # Calculate the min gradient
            '''if ((c1[0] - c3[0])**2)**0.5 + ((c2[0] - c3[0])**2)**0.5 < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                loc_min = [i, j]'''
    return loc_min


# Find superpixel centroid
def calculate_centers():
    centers = []
    for i in range(step, S_width - int(step/2), step): #（i,j）
        for j in range(step, S_height - int(step/2), step):              # Take the center point evenly in step
            nc = find_local_minimum(center=(i, j))
            color = S_labimg[nc[1], nc[0]]                               # Mapp to LAB coordinates
            center = [color[0], color[1], color[2], nc[0], nc[1]]        # [L,A,B,i,j]
            centers.append(center)
    return centers



def region_sege():
    global img, step, S_ITERATIONS, S_height, S_width, S_labimg, S_distances, S_clusters, S_center_counts, \
        S_centers, S_m, SLIC_k
    distances_m = 50
    super_pixel_k = 800                                                  # 800 / 8000 / 4000
    img = io.imread('image_pca1.png')
    print('Maxpixel:',img.max(),'inpixel:',img.min())
    S_m = distances_m                                                    # Distance weight
    S_k = super_pixel_k                                                  # ber
    step= int((img.shape[0]*img.shape[1]/S_k)**0.5)                      # Superpixel size
    print('Sup_size',step)
    S_ITERATIONS= 1
    S_height, S_width = img.shape[:2]
    S_labimg = color.rgb2lab(img)                                        # RGB-LAB
    S_distances = 1 * np.ones(img.shape[:2])
    S_clusters = -1 * S_distances
    centers_label = calculate_centers()                                  # Superpixel centroid table 5D
    S_center_counts = np.zeros(len(centers_label))
    print('Centroid Num:', S_center_counts.shape)
    S_centers = np.array(centers_label)

    '''S_center_counts = np.zeros(len(calculate_centers()))
    print(S_center_counts)
    S_centers = np.array(calculate_centers())'''

    generate_pixels()
    calculate_centers()
    display_contours([0.0, 0.0, 0.0])
    display_center()
    return S_centers, S_clusters
    # print(img,img_center,img_contours)
    # result = np.hstack([img_contours,img_center])


