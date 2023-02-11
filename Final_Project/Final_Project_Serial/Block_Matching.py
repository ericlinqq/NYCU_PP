# Block Matching after image rectification
import numpy as np
import cv2
from tqdm import trange
import time
from scipy.spatial.distance import cdist

def Read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

def Normalize_vec(vec):    
    # Reshape the vector into 1-dim
    vec = np.array(vec).reshape(-1)
    #print(vec.shape)

    # Compute norm of vector
    tmp = 0
    for i in range(len(vec)):
        tmp += vec[i] ** 2
    
    if tmp == 0:
        return vec
    else:
        vec = vec / tmp
        return vec

def sum_of_abs_diff(pixel_vals_1, pixel_vals_2):
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

#Function to find corresponding block in the other images using SAD
def compare_blocks(y, x, block_left, right_array, window, search_range):
    x_min = max(window, x - search_range)
    x_max = min(right_array.shape[1]-window, x + search_range)
    first = True
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+window, x: x+window]
        sad = sum_of_abs_diff(block_left, block_right)
        
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index

if __name__ == '__main__':
    
    # Read images
    img_path_l = "img3_rectified.png"
    img_path_r = "img4_rectified.png"
    img_l, img_gray_l = Read_img(img_path_l)
    img_r, img_gray_r = Read_img(img_path_r)

    # Block Matching
    h, w = img_gray_l.shape
    window_size = 9
    search_range = 96
    disparity_map = np.zeros(img_gray_l.shape)

    print(time.ctime())
    for y in trange(window_size, h-window_size):
        for x in range(window_size, w-window_size):
            block_left = img_gray_l[y:y + window_size, x:x + window_size]
            min_index = compare_blocks(y, x, block_left, img_gray_r, window_size, search_range) 
            disparity_map[y, x] = abs(min_index[1] - x)
    
    print(time.ctime())
    #print(disparity_map)

    disparity = np.uint8(disparity_map * 255 / np.max(disparity_map))
    heatmap = cv2.applyColorMap(disparity, cv2.COLORMAP_BONE)
    cv2.imshow("Disparity_map", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("Disparity_map.png", heatmap)
    # If we just implement Block Matching and don't limit the search range, it will be time consuming


