from typing import Mapping
import numpy as np
import cv2
from scipy.spatial import distance


# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

def Create_SIFT(img_gray):
    SIFT_Detector = cv2.SIFT_create()
    kp, des = SIFT_Detector.detectAndCompute(img_gray, None)
    return kp, des

def Compute_distance(a,b):
    distance = 0
    for i in range(128):
        distance += (a[i] - b[i]) ** 2
    return distance

def KNN(kp0, kp1, des0, des1):
    # For each key point in image0, we want to find the two closest points in image1 (using descriptor)
    m = len(kp0)
    n = len(kp1)

    mapping = np.zeros((m,2), dtype=np.int32)
    
    # Using scipy.spatial.distance to compute distance between all pairs of keypoints parallelly
    dist_matrix = distance.cdist(des0, des1, 'euclidean')
    
    # argsort
    tmp_map = np.argsort(dist_matrix)

    for i in range(m):
        for j in range(n):
            if tmp_map[i][j] == 0:
                mapping[i][0] = j
            elif tmp_map[i][j] == 1:
                mapping[i][1] = j
    
    return dist_matrix, mapping

def Test_match(dist_matrix, mapping):
    true_map = []

    for i in range(len(mapping)):
        dist0 = dist_matrix[i][mapping[i][0]]
        dist1 = dist_matrix[i][mapping[i][1]]
        if (dist0 < 0.5 * dist1):
            tmp_list = []
            tmp_list.append(i)
            tmp_list.append(mapping[i][0])
            true_map.append(tmp_list)
    
    true_map = np.array(true_map)
    return true_map

def Fundamental_F(left_points, right_points):
    A = np.ones((8,9))

    for i in range(8):
        A[i][0] = left_points[i][0] * right_points[i][0]
        A[i][1] = left_points[i][1] * right_points[i][0]
        A[i][2] = right_points[i][0]
        A[i][3] = left_points[i][0] * right_points[i][1]
        A[i][4] = left_points[i][1] * right_points[i][1]
        A[i][5] = right_points[i][1]
        A[i][6] = left_points[i][0]
        A[i][7] = left_points[i][1]
    
    # Using SVD to get F
    U, S, VT = np.linalg.svd(A)
    F = np.array(VT[-1]).reshape((3,3))

    # Make F into rank-2
    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2][2] = 0
    F = u @ s @ vt
    F = F / F[2][2]
    
    return F
 
def RANSAC(true_map, kp0, kp1, iterations, threshold):
    num_match = len(true_map)
    max_num_inliers = 0
    
    # Before we start the RANSAC, we nned to prepare all the coordinates of match points
    left_match_points = []
    right_match_points = []
    
    for i in range(len(true_map)):
        left_match_points.append(kp0[true_map[i][0]].pt)
        right_match_points.append(kp1[true_map[i][1]].pt)
    
    # Start RANSAC
    for i in range(iterations):
        inliers = 0
        
        # Randomly pick 8 match points
        pick_list = np.random.choice(num_match, 8, replace=False)
    
        # Accessing the coordinates
        left_points = []
        right_points = []

        for j in range(len(pick_list)):
            left_points.append(kp0[true_map[pick_list[j]][0]].pt)
            right_points.append(kp1[true_map[pick_list[j]][1]].pt)

        # Implement RANSAC algorithm
        F = Fundamental_F(left_points, right_points)

        # Computer how many inliers for this fundamental matrix (p'Fp = 0)
        for k in range(num_match):
            right = np.ones((1,3))
            left = np.ones((3,1))
            
            right[0][0] = right_match_points[k][0]
            right[0][1] = right_match_points[k][1]
            left[0][0] = left_match_points[k][0]
            left[1][0] = left_match_points[k][1]
            
            tmp = right @ F @ left
            
            if abs(tmp) < threshold:
                inliers += 1
            
        if inliers > max_num_inliers:
            max_num_inliers = inliers
            best_F = F
        
        #print(inliers)
    
    return best_F, left_match_points, right_match_points

if __name__ == '__main__':
    
    # Read input img and store them
    img_list = []
    img_gray_list = []
    for i in range(2):
        img_path = f"im{i}.png"
        tmp_img, tmp_img_gray = read_img(img_path)
        img_list.append(tmp_img)
        img_gray_list.append(tmp_img_gray)
    
    """
    # Test for writing
    for i in range(2):
        cv2.imwrite(f"output{i}.png", img_gray_list[i])
    """

    # SIFT
    kp0, des0 = Create_SIFT(img_gray_list[0])
    kp1, des1 = Create_SIFT(img_gray_list[1])

    """
    # Test for accessing coordinates of key points
    print(type(kp0))
    print(len(kp0))
    print(len(kp1))
    print(des0[0])
    print(kp0[0].pt)
    """
    
    # Feature Matching: KNN (2NN)
    dist_matrix, mapping = KNN(kp0, kp1, des0, des1)

    # Lowe's test for good matching
    true_map = Test_match(dist_matrix, mapping)
    print(true_map)
    
    
    # RANSAC for the best fundamental matrix
    iterations = 1000
    threshold = 0.02
    best_F, left_match_points, right_match_points = RANSAC(true_map, kp0, kp1, iterations, threshold)

    # Get the rectified matrix

    
    