import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans
from skimage.feature import peak_local_max

#---------------------------------------------------------------------------

def error_papillae(gt, pred, TH):
    coordinates_gt = peak_local_max(gt, min_distance=5, exclude_border=False)#lala ha messo 10, raffaella 5
    coordinates_pred = peak_local_max(pred, min_distance=5, exclude_border=False)#lala ha messo 10, raffaella 5/!\ size spots forse grandi
    N = len(coordinates_gt)
    dist_TP = []
    TP_gt = []
    FN = []
    for x_gt, y_gt in zip(coordinates_gt[:, 1], coordinates_gt[:, 0]):
        dist = []
        for x_pred, y_pred in zip(coordinates_pred[:, 1], coordinates_pred[:, 0]):
            dist.append(np.linalg.norm(np.array([x_gt, y_gt])-np.array([x_pred, y_pred])))
        arr_dist = np.array(dist)
        if len(arr_dist[arr_dist<TH])>0:
            dist_TP.append(min(arr_dist[arr_dist<TH]))
            TP_gt.append([x_gt, y_gt])
        else:
            FN.append([x_gt, y_gt])
    dist_TP_bis = []
    TP_pred = []
    FP = []
    for x_pred, y_pred in zip(coordinates_pred[:, 1], coordinates_pred[:, 0]):
        dist = []
        for x_gt, y_gt in zip(coordinates_gt[:, 1], coordinates_gt[:, 0]):
            dist.append(np.linalg.norm(np.array([x_pred, y_pred])-np.array([x_gt, y_gt])))
        arr_dist = np.array(dist)
        if len(arr_dist[arr_dist<TH])>0:
            dist_TP_bis.append(min(arr_dist[arr_dist<TH]))
            TP_pred.append([x_pred, y_pred])
        else:
            FP.append([x_pred, y_pred])
    return dist_TP, TP_gt, TP_pred, FN, FP, N

def create_circle_array(shape, centers, radius):
    array = np.zeros(shape, dtype=int)
    y_indices, x_indices = np.indices(shape)
    for center in centers:
        cx, cy = center
        distance = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        array[distance <= radius] = 1  
    return array

def dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return np.mean( (2. * intersection) / (union))

def error_dice(gt, pred, radius):
    coordinates_gt = peak_local_max(gt, min_distance=5, exclude_border=False)
    coordinates_pred = peak_local_max(pred, min_distance=5, exclude_border=False)
    gt_array = create_circle_array(gt.shape, coordinates_gt, radius)
    pred_array = create_circle_array(pred.shape, coordinates_pred, radius)
    return dice(gt_array, pred_array)
