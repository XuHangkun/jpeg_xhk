import numpy as np
from numba import njit
from tqdm import tqdm

@njit
def cal_motion_vector(ref, tag, coord, window_length = 12, step = 1, is_inter = False):

    def within_bound(i,j,h,w):
        if i < 0 or i >= h:
            return False
        if j < 0 or j >= w:
            return False
        return True

    def before(i,j,th,tw):
        if i < 0 or i > th:
            return False
        if j < 0 or j >= tw:
            return False
        return True
    
    h, w = ref.shape[0], ref.shape[1]
    if not within_bound(coord[0], coord[1], h, w):
        return None, None
    min_loss = None
    motion_vector = None
    for i in range(coord[0] - window_length, coord[0] + window_length, step):
        for j in range(coord[1] - window_length, coord[1] + window_length, step):
            if not within_bound(i,j,h,w):
                continue
            if is_inter and not before(i,j,coord[0],coord[1]):
                continue
            loss = np.mean(np.abs(ref[i,j] - tag))
            if min_loss is not None:
                if loss < min_loss:
                    min_loss = loss
                    motion_vector = [i-coord[0], j-coord[1]]
            else:
                min_loss = loss
                motion_vector = [i-coord[0], j-coord[1]]
    return min_loss, motion_vector

def similar_res(ref, tag, window = 12, step = 1):

    h,w = tag.shape[0], tag.shape[1]
    diffs = np.zeros_like(tag)
    for i in tqdm(range(h)):
        for j in range(w):
            min_loss, motion_vector = cal_motion_vector(ref, tag[i,j], [i,j], window_length=window, step=step)
            inter_min_loss, inter_motion_vector = cal_motion_vector(tag, tag[i,j], [i,j], window_length=window, step=step, is_inter=True)
            if inter_min_loss and inter_min_loss < min_loss:
                motion_vector = inter_motion_vector
                diff = tag[i+motion_vector[0], j+motion_vector[1]] - tag[i,j]
            else:
                diff = ref[i+motion_vector[0], j+motion_vector[1]] - tag[i,j]
            diffs[i,j] = diff
    return diffs