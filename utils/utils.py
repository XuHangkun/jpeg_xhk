import numpy as np
from numba import njit
from tqdm import tqdm
from copy import deepcopy

@njit
def cal_motion_vector(ref, tag, coord, window_length = 16, step = 1, is_inter = False, weight = None):

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
    
    min_loss = None
    motion_vector = None
    diff = None
    h, w = ref.shape[0], ref.shape[1]
    if not within_bound(coord[0], coord[1], h, w):
        return min_loss, motion_vector, diff
    
    for i in range(coord[0] - window_length, coord[0] + window_length, step):
        for j in range(coord[1] - window_length, coord[1] + window_length, step):
            for m in range(7):
                for n in range(7):
                    max_row = i * 8 + 7 + m
                    max_column = j * 8 + 7 + n
                    if not within_bound(max_row, max_column,h * 8, w * 8):
                        continue
                    if is_inter and not before(i,j,coord[0],coord[1]):
                        continue
                    # calculate ref !
                    tref = None
                    if m == 0 and n ==0:
                        tref = (8 - m) * (8 - n) * ref[i,j] / 64.
                    elif m == 0 and n != 0:
                        tref = ((8 - m) * (8 - n) * ref[i,j] + (8 - m) * n * ref[i, j+1])/64.
                    elif m != 0 and n == 0:
                        tref = ((8 - m) * (8 - n) * ref[i,j] + m * (8 - n) * ref[i+1,j])/64.
                    else:
                        tref = ((8 - m) * (8 - n) * ref[i,j] + (8 - m) * n * ref[i, j+1] + m * (8 - n) * ref[i+1,j] + m * n * ref[i+1, j+1])/64.
                    tmp_diff = np.round(tref)
                    tmp_diff = tmp_diff - tag
                    if weight is not None:
                        loss = np.mean(np.abs(tmp_diff) * weight)
                    else:
                        loss = np.mean(np.abs(tmp_diff))
                    if min_loss is not None:
                        if loss < min_loss:
                            min_loss = loss
                            motion_vector = [i*8 + m - coord[0]*8, j*8 + n-coord[1]*8]
                            diff = (tmp_diff).astype(ref.dtype)
                    else:
                        min_loss = loss
                        motion_vector = [i*8 + m - coord[0]*8, j*8 + n-coord[1]*8]
                        diff = (tmp_diff).astype(ref.dtype)
    return min_loss, motion_vector, diff

class JPEGCompression:

    def __init__(self) -> None:
        
        self.weight = None
        # self.weight = self.cal_weight()
        # print("JPEG Compression Weight : \n", self.weight)

    
    def cal_weight(self):

        factor = -1
        A = [[1,1,1,1,1,0,0,0]]
        for i in range(7):
            B = deepcopy(A[-1])
            B = B[1:] + [0]
            A.append(B)
        return np.array(A)

    def similar_res(self, ref, tag, window = 16, step = 1, contain_inter = False):
        h,w = tag.shape[0], tag.shape[1]
        diffs = np.zeros_like(tag)
        motion_vectors_shape = list(tag.shape)[:2] + [2]
        motion_vectors = np.zeros(motion_vectors_shape)
        for i in tqdm(range(h)):
            for j in range(w):
                min_loss, motion_vector, s_diff = cal_motion_vector(ref, tag[i,j], [i,j], window_length=window, step=step, weight=self.weight)
                if contain_inter:
                    inter_min_loss, inter_motion_vector, i_diff = cal_motion_vector(tag, tag[i,j], [i,j], window_length=window, step=step, is_inter=True, weight=self.weight)
                    if inter_min_loss and inter_min_loss < min_loss:
                        motion_vector = inter_motion_vector
                        # diff = tag[i+motion_vector[0], j+motion_vector[1]] - tag[i,j]
                        diff = i_diff
                    else:
                        # diff = ref[i+motion_vector[0], j+motion_vector[1]] - tag[i,j]
                        diff = s_diff
                else:
                    # diff = ref[i+motion_vector[0], j+motion_vector[1]] - tag[i,j]
                    diff = s_diff
                diffs[i,j] = diff
                motion_vectors[i,j] = np.array(motion_vector).astype(np.int8)
        return diffs, motion_vectors

def zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    computes the zigzag of a quantized block
    :param numpy.ndarray matrix: quantized matrix
    :returns: zigzag vectors in an array
    """
    # initializing the variables
    h = 0
    v = 0
    v_min = 0
    h_min = 0
    v_max = matrix.shape[0]
    h_max = matrix.shape[1]
    i = 0
    output = np.zeros((v_max * h_max))

    while (v < v_max) and (h < h_max):
        if ((h + v) % 2) == 0:  # going up
            if v == v_min:
                output[i] = matrix[v, h]  # first line
                if h == h_max:
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif (h == h_max - 1) and (v < v_max):  # last column
                output[i] = matrix[v, h]
                v = v + 1
                i = i + 1
            elif (v > v_min) and (h < h_max - 1):  # all other cases
                output[i] = matrix[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if (v == v_max - 1) and (h <= h_max - 1):  # last line
                output[i] = matrix[v, h]
                h = h + 1
                i = i + 1
            elif h == h_min:  # first column
                output[i] = matrix[v, h]
                if v == v_max - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif (v < v_max - 1) and (h > h_min):  # all other cases
                output[i] = matrix[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
        if (v == v_max - 1) and (h == h_max - 1):  # bottom right element
            output[i] = matrix[v, h]
            break
    return output

if __name__ == "__main__":
    obj_comp = JPEGCompression()
    print(zigzag(obj_comp.weight))