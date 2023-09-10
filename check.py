import pickle as pkl
import numpy as np
import jpegio as jio

def reshape(channel):
    h = channel.shape[0] // 8
    w = channel.shape[1] // 8
    return channel.reshape(h, 8, w, 8).transpose(0, 2, 1, 3)

def inverse_reshape(channel):
    h = channel.shape[0] * 8
    w = channel.shape[1] * 8
    return channel.transpose(0,2,1,3).reshape(h,w)

if __name__ == "__main__":

    f = open("diff_infos.pkl", "rb")
    infos = pkl.load(f)

    mf = open("motion_vectors.pkl", "wb")
    pkl.dump(infos["motion_vectors"], mf)

    motion_vector = infos["motion_vectors"][0][8,8]
    print(motion_vector)
    diff = reshape(infos["diff_coefs"][0])

    img0 = jio.read("../datas/img0.jpg")
    img1 = jio.read("../datas/img1.jpg")

    init = img0.coef_arrays[0]
    i,j = np.int8(motion_vector[0] // 8), np.int8(motion_vector[1] // 8)
    m,n = np.int8(motion_vector[0] % 8), np.int8(motion_vector[1] % 8)
    init_dcts = reshape(img0.coef_arrays[0])
    print(i,j,m,n)
    print(init_dcts.shape)
    tref = ((8 - m) * (8 - n) * init_dcts[8 + i, 8 + j] + (8 - m) * n * init_dcts[8 + i, 8 + j + 1] + \
        m * (8 - n) * init_dcts[8 + i + 1, 8 + j] + m * n * init_dcts[8 + i + 1, 8 + j + 1])/64.
    tref = np.round(tref).astype(diff.dtype)
    recover = tref - diff[8,8]
    print(recover)    
    print(diff[8,8])

    tag_dcts = reshape(img1.coef_arrays[0])
    print(tag_dcts[8,8] - recover)
