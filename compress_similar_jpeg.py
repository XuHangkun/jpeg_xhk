import jpegio as jio
from utils.utils import *
from dahuffman import HuffmanCodec
import matplotlib.pyplot as plt 
import pickle as pkl

def reshape(channel):
    h = channel.shape[0] // 8
    w = channel.shape[1] // 8
    return channel.reshape(h, 8, w, 8).transpose(0, 2, 1, 3)

def inverse_reshape(channel):
    h = channel.shape[0] * 8
    w = channel.shape[1] * 8
    return channel.transpose(0,2,1,3).reshape(h,w)

def compress():
    img0 = jio.read("../datas/img0.jpg")
    img1 = jio.read("../datas/img1.jpg")

    jpeg_zip = JPEGCompression()

    print(dir(img1))
    print(img1.coef_arrays[0][:8,:8])

    infos = {
        "diff_coefs" : [],
        "motion_vectors" : []
    }
    for i in range(3):
        y0 = reshape(img0.coef_arrays[i])
        y1 = reshape(img1.coef_arrays[i])
        diffs, motion_vectors = jpeg_zip.similar_res(y0, y1, window=8, contain_inter=False)
        diffs = inverse_reshape(diffs)
        infos["diff_coefs"].append(diffs)
        infos["motion_vectors"].append(motion_vectors)

    with open("diff_infos.pkl", "wb") as f:
        pkl.dump(infos, f)
    
    for i in range(3):
        for j in range(infos["diff_coefs"][i].shape[0]):
            for k in range(infos["diff_coefs"][i].shape[1]):
                img1.coef_arrays[i][j,k] = infos["diff_coefs"][i][j,k]
    jio.write(img1, "img_modified.jpg")

def analysis():
    import matplotlib.pyplot as plt
    with open("diff_coef.pkl", "rb") as f:
        diff_coef = pkl.load(f)
    img1 = jio.read("../datas/img1.jpg")

    plt.figure()
    # plt.hist(img1.coef_arrays[0].reshape(-1), bins = 100,range=(-1000,1000), histtype="step",label="init")
    # plt.hist(diff_coef[0].reshape(-1),bins=100, range=(-1000,1000), histtype="step",label="after")
    # obj = img1.coef_arrays[0]
    for obj, label in zip([img1.coef_arrays[0],diff_coef[0]], ["Init", "Diff"]):
    # for i in range(3):
    #     m,n = np.random.randint(obj.shape[0]//8), np.random.randint(obj.shape[1]//8)
    #     diff_coef_mean = zigzag(reshape(obj)[m,n])
    #     # diff_coef_std = zigzag(reshape(diff_coef[0])[m,n]
    #     plt.plot(np.arange(63), diff_coef_mean[1:])
        diff_coef_mean = zigzag(np.mean(reshape(obj),axis=(0,1)))
        diff_coef_std = zigzag(np.std(reshape(obj), axis=(0,1)))
        plt.errorbar(np.arange(63)+1, diff_coef_mean[1:], diff_coef_std[1:], label = label,fmt="o")
    plt.ylabel("AC Mean/Std")
    plt.xlabel("AC Index")
    plt.legend()
    # plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # compress()
    analysis()