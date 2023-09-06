import jpegio as jio
from utils.utils import *
from dahuffman import HuffmanCodec
import matplotlib.pyplot as plt 

def reshape(channel):
    h = channel.shape[0] // 8
    w = channel.shape[1] // 8
    return channel.reshape(h, 8, w, 8).transpose(0, 2, 1, 3)

def inverse_reshape(channel):
    h = channel.shape[0] * 8
    w = channel.shape[1] * 8
    return channel.transpose(0,2,1,3).reshape(h,w)

if __name__ == "__main__":

    img0 = jio.read("../datas/img0.jpg")
    img1 = jio.read("../datas/img1.jpg")

    print(dir(img1))
    print(img1.coef_arrays[0][:8,:8])
    # img1.coef_arrays[0][0,0] = 0
    for i in range(3):
        y0 = reshape(img0.coef_arrays[i])
        y1 = reshape(img1.coef_arrays[i])
        diffs = similar_res(y0, y1, 12)
        diffs = inverse_reshape(diffs)
        for j in range(diffs.shape[0]):
            for k in range(diffs.shape[1]):
                img1.coef_arrays[i][j,k] = diffs[j,k]
        print(np.std(y1), np.std(img1.coef_arrays[i]))
    jio.write(img1, "image_modified.jpg")
    test_j = jio.read("image_modified.jpg")
    print(test_j.coef_arrays[0][:8,:8])
    # plt.hist(y0.reshape(-1), bins = 100, range=(-100,100),label=f"std:{np.std(y0):.3f}",histtype="step")
    # plt.hist(diffs.reshape(-1), bins = 100, range=(-100,100),label=f"std:{np.std(diffs):.3f}",histtype="step")
    # # plt.yscale('log')
    # plt.legend()
    # plt.show()
    
    # codec = HuffmanCodec.from_data(list(y1))
    # encoded = codec.encode(list(y1))
    # print(f"Init Huffman Length : {len(encoded)}")
    # codec = HuffmanCodec.from_data(list(diffs))
    # encoded = codec.encode(list(diffs))
    # print(f"Init Huffman Length : {len(encoded)}")