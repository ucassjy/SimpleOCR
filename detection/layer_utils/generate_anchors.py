import numpy as np

def _ratio_scales(ratios, scales):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    ss = np.repeat(scales, 3)
    rs = np.tile(ratios, 3)
    size_ratios = ss * rs

    hs = np.sqrt(size_ratios)
    ws = hs / rs
    hs = np.array([[np.array(h)] for h in hs])
    ws = np.array([[np.array(w)] for w in ws])

    hws = np.concatenate((hs, ws), axis=1)
    return hws

def generate_anchors(height, width):
    """
    Generate anchor windows by enumerating aspect ratios, scales and angles.

    Input : height and width of the feature map
    Output : list contains np.array (x, y, h, w, theta), defined same as in readfile.py.
    """

    ratios = [0.125, 0.2, 0.5]
    scales = [256*8, 256*16, 256*32]
    angles = [-30.0, 0.0, 30.0, 60.0, 90.0, 120.0]
    xs = np.arange(height) * 16
    ys = np.arange(width) * 16

    xys = np.transpose([np.tile(xs, ys.shape[0]), np.repeat(ys, xs.shape[0])])

    hws = _ratio_scales(ratios, scales)
    angles = np.array([[np.array(a)] for a in angles])

    anchors = [np.concatenate((xys[i], hws[j], angles[k])) for i in range(len(xys))
														   for j in range(len(hws))
                                                           for k in range(6)]
    return np.array(anchors, dtype=np.float32)

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    import time
    t = time.time()
    a = generate_anchors(3,5)
    print(time.time() - t)
    print(a, a.shape[0])
