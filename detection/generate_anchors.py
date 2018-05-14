import math
import itertools
import numpy as np

def _ratio_scales(ratios, scales):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    ss = np.repeat(scales, 3)
    rs = np.tile(ratios, 3)
    size_ratios = ss / rs

    hs = np.sqrt(size_ratios)
    ws = hs * rs
    hs = np.array([[np.array(h)] for h in hs])
    ws = np.array([[np.array(w)] for w in ws])

    hws = np.concatenate((hs, ws), axis=1)
    return hws

def generate_anchors(xmax, ymax):
    """
    Generate anchor windows by enumerating aspect ratios, scales and angles.

    Input : height and width of the feature map
    Output : list contains np.array (x, y, h, w, theta), defined same as in readfile.py.
    """

    ratios = np.array([2, 5, 8])
    scales = np.array([8, 16, 32])
    angles = [-math.pi/6, 0.0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3]
    xs = np.arange(0, xmax)
    ys = np.arange(0, ymax)

    xys = np.transpose([np.tile(xs, ymax), np.repeat(ys, xmax)])
    hws = _ratio_scales(ratios, scales)
    angles = np.array([[np.array(a)] for a in angles])

    anchors = [np.concatenate((xys[i], hws[j], angles[k])) for i in range(len(xys))
                                                           for j in range(len(hws))
                                                           for k in range(6)]
    return anchors

if __name__ == '__main__':
    a = generate_anchors(10, 10)
    print(a, len(a))
