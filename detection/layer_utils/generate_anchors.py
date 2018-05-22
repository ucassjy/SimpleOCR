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

def generate_anchors():
    """
    Generate anchor windows by enumerating aspect ratios, scales and angles.

    Input : height and width of the feature map
    Output : list contains np.array (x, y, h, w, theta), defined same as in readfile.py.
    """

    ratios = [0.125, 0.2, 0.5]
    scales = [256*8, 256*16, 256*32]
    angles = [-30.0, 0.0, 30.0, 60.0, 90.0, 120.0]

    hws = _ratio_scales(ratios, scales)
    angles = np.array([[np.array(a)] for a in angles])

    anchors = [np.concatenate(((0, 0), hws[j], angles[k])) for j in range(len(hws))
                                                           for k in range(6)]

    return np.array(anchors)

# def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
#                      scales=2 ** np.arange(3, 6)):
#   """
#   Generate anchor (reference) windows by enumerating aspect ratios X
#   scales wrt a reference (0, 0, 15, 15) window.
#   """
#
#   base_anchor = np.array([1, 1, base_size, base_size]) - 1
#   ratio_anchors = _ratio_enum(base_anchor, ratios)
#   anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
#                        for i in range(ratio_anchors.shape[0])])
#   return anchors
#
#
# def _whctrs(anchor):
#   """
#   Return width, height, x center, and y center for an anchor (window).
#   """
#
#   w = anchor[2] - anchor[0] + 1
#   h = anchor[3] - anchor[1] + 1
#   x_ctr = anchor[0] + 0.5 * (w - 1)
#   y_ctr = anchor[1] + 0.5 * (h - 1)
#   return w, h, x_ctr, y_ctr
#
#
# def _mkanchors(ws, hs, x_ctr, y_ctr):
#   """
#   Given a vector of widths (ws) and heights (hs) around a center
#   (x_ctr, y_ctr), output a set of anchors (windows).
#   """
#
#   ws = ws[:, np.newaxis]
#   hs = hs[:, np.newaxis]
#   anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
#                        y_ctr - 0.5 * (hs - 1),
#                        x_ctr + 0.5 * (ws - 1),
#                        y_ctr + 0.5 * (hs - 1)))
#   return anchors
#
#
# def _ratio_enum(anchor, ratios):
#   """
#   Enumerate a set of anchors for each aspect ratio wrt an anchor.
#   """
#
#   w, h, x_ctr, y_ctr = _whctrs(anchor)
#   size = w * h
#   size_ratios = size / ratios
#   ws = np.round(np.sqrt(size_ratios))
#   hs = np.round(ws * ratios)
#   anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
#   return anchors
#
#
# def _scale_enum(anchor, scales):
#   """
#   Enumerate a set of anchors for each scale wrt an anchor.
#   """
#
#   w, h, x_ctr, y_ctr = _whctrs(anchor)
#   ws = w * scales
#   hs = h * scales
#   anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
#   return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a, len(a))
