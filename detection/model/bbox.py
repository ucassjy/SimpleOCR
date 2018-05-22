import numpy as np
import tensorflow as tf
import cv2

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 3]
    ex_heights = ex_rois[:, 2]
    ex_ctr_x = ex_rois[:, 0]
    ex_ctr_y = ex_rois[:, 1]
    ex_angle = ex_rois[:, 4]

    gt_widths = gt_rois[:, 3]
    gt_heights = gt_rois[:, 2]
    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]
    gt_angle = gt_rois[:, 4]

    # print('ex_widths :', ex_widths)
    # print('ex_heights:', ex_heights)
    # print('gt_widths :', gt_widths)
    # print('gt_heights:', gt_heights)
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    targets_da = gt_angle - ex_angle

    targets_da[np.where(targets_da < -30)] += 180
    targets_da[np.where(targets_da >= 120)] -= 180

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_da)).transpose()
    return targets

def bbox_transform_inv_tf(boxes, deltas):
    boxes = tf.cast(boxes, deltas.dtype)
    widths = boxes[:, 3]
    heights = boxes[:, 2]
    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1]
    angle = boxes[:, 4]

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dh = deltas[:, 2]
    dw = deltas[:, 3]
    da = deltas[:, 4]

    # print ()


    # print ('angles.shape',angle.shape)
    # print ('da.shape',da.shape)
    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)
    pred_a = tf.add(angle, da)

	# for i in range(int(pred_a.shape[0])):
	# 	for j in range(int(pred_a.shape[1]))
	# pred_a
    a_up = tf.add(pred_a, 180)
    print("pred_a,a_up",pred_a,a_up)
    a_down = tf.subtract(pred_a, 180)
    pred_a = tf.where(pred_a >= 135, pred_a, a_down)
    pred_a = tf.where(pred_a < -45, pred_a, a_up)

    return tf.stack([pred_ctr_x, pred_ctr_y, pred_h, pred_w, pred_a], axis=1)

def clip_boxes_tf(boxes, im_info):
    """ Clip boxes out of the image into image. """

    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    h = boxes[:, 2]
    w = boxes[:, 3]
    angle = boxes[:, 4]

    sin_abs = tf.abs(tf.sin(angle))
    cos_abs = tf.abs(tf.cos(angle))
    y_top = y_center + (w * sin_abs + h * cos_abs) / 2
    y_bot = y_center - (w * sin_abs + h * cos_abs) / 2
    x_top = x_center + (w * cos_abs + h * sin_abs) / 2
    x_bot = x_center - (w * cos_abs + h * sin_abs) / 2

    x_bot = tf.maximum(x_bot, 0)
    y_bot = tf.maximum(y_bot, 0)
    x_top = tf.minimum(x_top, im_info[1] - 1)
    y_top = tf.minimum(y_top, im_info[0] - 1)

    h_new = (tf.multiply(cos_abs, y_top - y_bot) - tf.multiply(sin_abs, x_top - x_bot)) / (tf.pow(cos_abs, 2) - tf.pow(sin_abs, 2))
    w_new = tf.multiply(w, h_new) / h
    tf.Assert(tf.greater_equal(tf.reduce_min(h_new), 0.), [h_new])
    tf.Assert(tf.greater_equal(tf.reduce_min(w_new), 0.), [w_new])
    return tf.stack([x_center, y_center, h_new, w_new, angle], axis=1)

def clockwise_sort(points):
    """ Sort input points clockwise. """

    if len(points) <= 3:
        return points
    # print ('len points', len(points))
    l = np.argmin(points[:, 0])
    # p_flag is one of the points at the leftmost
    p_flag = points[l]
    # print ('p_flag.shape=',p_flag.shape)
    # separate points above p_flag and below p_flag
    above = np.array([p for p in points if p[1] >= p_flag[1]])
    below = np.array([p for p in points if p[1] < p_flag[1]])
    # sort 'above' and 'below' separately by x coordinate
    if above.shape[0] == 0:
        points = below[below[:, 0].argsort()][::-1]
    elif below.shape[0] == 0:
        points = above[above[:, 0].argsort()]
    else:
        above = above[above[:, 0].argsort()]
        below = below[below[:, 0].argsort()][::-1]
        points = np.concatenate((above, below))
    return points

def bbox_overlaps(boxes, query_boxes):
    """ Calculate IoU(intersection-over-union) and angle difference for each input boxes and query_boxes. """

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.reshape(np.zeros((N, K)), (N,K))
    delta_theta = np.reshape(np.zeros((N, K)), (N,K))

    for k in range(K):
        # print (k)
        rect1 = ((query_boxes[k][0], query_boxes[k][1]),
                 (query_boxes[k][2], query_boxes[k][3]),
                 query_boxes[k][4])
        for n in range(N):
            rect2 = ((boxes[n][0], boxes[n][1]),
                     (boxes[n][2], boxes[n][3]),
                     boxes[n][4])
            # can check official document of opencv for details
            # print (rect1,rect2)
            num_int, points = cv2.rotatedRectangleIntersection(rect1, rect2)
            S1 = query_boxes[k][2] * query_boxes[k][3]
            S2 = boxes[n][2] * boxes[n][3]
            if num_int == 1 and len(points) > 2:
                # points = clockwise_sort(points[:,0])
                s = cv2.contourArea(cv2.convexHull(points,returnPoints=True))
                overlaps[n][k] = s / (S1 + S2 - s)
            elif num_int == 2:
                overlaps[n][k] = min(S1, S2) / max(S1, S2)
            delta_theta[n][k] = np.abs(query_boxes[k][4] - boxes[n][4])
            # print (overlaps[n][k])
    return overlaps, delta_theta
