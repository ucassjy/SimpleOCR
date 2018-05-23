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

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    targets_da = gt_angle - ex_angle

    targets_da[np.where(targets_da < -np.pi / 4)] += np.pi
    targets_da[np.where(targets_da >= np.pi * (3.0/4))] -= np.pi

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

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)
    pred_a = tf.add(angle, da)

    a_up = tf.add(pred_a, np.pi)
    a_down = tf.subtract(pred_a, np.pi)
    pred_a = tf.where(pred_a >= np.pi * (3.0/4), pred_a, a_down)
    pred_a = tf.where(pred_a < -np.pi / 4, pred_a, a_up)

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

def bbox_overlaps(boxes, query_boxes):
    """ Calculate IoU(intersection-over-union) and angle difference for each input boxes and query_boxes. """

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    boxes = np.round(boxes, decimals=2)
    query_boxes = np.round(query_boxes, decimals=2)
    overlaps = np.reshape(np.zeros((N, K)), (N,K))
    delta_theta = np.reshape(np.zeros((N, K)), (N,K))

    for k in range(K):
        rect1 = ((query_boxes[k][0], query_boxes[k][1]),
                 (query_boxes[k][2], query_boxes[k][3]),
                 query_boxes[k][4])
        for n in range(N):
            rect2 = ((boxes[n][0], boxes[n][1]),
                     (boxes[n][2], boxes[n][3]),
                     boxes[n][4])
            # can check official document of opencv for details
            try:
                num_int, points = cv2.rotatedRectangleIntersection(rect1, rect2)
            except :
                num_int = 0
                overlaps[n][k] = 0.9
            S1 = query_boxes[k][2] * query_boxes[k][3]
            S2 = boxes[n][2] * boxes[n][3]
            if num_int == 1 and len(points) > 2:
                s = cv2.contourArea(cv2.convexHull(points,returnPoints=True))
                overlaps[n][k] = s / (S1 + S2 - s)
            elif num_int == 2:
                overlaps[n][k] = min(S1, S2) / max(S1, S2)
            delta_theta[n][k] = np.abs(query_boxes[k][4] - boxes[n][4])
    return overlaps, delta_theta
