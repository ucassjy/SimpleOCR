import os
import numpy as np
import numpy.random as npr

from model.bbox import bbox_transform, bbox_overlaps

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    # all_anchors(x,y,h,w,theta)
    inds_inside = np.where(
        (all_anchors[:, 1] + (all_anchors[:, 3] * np.abs(np.sin(all_anchors[:, 4])) + all_anchors[:, 2] * np.abs(np.cos(all_anchors[:, 4]))) / 2 <= im_info[0] + _allowed_border) &
        (all_anchors[:, 0] + (all_anchors[:, 3] * np.abs(np.cos(all_anchors[:, 4])) + all_anchors[:, 2] * np.abs(np.sin(all_anchors[:, 4]))) / 2 <= im_info[1] + _allowed_border) &
        (all_anchors[:, 1] - (all_anchors[:, 3] * np.abs(np.sin(all_anchors[:, 4])) + all_anchors[:, 2] * np.abs(np.cos(all_anchors[:, 4]))) / 2 >= -_allowed_border) &
        (all_anchors[:, 0] - (all_anchors[:, 3] * np.abs(np.cos(all_anchors[:, 4])) + all_anchors[:, 2] * np.abs(np.sin(all_anchors[:, 4]))) / 2 >= -_allowed_border)
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    overlaps, delta_theta = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    argmax_overlaps = overlaps.argmax(axis=1) # in range(num_gt_boxes)
    # Get the max overlaps among all gt_boxes for each anchor in image
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # len = anchors

    # The anchors of the highest overlap with respect to certain gt_boxes
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])] # len = gt_boxes
    # Truth table
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # Truth table of anchors with IoU > 0.7
    high_overlaps = overlaps > 0.7

    # Positive anchors: anchors with highest IoU w.r.t a gt_box or anchors with IoU > 0.7  and delta_theta < pi/12
    positive_case1 = gt_argmax_overlaps
    positive_case2 = np.where(np.logical_and(high_overlaps,delta_theta < 15.0))[0]

    # positive = np.where(np.logical_and(high_overlaps,delta_theta < 15.0))[0]

    # Negative anchors:
    negative_case1 = max_overlaps < 0.3
    negative_case2 = np.where(np.logical_and(high_overlaps, delta_theta > 15.0))[0]

    # Labeling the anchors

    labels[negative_case1] = 0
    labels[negative_case2] = 0
    labels[positive_case1] = 1
    labels[positive_case2] = 1

    # subsample positive labels if we have too many
    num_fg = 128 # fg = foreground
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = 256 - np.sum(labels == 1) #bg = background
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    print('fg and bg in anchor_target_layer : ',
            len(fg_inds) if len(fg_inds) < 128 else 128,
            256 - len(fg_inds) if len(fg_inds) < 128 else 128)

    bbox_targets = np.zeros((len(inds_inside), 5), dtype=np.float32)
    bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :]).astype(np.float32, copy=False)
    bbox_inside_weights = np.zeros((len(inds_inside), 5), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0, 1.0))
    bbox_outside_weights = np.zeros((len(inds_inside), 5), dtype=np.float32)

    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    positive_weights =  np.ones((1, 5)) * 1.0 / num_examples
    negative_weights = np.zeros((1, 5)) * 1.0 / num_examples
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A * 5))
    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 5))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 5))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
