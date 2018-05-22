import numpy as np
import numpy.random as npr

from model.bbox import bbox_transform, bbox_overlaps

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    num_images = 1
    rois_per_image = 256 / num_images
    fg_rois_per_image = np.round(0.25 * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image)

    rois = rois.reshape(-1, 6)
    roi_scores = roi_scores.reshape(-1)

    roi_scores = np.array(roi_scores,dtype=np.float32)

    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, 5 * 2)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, 5 * 2)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    print ('shape bbox_targets, bbox_inside_weights, bbox_outside_weights', bbox_targets.shape, bbox_inside_weights.shape, bbox_outside_weights.shape)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, th, tw, ta)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 5 * 2), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(5 * cls)
        end = start + 5
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    targets = bbox_transform(ex_rois, gt_rois)
    # print('targets : ', targets)

    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    print('In proposal_target_layer!!!')

    overlaps, delta_theta = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:6], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :5], dtype=np.float))
    # print ('overlaps' , overlaps.shape)
    print ('overlaps.max', np.max(overlaps))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = overlaps == gt_max_overlaps

    high_overlaps = overlaps > 0.5
    low_overlaps = overlaps < 0.3
    positive = np.where(np.logical_and(np.logical_or(gt_argmax_overlaps,high_overlaps),delta_theta < 15.0))[0]
    negative = np.where(np.logical_or(low_overlaps, np.logical_and(high_overlaps, delta_theta > 15.0)))[0]

    # print ('max_overlaps', max_overlaps, max_overlaps.shape)
    # np.set_printoptions(threshold=np.inf)
    # print('rois : ', all_rois[:, 1:6])
    # print('max_overlaps : ', max_overlaps)
    # print('gt : ', gt_boxes[:, :5])
    labels = np.ones(max_overlaps.shape[0], dtype=np.float32)
    print ('labels shape:' , labels.shape)
    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(max_overlaps >= 0.5)[0]
    fg_inds = positive


    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    # bg_inds = np.where((max_overlaps < 0.5) &
    #                     (max_overlaps >= 0.1))[0]
    bg_inds = negative


    print ('num_fg', fg_inds, 'num_bg', bg_inds)

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]
    print ('roi_size',rois.shape, 'all_rois_size', all_rois.shape)
    bbox_target_data = _compute_targets(
        rois[:, 1:6], gt_boxes[gt_assignment[keep_inds], :5], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
