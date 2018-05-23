from model.bbox import bbox_transform_inv_tf, clip_boxes_tf
import tensorflow as tf
import numpy as np
import cv2
import time
import math

def rotate_cpu_nms(dets, scores, threshold):
	'''
	Parameters
	----------------
	dets: (N, 6) --- x_ctr, y_ctr, height, width, angle, score
	threshold: 0.7 or 0.5 IoU
	----------------
	Returns
	----------------
	keep: keep the remaining index of dets
	'''
	max_size = 4000
	keep = []

	dets = np.round(dets, decimals=2)
	order = scores.argsort()[::-1]
	ndets = dets.shape[0]
	suppressed = np.zeros((ndets), dtype = np.int)
	ovr = 0.0

	for _i in range(max_size):
		i = order[_i]
		if suppressed[i] == 1:
			continue
		keep.append(i)
		r1 = ((dets[i,0],dets[i,1]),(dets[i,3],dets[i,2]),dets[i,4])
		area_r1 = dets[i,2]*dets[i,3]
		for _j in range(_i+1,max_size):
			j = order[_j]
			if suppressed[j] == 1:
				continue
			r2 = ((dets[j,0],dets[j,1]),(dets[j,3],dets[j,2]),dets[j,4])
			area_r2 = dets[j,2]*dets[j,3]

			n, int_pts = cv2.rotatedRectangleIntersection(r1, r2)
			if n == 1:
				order_pts = cv2.convexHull(int_pts, returnPoints = True)
				int_area = cv2.contourArea(order_pts)
				ovr = int_area / (area_r1+area_r2-int_area)
			elif n == 2:
				ovr = min(area_r1, area_r2) / max(area_r1, area_r2)

			if ovr >= threshold:
				suppressed[j] = 1

	return np.array(keep, dtype=np.int32)


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
    # Get the scores and bounding boxes
	scores = rpn_cls_prob[:, :, :, num_anchors:]
	scores = tf.reshape(scores, shape=(-1,))
	rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 5))

	proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
	proposals = clip_boxes_tf(proposals, im_info[:2])

    # Non-maximal suppression
	threshold = 0.5
	indices = tf.py_func(rotate_cpu_nms,[proposals, scores, threshold],tf.int32,stateful=False,name=None)

	boxes = tf.gather(proposals, indices)
	boxes = tf.to_float(boxes)
	scores = tf.gather(scores, indices)
	scores = tf.reshape(scores, shape=(-1, 1))
	scores = tf.to_float(scores)

	# Only support single image as input
	batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
	blob = tf.concat([batch_inds, boxes], 1)

	return blob, scores

if __name__ == "__main__":

	boxes = np.array([
			[50, 50, 100, 100, 0],
			[60, 60, 100, 100, 0],#keep 0.68
			[50, 50, 100, 100, 45.0],#discard 0.70
			[200, 200, 100, 100, 0],#keep 0.0

		])
	scores = np.array([0.99, 0.88, 0.66, 0.77])

	#boxes = np.tile(boxes, (4500 / 4, 1))

	#for ind in range(4500):
	#	boxes[ind, 5] = 0

	a = rotate_cpu_nms(boxes, scores, 0.7)

	print (boxes[a])
