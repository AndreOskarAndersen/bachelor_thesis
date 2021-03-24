import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.filters
import skimage.transform
import skimage.draw
import os
from tqdm.notebook import tqdm
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def create_heatmaps(keypoints, input_shape = (256, 256), output_shape = (64, 64)):
	""" GIVEN ONE OF THE ROWS IN THE CSV, CREATES THE 17 CORRESPONDING HEATMAPS """
	keypoints = keypoints[0, 1:] # We dont need the ID of the image

	x_val = keypoints[::3]
	y_val = keypoints[1::3]
	v_val = keypoints[2::3]

	res_arr = []

	for x, y, v in zip(x_val, y_val, v_val):
		try:
			x, y, v = int(x), int(y), int(v)
		except:
			x, y, v = int(float(x)), int(float(y)), int(float(v))
		res = np.zeros(input_shape)

		if (x >= 0 and x <= input_shape[1] and y >= 0 and y <= input_shape[0]):
			res[y - 1, x - 1] = 1

		res = skimage.transform.resize(res, output_shape)
		
		if (v == 1):
			res = skimage.filters.gaussian(res, sigma = 1)
		elif (v == 2):
			res = skimage.filters.gaussian(res, sigma = 0.5)

		res_arr.append(res)

	return res_arr

def draw_keypoints(I, keypoints, r = 0.75, g = 0.6, b = 0.2, radius = 2):
	"""Draws keypoints on image"""
	I_copy = np.copy(I)
	
	keypoints = list(map(int, keypoints))
 
	x_val = keypoints[::3]
	y_val = keypoints[1::3]
	v_val = keypoints[2::3]
 

	for x, y, v in zip(x_val, y_val, v_val):
		if (v != 0):
			rr, cc = skimage.draw.circle(y, x, radius)
			I_copy[rr - 1, cc - 1] = [r, g, b]

	return I_copy

def draw_bbox(I, bbox, r, g, b):
	I_copy = np.copy(I)

	x_min = np.ceil(bbox[0]).astype("int")
	x_max = np.floor(bbox[0] + bbox[2]).astype("int")
	y_min = np.ceil(bbox[1]).astype("int")
	y_max = np.floor(bbox[1] + bbox[3]).astype("int")

	rr, cc = skimage.draw.line(y_min, x_min, y_min, x_max)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	rr, cc = skimage.draw.line(y_max, x_min, y_max, x_max)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	rr, cc = skimage.draw.line(y_min, x_min, y_max, x_min)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	rr, cc = skimage.draw.line(y_min, x_max, y_max, x_max)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	return I_copy

def draw_segmentation_extremes(I, segmentation, r, g, b, radius):
	assert len(segmentation) == 1

	I_copy = np.copy(I)
	segmentation = np.array(segmentation[0]).reshape((-1, 2))
	x_min, y_min = np.min(segmentation, axis = 0)[0], np.min(segmentation, axis = 0)[1]
	x_max, y_max = np.max(segmentation, axis = 0)[0], np.max(segmentation, axis = 0)[1]

	rr, cc = skimage.draw.circle(y_min, x_min, radius)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	rr, cc = skimage.draw.circle(y_max, x_min, radius)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	rr, cc = skimage.draw.circle(y_min, x_max, radius)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	rr, cc = skimage.draw.circle(y_max, x_max, radius)
	I_copy[rr - 1, cc - 1] = [r, g, b]

	return I_copy

def grey_to_rgb(img):
    """ Transforms the img from grey-scale to rgb """
    return np.stack((img,)*3, axis = -1)
  
def get_mean_rgb(dir_path, img_names):
	""" returns the average rgb vector """
	imgs_names = os.listdir(dir_path)
	N = len(img_names)
	average_rgb = 0
	
	for img_name in tqdm(img_names):
		img = plt.imread(dir_path + img_name)
		
		if (len(img.shape) == 2):
			rgb_img = np.stack((img,)*3, axis = -1).reshape((-1, 3))
			average_rgb += np.mean(rgb_img.reshape((-1, 3)), axis = 0)/N
		
		else:
			average_rgb += np.mean(img.reshape((-1, 3)), axis = 0)/N
		
	return average_rgb

def turn_featuremaps_to_keypoints(feature_maps):
	""" Turns the 17 featuremaps into an array of size 51 of keypoints """
	arr = []
	for feature_map in feature_maps:
		index = np.unravel_index(feature_map.argmax(), feature_map.shape)
		for el in index:
			arr.append(el)

		arr.append(2)
  
	return arr

def draw_predicitions_and_gt(img, gt, pred):
	""" Draws the 17 true keypoints in red and the 17 predicted featuremaps in green onto the image given by img"""

	img = draw_keypoints(img, gt, r = 0, g = 1, b = 0)

	pred_keypoints = np.array(turn_featuremaps_to_keypoints(pred)) * 256/64

	img = draw_keypoints(img, pred_keypoints, r = 1, g = 0, b = 0)

	return img, pred_keypoints
	