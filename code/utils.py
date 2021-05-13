import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.filters
import skimage.draw
import os
import torch
from tqdm.notebook import tqdm
from warnings import simplefilter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

simplefilter(action='ignore', category=FutureWarning)


def create_heatmaps(keypoints, input_shape = (256, 256), output_shape = (64, 64), is_csv_row = False):
	""" Creates heatmaps, given keypoints """
 
	if (is_csv_row):
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
		res = np.zeros(output_shape)

		if (x >= 0 and x < input_shape[1] and y >= 0 and y < input_shape[0]):

			x = x * output_shape[1]/input_shape[1]
			y = y * output_shape[0]/input_shape[0]

			x_temp = np.round(x).astype("int")
			y_temp = np.round(y).astype("int")
   
			if (x_temp >= output_shape[1]):
				x_temp = np.floor(x).astype("int")
    
			if (y_temp >= output_shape[0]):
				y_temp = np.floor(y).astype("int")

			if (v != 0):
				res[y_temp, x_temp] = 1
			
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
			for row, col in zip(rr, cc):
				if (I_copy.shape[2] == 3):
					I_copy[row, col, :] = [r, g, b]
				else:
					I_copy[:, row, col] = [r, g, b]

	return I_copy

def draw_bbox(I, bbox, r, g, b):
	I_copy = np.copy(I)

	x_min = np.ceil(bbox[0]).astype("int")
	x_max = np.floor(bbox[0] + bbox[2]).astype("int")
	y_min = np.ceil(bbox[1]).astype("int")
	y_max = np.floor(bbox[1] + bbox[3]).astype("int")

	rr, cc = skimage.draw.line(y_min, x_min, y_min, x_max)
	I_copy[rr, cc] = [r, g, b]

	rr, cc = skimage.draw.line(y_max, x_min, y_max, x_max)
	I_copy[rr, cc] = [r, g, b]

	rr, cc = skimage.draw.line(y_min, x_min, y_max, x_min)
	I_copy[rr, cc] = [r, g, b]

	rr, cc = skimage.draw.line(y_min, x_max, y_max, x_max)
	I_copy[rr, cc] = [r, g, b]

	return I_copy

def draw_segmentation_extremes(I, segmentation, r, g, b, radius):
	assert len(segmentation) == 1

	I_copy = np.copy(I)
	segmentation = np.array(segmentation[0]).reshape((-1, 2))
	x_min, y_min = np.min(segmentation, axis = 0)[0], np.min(segmentation, axis = 0)[1]
	x_max, y_max = np.max(segmentation, axis = 0)[0], np.max(segmentation, axis = 0)[1]

	rr, cc = skimage.draw.circle(y_min, x_min, radius)
	I_copy[rr, cc] = [r, g, b]

	rr, cc = skimage.draw.circle(y_max, x_min, radius)
	I_copy[rr, cc] = [r, g, b]

	rr, cc = skimage.draw.circle(y_min, x_max, radius)
	I_copy[rr, cc] = [r, g, b]

	rr, cc = skimage.draw.circle(y_max, x_max, radius)
	I_copy[rr, cc] = [r, g, b]

	return I_copy

def draw_skeleton(keypoints, img_shape = (64, 64)):
	""" Draws a skeleton corresponding to the keypoints onto a zero-matrix with shape img_shape """

	img = np.zeros((img_shape[0], img_shape[1], 3))
 
	keypoints = np.round(keypoints).astype("int")

	keypoints = np.array(keypoints).reshape((-1, 3))

	"""
	COCO keypoint-annotation uses the following indexing:
	0: nose
	1: left eye
	2: right eye
	3: left ear
	4: right ear
	5: left shoulder
	6: right shoulder
	7: left elbow
	8: right elbow
	9: left wrist
	10: right wrist
	11: left hip
	12: right hip
	13: left knee
	14: right knee
	15: left ankle
	16: right ankle
	"""
 
	# Colors
	light_blue = [0.2, 0.8, 1]
	light_green = [0.1, 1, 0.1]
	orange = [1, 0.4, 0]
	red = [1, 0, 0]
	yellow = [1, 1, 0]
	pink = [1, 0.3, 0.8]

	# Connecting nose to left eye
	if (keypoints[0, 2] != 0 and keypoints[1, 2] != 0):
		x_1 = keypoints[0, 0]
		x_2 = keypoints[1, 0]
		y_1 = keypoints[0, 1]
		y_2 = keypoints[1, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting nose to right eye
	if (keypoints[0, 2] != 0 and keypoints[2, 2] != 0):
		x_1 = keypoints[0, 0]
		x_2 = keypoints[2, 0]
		y_1 = keypoints[0, 1]
		y_2 = keypoints[2, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting left ear to left eye
	if (keypoints[3, 2] != 0 and keypoints[1, 2] != 0):
		x_1 = keypoints[3, 0]	
		x_2 = keypoints[1, 0]
		y_1 = keypoints[3, 1]
		y_2 = keypoints[1, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting right ear to right eye
	if (keypoints[2, 2] != 0 and keypoints[4, 2] != 0):
		x_1 = keypoints[2, 0]
		x_2 = keypoints[4, 0]
		y_1 = keypoints[2, 1]
		y_2 = keypoints[4, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting left shoulder to left elbow
	if (keypoints[5, 2] != 0 and keypoints[7, 2] != 0):
		x_1 = keypoints[5, 0]
		x_2 = keypoints[7, 0]
		y_1 = keypoints[5, 1]
		y_2 = keypoints[7, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = yellow

	# Connecting right shoulder to right elbow
	if (keypoints[6, 2] != 0 and keypoints[8, 2] != 0):
		x_1 = keypoints[6, 0]
		x_2 = keypoints[8, 0]
		y_1 = keypoints[6, 1]
		y_2 = keypoints[8, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = pink

	# Connecting left elbow to left wrist
	if (keypoints[7, 2] != 0 and keypoints[9, 2] != 0):
		x_1 = keypoints[7, 0]
		x_2 = keypoints[9, 0]
		y_1 = keypoints[7, 1]
		y_2 = keypoints[9, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = yellow
  
	# Connecting right elbow to right wrist
	if (keypoints[8, 2] != 0 and keypoints[10, 2] != 0):
		x_1 = keypoints[8, 0]
		x_2 = keypoints[10, 0]
		y_1 = keypoints[8, 1]
		y_2 = keypoints[10, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = pink

	# Connecting left hip to left knee
	if (keypoints[11, 2] != 0 and keypoints[13, 2] != 0):
		x_1 = keypoints[11, 0]
		x_2 = keypoints[13, 0]
		y_1 = keypoints[11, 1]
		y_2 = keypoints[13, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_blue

	# Connecting right hip to right knee
	if (keypoints[12, 2] != 0 and keypoints[14, 2] != 0):
		x_1 = keypoints[12, 0]
		x_2 = keypoints[14, 0]
		y_1 = keypoints[12, 1]
		y_2 = keypoints[14, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_green

	# Connecting left knee to left ankle
	if (keypoints[13, 2] != 0 and keypoints[15, 2] != 0):
		x_1 = keypoints[13, 0]
		x_2 = keypoints[15, 0]
		y_1 = keypoints[13, 1]
		y_2 = keypoints[15, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_blue

	# Connecting right knee to right ankle
	if (keypoints[14, 2] != 0 and keypoints[16, 2] != 0):
		x_1 = keypoints[14, 0]
		x_2 = keypoints[16, 0]
		y_1 = keypoints[14, 1]
		y_2 = keypoints[16, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_green

	# Connecting left hip to left shoulder
	if (keypoints[5, 2] != 0 and keypoints[11, 2] != 0):
		x_1 = keypoints[5, 0]
		x_2 = keypoints[11, 0]
		y_1 = keypoints[5, 1]
		y_2 = keypoints[11, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange

	# Connecting right hip to right shoulder
	if (keypoints[6, 2] != 0 and keypoints[12, 2] != 0):
		x_1 = keypoints[6, 0]
		x_2 = keypoints[12, 0]
		y_1 = keypoints[6, 1]
		y_2 = keypoints[12, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange
  
	# Connecting hips
	if (keypoints[11, 2] != 0 and keypoints[12, 2] != 0):
		x_1 = keypoints[11, 0]
		x_2 = keypoints[12, 0]
		y_1 = keypoints[11, 1]
		y_2 = keypoints[12, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange
  
	# Connecting shoulders
	if (keypoints[5, 2] != 0 and keypoints[6, 2] != 0):
		x_1 = keypoints[5, 0]
		x_2 = keypoints[6, 0]
		y_1 = keypoints[5, 1]
		y_2 = keypoints[6, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange
  
	# Connecting right ear to right shoulder
	if (keypoints[4, 2] != 0 and keypoints[6, 2] != 0):
		x_1 = keypoints[4, 0]
		x_2 = keypoints[6, 0]
		y_1 = keypoints[4, 1]
		y_2 = keypoints[6, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red
  
	# Connecting left ear to left shoulder
	if (keypoints[3, 2] != 0 and keypoints[5, 2] != 0):
		x_1 = keypoints[3, 0]
		x_2 = keypoints[5, 0]
		y_1 = keypoints[3, 1]
		y_2 = keypoints[5, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	img = draw_keypoints(img, keypoints.reshape(-1), radius = 1, r = 1, b = 1, g = 1)

	return img

def draw_necessary_skeleton(pred_keypoints, gt_keypoints, img_shape = (64, 64)):
	""" Draws a skeleton corresponding to the keypoints onto a zero-matrix with shape img_shape. Only joints where v != 0 from the groundtruth keypoints are drawn"""

	img = np.zeros((img_shape[0], img_shape[1], 3))
 
	pred_keypoints = np.round(pred_keypoints).astype("int")
	gt_keypoints = np.round(gt_keypoints).astype("int")

	pred_keypoints = np.array(pred_keypoints).reshape((-1, 3))
	gt_keypoints = np.array(gt_keypoints).reshape((-1, 3))

	"""
	COCO keypoint-annotation uses the following indexing:
	0: nose
	1: left eye
	2: right eye
	3: left ear
	4: right ear
	5: left shoulder
	6: right shoulder
	7: left elbow
	8: right elbow
	9: left wrist
	10: right wrist
	11: left hip
	12: right hip
	13: left knee
	14: right knee
	15: left ankle
	16: right ankle
	"""
 
	# Colors
	light_blue = [0.2, 0.8, 1]
	light_green = [0.1, 1, 0.1]
	orange = [1, 0.4, 0]
	red = [1, 0, 0]
	yellow = [1, 1, 0]
	pink = [1, 0.3, 0.8]

	# Connecting nose to left eye
	if (gt_keypoints[0, 2] != 0 and gt_keypoints[1, 2] != 0):
		x_1 = pred_keypoints[0, 0]
		x_2 = pred_keypoints[1, 0]
		y_1 = pred_keypoints[0, 1]
		y_2 = pred_keypoints[1, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting nose to right eye
	if (gt_keypoints[0, 2] != 0 and gt_keypoints[2, 2] != 0):
		x_1 = pred_keypoints[0, 0]
		x_2 = pred_keypoints[2, 0]
		y_1 = pred_keypoints[0, 1]
		y_2 = pred_keypoints[2, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting left ear to left eye
	if (gt_keypoints[3, 2] != 0 and gt_keypoints[1, 2] != 0):
		x_1 = pred_keypoints[3, 0]	
		x_2 = pred_keypoints[1, 0]
		y_1 = pred_keypoints[3, 1]
		y_2 = pred_keypoints[1, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting right ear to right eye
	if (gt_keypoints[2, 2] != 0 and gt_keypoints[4, 2] != 0):
		x_1 = pred_keypoints[2, 0]
		x_2 = pred_keypoints[4, 0]
		y_1 = pred_keypoints[2, 1]
		y_2 = pred_keypoints[4, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	# Connecting left shoulder to left elbow
	if (gt_keypoints[5, 2] != 0 and gt_keypoints[7, 2] != 0):
		x_1 = pred_keypoints[5, 0]
		x_2 = pred_keypoints[7, 0]
		y_1 = pred_keypoints[5, 1]
		y_2 = pred_keypoints[7, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = yellow

	# Connecting right shoulder to right elbow
	if (gt_keypoints[6, 2] != 0 and gt_keypoints[8, 2] != 0):
		x_1 = pred_keypoints[6, 0]
		x_2 = pred_keypoints[8, 0]
		y_1 = pred_keypoints[6, 1]
		y_2 = pred_keypoints[8, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = pink

	# Connecting left elbow to left wrist
	if (gt_keypoints[7, 2] != 0 and gt_keypoints[9, 2] != 0):
		x_1 = pred_keypoints[7, 0]
		x_2 = pred_keypoints[9, 0]
		y_1 = pred_keypoints[7, 1]
		y_2 = pred_keypoints[9, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = yellow
  
	# Connecting right elbow to right wrist
	if (gt_keypoints[8, 2] != 0 and gt_keypoints[10, 2] != 0):
		x_1 = pred_keypoints[8, 0]
		x_2 = pred_keypoints[10, 0]
		y_1 = pred_keypoints[8, 1]
		y_2 = pred_keypoints[10, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = pink

	# Connecting left hip to left knee
	if (gt_keypoints[11, 2] != 0 and gt_keypoints[13, 2] != 0):
		x_1 = pred_keypoints[11, 0]
		x_2 = pred_keypoints[13, 0]
		y_1 = pred_keypoints[11, 1]
		y_2 = pred_keypoints[13, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_blue

	# Connecting right hip to right knee
	if (gt_keypoints[12, 2] != 0 and gt_keypoints[14, 2] != 0):
		x_1 = pred_keypoints[12, 0]
		x_2 = pred_keypoints[14, 0]
		y_1 = pred_keypoints[12, 1]
		y_2 = pred_keypoints[14, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_green

	# Connecting left knee to left ankle
	if (gt_keypoints[13, 2] != 0 and gt_keypoints[15, 2] != 0):
		x_1 = pred_keypoints[13, 0]
		x_2 = pred_keypoints[15, 0]
		y_1 = pred_keypoints[13, 1]
		y_2 = pred_keypoints[15, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_blue

	# Connecting right knee to right ankle
	if (gt_keypoints[14, 2] != 0 and gt_keypoints[16, 2] != 0):
		x_1 = pred_keypoints[14, 0]
		x_2 = pred_keypoints[16, 0]
		y_1 = pred_keypoints[14, 1]
		y_2 = pred_keypoints[16, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = light_green

	# Connecting left hip to left shoulder
	if (gt_keypoints[5, 2] != 0 and gt_keypoints[11, 2] != 0):
		x_1 = pred_keypoints[5, 0]
		x_2 = pred_keypoints[11, 0]
		y_1 = pred_keypoints[5, 1]
		y_2 = pred_keypoints[11, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange

	# Connecting right hip to right shoulder
	if (gt_keypoints[6, 2] != 0 and gt_keypoints[12, 2] != 0):
		x_1 = pred_keypoints[6, 0]
		x_2 = pred_keypoints[12, 0]
		y_1 = pred_keypoints[6, 1]
		y_2 = pred_keypoints[12, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange
  
	# Connecting hips
	if (gt_keypoints[11, 2] != 0 and gt_keypoints[12, 2] != 0):
		x_1 = pred_keypoints[11, 0]
		x_2 = pred_keypoints[12, 0]
		y_1 = pred_keypoints[11, 1]
		y_2 = pred_keypoints[12, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange
  
	# Connecting shoulders
	if (gt_keypoints[5, 2] != 0 and gt_keypoints[6, 2] != 0):
		x_1 = pred_keypoints[5, 0]
		x_2 = pred_keypoints[6, 0]
		y_1 = pred_keypoints[5, 1]
		y_2 = pred_keypoints[6, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = orange
  
	# Connecting right ear to right shoulder
	if (gt_keypoints[4, 2] != 0 and gt_keypoints[6, 2] != 0):
		x_1 = pred_keypoints[4, 0]
		x_2 = pred_keypoints[6, 0]
		y_1 = pred_keypoints[4, 1]
		y_2 = pred_keypoints[6, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red
  
	# Connecting left ear to left shoulder
	if (gt_keypoints[3, 2] != 0 and gt_keypoints[5, 2] != 0):
		x_1 = pred_keypoints[3, 0]
		x_2 = pred_keypoints[5, 0]
		y_1 = pred_keypoints[3, 1]
		y_2 = pred_keypoints[5, 1]

		rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
		img[rr, cc] = red

	pred_keypoints = pred_keypoints[gt_keypoints[:, -1] != 0]
	img = draw_keypoints(img, pred_keypoints.reshape(-1), radius = 1, r = 1, b = 1, g = 1)

	return img

def grey_to_rgb(img):
    """ Transforms the img from grey-scale to rgb """
    return np.stack((img,)*3, axis = -1)
  
def get_mean_rgb(dir_path):
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

	if (len(feature_maps.shape) == 4):
		feature_maps = feature_maps[0]

	for feature_map in feature_maps:
		index = np.unravel_index(feature_map.argmax(), feature_map.shape) # Finds 2D argmax of feature_map
		index = (index[1], index[0]) # turns (y, x) into (x, y)
		for el in index:
			arr.append(el)

		if (torch.sum(feature_map) ==  0): # If the keypoint is not visible
			arr.append(0)
		else:
			arr.append(2)
  
	return arr

def draw_predicitions_and_gt(img, gt, pred):
	""" Draws the 17 true keypoints in red and the 17 predicted featuremaps in green onto the image given by img"""

	gt_heatmaps = np.array(turn_featuremaps_to_keypoints(gt)) * 256/64
	pred_keypoints = np.array(turn_featuremaps_to_keypoints(pred)) * 256/64

	img = draw_keypoints(img, gt_heatmaps, r = 0, g = 1, b = 0)
	img = draw_keypoints(img, pred_keypoints, r = 1, g = 0, b = 0)

	return img
	
 
def PCK(gt_heatmaps, pred_heatmaps, normalizing_const = 6.4, threshold = 0.5):
	"""
	Computes the Percentage of Correct Keypoints between the groundtruth heatmaps and predicted heatmaps.
	Unannotated keypoints are ignored.
 
	Parameters:
		gt_heatmaps: tensor of groundtruth heatmaps
		pred_heatmaps: tensor of predicted heatmaps
		normalizing_const: constant used for normalizing the distance between gt_heatmaps and pred_heatmaps
		threshold: threshold used for ensuring if pred_heatmaps is equal to gt_heatmaps
	"""

	# Turning the heatmaps into arrays
	gt_kp = np.array(turn_featuremaps_to_keypoints(gt_heatmaps)).reshape((-1, 3))
	pred_kp = np.array(turn_featuremaps_to_keypoints(pred_heatmaps)).reshape((-1, 3))

	# Removing unannotated joints
	pred_kp = pred_kp[gt_kp[:, -1] != 0]
	gt_kp = gt_kp[gt_kp[:, -1] != 0]

	# Removing visibility flag
	gt_kp = gt_kp[:, :-1]
	pred_kp = pred_kp[:, :-1]

	# Distance between ground truth keypoints and predictions
	dist = np.linalg.norm(gt_kp - pred_kp, axis = 1)

	# Normalizing distance
	dist = dist/normalizing_const

	# Counting the amount of correctly predicted joints
	num_correct = len(dist[dist < threshold])

	# Returning the ratio of correctly predicted joints
	return num_correct/len(dist)

def find_correct_incorrect(gt_heatmaps, pred_heatmaps, normalizing_const = 6.4, threshold = 0.5):
	""" Returns the indexes of the keypoints that are classified as being correct and incorrect according to PCK"""

	# Turning the heatmaps into arrays
	gt_kp = np.array(turn_featuremaps_to_keypoints(gt_heatmaps)).reshape((-1, 3))
	pred_kp = np.array(turn_featuremaps_to_keypoints(pred_heatmaps)).reshape((-1, 3))
 
	# Finding indexes of keypoints unnanotated
	unanotated_indexes = np.where(gt_kp[:, -1] == 0)
 
	# Removing unannotated joints
	pred_kp = pred_kp[gt_kp[:, -1] != 0]
	gt_kp = gt_kp[gt_kp[:, -1] != 0] 

	# Distance between ground truth keypoints and predictions
	dist = np.linalg.norm(gt_kp[:, :-1] - pred_kp[:, :-1], axis = 1)

	# Normalizing distance
	dist = dist/normalizing_const

	# Finding indexes of correct
	correct = np.where(dist < threshold)
	
	# Finding indexes of incorrect
	incorrect = np.where(dist >= threshold)
 
	return correct, incorrect, unanotated_indexes

def load_data(IMGS_PATH, HEATMAPS_PATH):
	""" 
		Load all the images and the corresponding heatmaps, located at IMGS_PATH and HEATMAPS_PATH
		NOTE: IT IS ASSUMED, THAT THE IMAGES ARE .npy!
	"""

	img_res = []
	heatmap_res = []
	imgs = os.listdir(IMGS_PATH)

	for img in tqdm(imgs, leave = False):
		img_path = IMGS_PATH + img
		heatmap_dir = HEATMAPS_PATH + img[:-4] + "/"
		heatmaps = []

		for i in range(17):
			heatmaps.append(torch.from_numpy(np.load(heatmap_dir + str(i) + ".npy")))
		
		img_res.append(np.load(img_path))
		heatmap_res.append(torch.stack(heatmaps))

	img_res = np.array(img_res).reshape((len(imgs), -1))
	heatmap_res = torch.stack(heatmap_res)
 
	return img_res, heatmap_res, imgs

def get_kmeans(X, max_k = 10, max_iter = 1000, fig_saving_path = None):
    Ks = np.arange(2, max_k + 1, 1) # Which k's to use for KMeans
    sil = [] # Storing silhouette score
    best_model = None # Storing the best model, based on silhouette score
    overall_best_sil = -1 # Storing the best silhouette score
    best_centroids = None
    inertias = []

    for k in tqdm(Ks, desc = "k", leave = False):
        model = KMeans(n_clusters = k, max_iter = max_iter).fit(X)
        
        # Computes silhouette score
        labels = model.labels_
        cur_sil = silhouette_score(X, labels, metric = 'euclidean')

        # Inertia
        inertias.append(model.inertia_)

        # Compares current silhouette score with the best silhouette score of the current k
        if (cur_sil > overall_best_sil):
            best_model = model
            overall_best_sil = cur_sil
            best_centroids = model.cluster_centers_

        # Appends the best silhouette score of each k
        sil.append(cur_sil)

    # Plotting silhouette score
    sil = np.array(sil)
    plt.figure()
    plt.plot(np.arange(2, max_k + 1, 1), sil)
    plt.xticks(np.arange(2, max_k + 1, 1))
    plt.title("Best silhouette score for each cluster")
    plt.xlabel("Amount of clusters")
    plt.ylabel("Silhouette score")
    
    if (fig_saving_path is not None):
        plt.savefig(fig_saving_path + "silhouette_score.png")
    
    plt.show()

    # Plotting inertia
    plt.figure()
    plt.plot(np.arange(2, max_k + 1, 1), inertias)
    plt.xticks(np.arange(2, max_k + 1, 1))
    plt.title("Inertia for each amount of clusters")
    plt.xlabel("Amount of clusters")
    plt.ylabel("Inertia score")
    
    if (fig_saving_path is not None):
        plt.savefig(fig_saving_path + "intertia_score.png")
    
    plt.show()

    return best_model, sil, best_centroids
 

def get_kmeans_alternative(X, min_k = 2, max_k = 10, max_iter = 1000, fig_saving_path = None):
	""" 
	Runs kmeans, but instead of finding the synthetic centroid of each cluster, it finds the nearest true observation and assigns that as the centroid.
	Each observation is labelled accordingly.
	"""

	Ks = np.arange(min_k, max_k + 1, 1) # Which k's to use for KMeans
	sil = [] # Storing silhouette score
	best_model = None # Storing the best model, based on silhouette score
	overall_best_sil = -1 # Storing the best silhouette score
	best_centroids = None
	best_labels = None

	for k in tqdm(Ks, desc = "k", leave = False):
		model = KMeans(n_clusters = k, max_iter = max_iter).fit(X)
		centroids = model.cluster_centers_
		
		true_centroids = np.zeros(centroids.shape)

		# Finds the nearest true centroids
		for i, centroid in enumerate(centroids):
			dists = np.linalg.norm(centroid - X, axis = 1)
			true_centroids[i] = X[np.argmin(dists)]
			dists[np.argmin(dists)] = np.max(dists) # Makes sure that multiple centroids are not set to the same nearest observation

		# Labels each datapoint accordingly to which true centroid it is the closest to
		labels = np.zeros(X.shape[0])
		dists = None # the best distances from a cluster to each observation
		for i, centroid in enumerate(true_centroids):
			centroid = centroid.reshape((1, -1))
			dist = np.linalg.norm(centroid - X, axis = 1)
	
			if (dists is None):
				dists = dist
			else:
				labels[dist < dists] = i
				dists[dist < dists] = dist[dist < dists]
		
		# Computes silhouette score
		cur_sil = silhouette_score(X, labels, metric = 'euclidean')

		# Compares current silhouette score with the best silhouette score of the current k
		if (cur_sil > overall_best_sil):
			best_model = model
			overall_best_sil = cur_sil
			best_centroids = np.copy(true_centroids)
			best_labels = np.copy(labels)

		# Appends the best silhouette score of each k
		sil.append(cur_sil)

	# Plotting silhouette score
	sil = np.array(sil)
	plt.figure()
	plt.plot(np.arange(min_k, max_k + 1, 1), sil)
	plt.xticks(np.arange(min_k, max_k + 1, 1))
	plt.title("Best silhouette score for each cluster")
	plt.xlabel("Amount of clusters")
	plt.ylabel("Silhouette score")

	if (fig_saving_path is not None):
			plt.savefig(fig_saving_path + "silhouette_score.png")

	plt.show()

	return best_model, best_centroids, best_labels.astype("uint8")

def get_kmedoids(X, max_k = 10, max_iter = 1000):
	Ks = np.arange(2, max_k + 1, 1) # Which k's to use for KMeans
	sil = [] # Storing silhouette score
	best_model = None # Storing the best model, based on silhouette score
	overall_best_sil = -1 # Storing the best silhouette score
	best_centroids = None
	inertias = []

	for k in tqdm(Ks, desc = "k", leave = False):
		model = KMedoids(n_clusters = k, max_iter = max_iter).fit(X)
		
		# Computes silhouette score
		labels = model.labels_
		cur_sil = silhouette_score(X, labels, metric = 'euclidean')

		# Inertia
		inertias.append(model.inertia_)

		# Compares current silhouette score with the best silhouette score of the current k
		if (cur_sil > overall_best_sil):
			best_model = model
			overall_best_sil = cur_sil
			best_centroids = model.cluster_centers_

		# Appends the best silhouette score of each k
		sil.append(cur_sil)

	# Plotting silhouette score
	sil = np.array(sil)
	plt.figure()
	plt.plot(np.arange(2, max_k + 1, 1), sil)
	plt.xticks(np.arange(2, max_k + 1, 1))
	plt.title("Best silhouette score for each cluster")
	plt.xlabel("Amount of clusters")
	plt.ylabel("Silhouette score")
	plt.show()

	# Plotting inertia
	plt.figure()
	plt.plot(np.arange(2, max_k + 1, 1), inertias)
	plt.xticks(np.arange(2, max_k + 1, 1))
	plt.title("Inertia for each amount of clusters")
	plt.xlabel("Amount of clusters")
	plt.ylabel("Inertia score")
	plt.show()

	return best_model, sil, best_centroids

def visualize_clusters_pca(X, heatmaps, labels, fig_saving_path = None):
	NUM_CLUSTERS = len(np.unique(labels))
	clusters = [[] for _ in range(NUM_CLUSTERS)]
	points_in_each_cluster = np.zeros(NUM_CLUSTERS)

	# Seperates the clusters
	for x, l in zip(X, labels):
		points_in_each_cluster[l] += 1
		clusters[l].append(x)

	# Reshapes the clusters
	for i in range(NUM_CLUSTERS):
		clusters[i] = np.array(clusters[i]).reshape((int(points_in_each_cluster[i]), -1))

	figs, axs = plt.subplots(1, 2, figsize = (20, 10))

	# Draws the clusters in 2D space
	for i in tqdm(range(NUM_CLUSTERS), leave = False):
		pca = PCA(n_components=2)
		
		X = pca.fit_transform(StandardScaler().fit_transform(clusters[i]))

		axs[i].set_xlabel("Principal component 1\nExplained Variance ratio: {:.3f}".format(pca.explained_variance_ratio_[0]))
		axs[i].set_ylabel("Principal component 2\nExplained Variance ratio: {:.3f}".format(pca.explained_variance_ratio_[1]))
		axs[i].set_title("Cluster {}\n$n =${}".format(i, int(points_in_each_cluster[i])))
		
		for x, y, l in zip(X, heatmaps, labels):
			if (l == i):
				y = turn_featuremaps_to_keypoints(y)
				image = draw_skeleton(y)
				im = OffsetImage(image, zoom = 0.5)
				ab = AnnotationBbox(im, (x[0], x[1]), xycoords = "data", frameon = False)
				axs[i].add_artist(ab)
				axs[i].update_datalim([(x[0], x[1])])
				axs[i].autoscale()

	plt.tight_layout()

	if (fig_saving_path is not None):
		plt.savefig(fig_saving_path + "cluster.png")

	plt.show()
  
def visualize_clusters_tsne(X, heatmaps, labels):
	NUM_CLUSTERS = len(np.unique(labels))
	clusters = [[] for _ in range(NUM_CLUSTERS)]
	points_in_each_cluster = np.zeros(NUM_CLUSTERS)

	# Seperates the clusters
	for x, l in zip(X, labels):
		points_in_each_cluster[l] += 1
		clusters[l].append(x)

	# Reshapes the clusters
	for i in range(NUM_CLUSTERS):
		clusters[i] = np.array(clusters[i]).reshape((int(points_in_each_cluster[i]), -1))

	# Draws the clusters in 2D space
	for i in tqdm(range(NUM_CLUSTERS), leave = False):

		if (clusters[i].shape[1] != 2):
			X = TSNE(n_components = 2).fit_transform(X)
		
		fig, ax = plt.subplots(figsize = (10, 10))
		plt.figure()
		for x, y, l in zip(X, heatmaps, labels):
			if (l == i):
				y = turn_featuremaps_to_keypoints(y)
				image = draw_skeleton(y)
				im = OffsetImage(image, zoom = 0.5)
				ab = AnnotationBbox(im, (x[0], x[1]), xycoords = "data", frameon = False)
				ax.add_artist(ab)
				ax.update_datalim([(x[0], x[1])])
				ax.autoscale()

		plt.show()