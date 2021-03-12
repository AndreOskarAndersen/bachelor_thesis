import numpy as np
import skimage
import skimage.filters
import skimage.transform
import skimage.draw

def create_heatmaps(keypoints, input_shape = (256, 256), output_shape = (64, 64)):
    """ GIVEN ONE OF THE ROWS IN THE CSV, CREATES THE 17 CORRESPONDING HEATMAPSS """
    keypoints = keypoints[0, 1:] # We dont need the ID of the image

    x_val = keypoints[::3]
    y_val = keypoints[1::3]
    v_val = keypoints[2::3]

    res_arr = []

    for x, y, v in zip(x_val, y_val, v_val):
        x, y, v = int(x), int(y), int(v)
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