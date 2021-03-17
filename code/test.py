import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import skimage
import skimage.transform
import skimage.filters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.optim as optim
import time
import cv2
import re
from tqdm.notebook import tqdm
from torch.nn.functional import relu
from torchsummary import summary
from sklearn.utils import shuffle
from SHG import SHG
from utils import *

TRAIN_LABELS_PATH = "D:/bsc_data/train/outputs.txt"
#TEST_LABELS_PATH = "D:/bsc_data/test/outputs.txt"
VAL_LABELS_PATH = "D:/bsc_data/validation/outputs.txt"

HEADER = ["ID"]
for i in range(17):
    HEADER.append("x{}".format(i))
    HEADER.append("y{}".format(i))
    HEADER.append("v{}".format(i))

train_labels = pd.read_csv(TRAIN_LABELS_PATH, delimiter = ",", names = HEADER)
#test_labels = pd.read_csv(TEST_LABELS_PATH, delimiter = ",", names = HEADER)
val_labels = pd.read_csv(VAL_LABELS_PATH, delimiter = ",", index_col=0, header = None).T
val_labels.columns = HEADER

TRAIN_IMGS_PATH = "D:/bsc_data/train/image/"
#TEST_IMGS_PATH = "D:/bsc_data/test/image/"
VAL_IMGS_PATH = "D:/bsc_data/validation/image/"

train_imgs = os.listdir(TRAIN_IMGS_PATH)
#test_imgs = os.listdir(TEST_IMGS_PATH)
val_imgs = os.listdir(VAL_IMGS_PATH)

train_labels, train_imgs = shuffle(train_labels, train_imgs)
#test_labels, test_imgs = shuffle(test_labels, test_imgs)

LEARNING_RATE = 2.5e-4
NUM_EPOCHS = 100
MINI_BATCH_SIZE = 16
MINI_BATCHES = np.array_split(train_imgs, len(train_imgs)/MINI_BATCH_SIZE)
SAVED_MODEL_PATH = "D:/bsc_data/models/Wed_Mar_17_16-05-12_2021/epoch_0.pth"
cur_model_path = None
start_epoch = 0
average_loss = []

try:
    average_rgb = np.loadtxt("./average_rgb.npy")
except:
    average_rgb = get_mean_rgb(TRAIN_IMGS_PATH, train_imgs)
    np.savetxt("./average_rgb.npy", average_rgb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SHG(num_hourglasses=1).to(device)

if SAVED_MODEL_PATH is not None:
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    start_epoch = int(re.findall("(?<=epoch_)(.*)(?=.pth)", SAVED_MODEL_PATH)[0]) + 1
    cur_model_path = re.findall("^(.*)(?=epoch)", SAVED_MODEL_PATH)[0]
    average_loss = np.loadtxt(cur_model_path + "/loss.npy", delimiter = ",")

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr = LEARNING_RATE)

if (cur_model_path is None):
    cur_model_path = "D:/bsc_data/models/" + time.asctime().replace(" ", "_").replace(":", "-")
    os.mkdir(cur_model_path)
    
x = plt.imread(VAL_IMGS_PATH + val_imgs[0])

gt_kp = val_labels.loc[val_labels["ID"] == val_imgs[0][:-4]].to_numpy()[0][1:]

model.eval()
x_tensor = torch.from_numpy(x).permute((2, 0, 1)).to(device)
x_tensor = x_tensor.reshape((1, x_tensor.shape[0], x_tensor.shape[1], x_tensor.shape[2]))
pred = model(x_tensor).cpu().data.numpy()[0]
print(pred.shape)
print(np.max(pred))

img, pred_keypoints = draw_predicitions_and_gt(x, gt_kp, pred)
plt.imshow(img)
plt.show()
print("hej")