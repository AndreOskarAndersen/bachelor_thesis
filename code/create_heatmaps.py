import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils import *
"""
TRAIN_LABELS_PATH = "D:/bsc_data/train/outputs.txt"
TRAIN_SAVING_LABELS_PATH = "D:/bsc_data/train/heatmaps/"
HEADER = ["ID"]
for i in range(17):
    HEADER.append("x{}".format(i))
    HEADER.append("y{}".format(i))
    HEADER.append("v{}".format(i))
train_labels = pd.read_csv(TRAIN_LABELS_PATH, delimiter = ",", names = HEADER)

ids = train_labels["ID"].values

for name in tqdm(ids):
    HEATMAPS_DIR = TRAIN_SAVING_LABELS_PATH + name + "/"
    os.mkdir(HEATMAPS_DIR)
    heatmaps = create_heatmaps(train_labels.loc[train_labels["ID"] == name].to_numpy())
    for i, heatmap in enumerate(heatmaps):
        np.save(HEATMAPS_DIR + str(i) + ".npy", heatmap)
"""
        
VAL_LABELS_PATH = "C:/Users/André/OneDrive 2/OneDrive/Skrivebord/bsc_data/validation/outputs.txt"
VAL_SAVING_LABELS_PATH = "C:/Users/André/OneDrive 2/OneDrive/Skrivebord/bsc_data/validation/heatmaps/"
HEADER = ["ID"]
for i in range(17):
    HEADER.append("x{}".format(i))
    HEADER.append("y{}".format(i))
    HEADER.append("v{}".format(i))
val_labels = pd.read_csv(VAL_LABELS_PATH, delimiter = ",", names = HEADER)

ids = val_labels["ID"].values

for name in ids:
    HEATMAPS_DIR = VAL_SAVING_LABELS_PATH + name + "/"
    os.mkdir(HEATMAPS_DIR)
    heatmaps = create_heatmaps(val_labels.loc[val_labels["ID"] == name].to_numpy())
    for i, heatmap in enumerate(heatmaps):
        np.save(HEATMAPS_DIR + str(i) + ".npy", heatmap)