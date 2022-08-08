import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from config import result_dir
from config import def_ranges
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math


DATA_DIR = result_dir
SHAPE = ["airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox"]


data = {}
model_list = list(def_ranges["model"].keys())
corruption_list = list(def_ranges["corruption"].keys())
corruption_list.remove("none")
severity_list = def_ranges["severity"]

for model in model_list:
    data[model] = {"ground": [], "pred": []}


for filename in os.listdir(DATA_DIR):
    if len(filename) < 4 or filename[-4:] != ".npy":
        continue
    tokens = filename.split('.')[0].split('_')
    gt = tokens[-1]
    severity = int(tokens[-2])
    model = tokens[0]
    corruption = "_".join(tokens[1:-2])
    print(model, corruption, severity, gt)
    filepath = os.path.join(DATA_DIR, filename)
    x = np.load(filepath)
    data[model][gt].append(x)


matrixes = []
fig, axes = plt.subplots(3, 2, figsize=(20, 26))
for model_id, model in enumerate(model_list):
    ax = axes[model_id//2][model_id%2]
    total_number = len(data[model]["ground"]) * data[model]["ground"][0].shape[0]
    y_true = np.hstack(data[model]["ground"]).reshape(-1)
    y_pred = np.hstack(data[model]["pred"]).reshape(-1)
    m = confusion_matrix(y_true, y_pred)
    m = m / np.tile(np.sum(m, axis=1).reshape(40, 1), (1, 40))
    matrixes.append(m)
    sns.heatmap(m, ax=ax, vmin=0, vmax=1, center=1)
    ax.set_xticks([i+0.5 for i in range(len(SHAPE))])
    ax.set_xticklabels(SHAPE, rotation=45, ha="right")
    ax.set_yticks([i+0.5 for i in range(len(SHAPE))])
    ax.set_yticklabels(SHAPE)
    ax.set_title(def_ranges["model"][model])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_aspect('equal', adjustable='box')
plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.savefig("figures/confusion_matrix_2.pdf")

plt.clf()
fig, ax = plt.subplots(figsize=(10,8))
m = np.sum(np.stack(matrixes, axis=0), axis=0) / 6

sns.heatmap(m, ax=ax, vmin=0, vmax=1, center=1)
ax.set_xticks([i+0.5 for i in range(len(SHAPE))])
ax.set_xticklabels(SHAPE, rotation=45, ha="right")
ax.set_yticks([i+0.5 for i in range(len(SHAPE))])
ax.set_yticklabels(SHAPE)
ax.set_xlabel("Prediction")
ax.set_ylabel("Ground truth")
ax.set_aspect('equal', adjustable='box')
plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.savefig("figures/confusion_matrix_average_2.pdf")
