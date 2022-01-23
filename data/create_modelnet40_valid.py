#!/usr/bin/env python
import os
import h5py
import numpy as np

np.random.seed(123)
def main():
    modelnet40_dir = "./data/modelnet40_ply_hdf5_2048/"

    modelnet40_train_minus_valid_path = "ply_data_trainminusval.h5"
    modelnet40_valid_path             = "ply_data_valid.h5"

    modelnet40_train_minus_valid_file = os.path.join(modelnet40_dir, "train_minus_valid_files.txt")
    modelnet40_valid_file             = os.path.join(modelnet40_dir, "valid_files.txt")

    modelnet40_train_file = os.path.join(modelnet40_dir, "train_files.txt")
    with open(modelnet40_train_file, "r") as f:
        modelnet40_train_paths = [l.strip() for l in f.readlines()]

    data   = []
    labels = []
    for modelnet40_train_path in modelnet40_train_paths:
        train_h5 = h5py.File(modelnet40_train_path, "r")

        data.append(train_h5["data"][:])
        labels.append(train_h5["label"][:])

    data   = np.concatenate(data)
    labels = np.concatenate(labels)

    train_data  = []
    train_label = []
    valid_data  = []
    valid_label = []
    for i in range(40):
        cls_inds  = np.where(labels == i)[0]
        num_objs  = len(cls_inds)
        num_train = int(num_objs * 0.8)
        num_valid = num_objs - num_train
        cls_data  = data[cls_inds]

        np.random.shuffle(cls_data)

        train_data.append(cls_data[:num_train])
        valid_data.append(cls_data[num_train:])

        train_label += [i] * num_train
        valid_label += [i] * num_valid

    train_data  = np.concatenate(train_data)
    valid_data  = np.concatenate(valid_data)
    train_label = np.array(train_label).reshape(-1, 1)
    valid_label = np.array(valid_label).reshape(-1, 1)

    with open(modelnet40_train_minus_valid_file, "w") as f:
        f.write(os.path.join(modelnet40_dir, modelnet40_train_minus_valid_path) + "\n")

    with open(modelnet40_valid_file, "w") as f:
        f.write(os.path.join(modelnet40_dir, modelnet40_valid_path) + "\n")

    with h5py.File(os.path.join(modelnet40_dir, modelnet40_train_minus_valid_path), "w") as f:
        f.create_dataset("data",  data=train_data)
        f.create_dataset("label", data=train_label)

    with h5py.File(os.path.join(modelnet40_dir, modelnet40_valid_path), "w") as f:
        f.create_dataset("data",  data=valid_data)
        f.create_dataset("label", data=valid_label)

    print('data: {}'.format(data.shape))
    print('min_label: {}'.format(labels.min()))
    print('max_label: {}'.format(labels.max()))

if __name__ == "__main__":
    main()
