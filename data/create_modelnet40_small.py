#!/usr/bin/env python
import os
import h5py
import numpy as np

np.random.seed(123)


def main(split_size):
    modelnet40_dir = "./data/modelnet40_ply_hdf5_2048/"

    modelnet40_train_file = os.path.join(
        modelnet40_dir, "train_minus_valid_files.txt")

    modelnet40_train_split_file = os.path.join(
        modelnet40_dir, f"train_minus_valid_split_{split_size}_files.txt")

    modelnet40_train_split_path = f"ply_data_trainminusval_split_{split_size}.h5"

    with open(modelnet40_train_file, "r") as f:
        modelnet40_train_paths = [l.strip() for l in f.readlines()]

    data = []
    labels = []
    for modelnet40_train_path in modelnet40_train_paths:
        train_h5 = h5py.File(modelnet40_train_path, "r")

        data.append(train_h5["data"][:])
        labels.append(train_h5["label"][:])

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    train_data = []
    train_label = []
    for i in range(40):
        cls_inds = np.where(labels == i)[0]
        num_objs = len(cls_inds)
        num_train = int(num_objs * split_size)
        cls_data = data[cls_inds]

        np.random.shuffle(cls_data)

        train_data.append(cls_data[:num_train])
        train_label += [i] * num_train

    train_data = np.concatenate(train_data)
    train_label = np.array(train_label).reshape(-1, 1)

    with open(modelnet40_train_split_file, "w") as f:
        f.write(os.path.join(modelnet40_dir,
                             modelnet40_train_split_path) + "\n")

    with h5py.File(
            os.path.join(modelnet40_dir, modelnet40_train_split_path),
            "w") as f:
        f.create_dataset("data", data=train_data)
        f.create_dataset("label", data=train_label)

    print('data: {}'.format(data.shape))
    print('train data: {}'.format(train_data.shape))
    print('min_label: {}'.format(labels.min()))
    print('max_label: {}'.format(labels.max()))


if __name__ == "__main__":
    main(0.5 / 0.8)
    main(0.25 / 0.8)
