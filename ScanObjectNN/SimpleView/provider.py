import os
import sys
import numpy as np
import h5py
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(ROOT_DIR, '../')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'data/modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(os.path.join(DATA_DIR, filename))

def load_h5_data_label_seg(h5_filename):
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
     Reference: # source https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([
        cosz, -sinz, zero,
        sinz, cosz, zero,
        zero, zero, one
    ], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([
        cosy, zero, siny,
        zero, one, zero,
        -siny, zero, cosy
    ], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([
        one, zero, zero,
        zero, cosx, -sinx,
        zero, sinx, cosx
    ], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    return rot_mat


def point_transform(points, angle, translation):
    """
    :param points: [batch, height*width, 3]
    :param angle: [3] or [batch, 3]
    :param translation: [3] or [batch, 3]
    :return:
    """

    rot_mat = euler2mat(angle)
    rot_mat = rot_mat.to(points.device)

    if len(angle.size()) == 1:
        points = torch.matmul(points, torch.transpose(rot_mat, 0, 1))
    else:
        points = torch.matmul(points, torch.transpose(rot_mat, 1, 2))

    translation = translation.to(points.device)
    if len(angle.size()) == 2:
        translation = translation.unsqueeze(1)
    points = points - translation
    return points


def get_modelnet_data(file_name, rel_path="../"):
    with open(file_name) as file:
        files = [rel_path + line.rstrip() for line in file]

    total_data = np.array([]).reshape((0, 2048, 3))
    total_labels = np.array([]).reshape((0, 1))
    for i in range(len(files)):
        data, labels = load_h5(files[i])
        total_data   = np.concatenate((total_data, data))
        total_labels = np.concatenate((total_labels, labels))
    total_labels = total_labels.astype(int)

    return total_data, total_labels
