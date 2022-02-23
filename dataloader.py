import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

from pc_utils import (rotate_point_cloud, PointcloudScaleAndTranslate)
import rs_cnn.data.data_utils as rscnn_d_utils
from rs_cnn.data.ModelNet40Loader import ModelNet40Cls as rscnn_ModelNet40Cls
import PCT_Pytorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils
from pointnet2_tf.modelnet_h5_dataset import ModelNetH5Dataset as pointnet2_ModelNetH5Dataset
from dgcnn.pytorch.data import ModelNet40 as dgcnn_ModelNet40


# distilled from the following sources:
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/data/ModelNet40Loader.py
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/train_cls.py
class ModelNet40Rscnn(Dataset):
    def __init__(self, split, data_path, train_data_path,
                 valid_data_path, test_data_path, num_points):

        self.split = split
        self.num_points = num_points
        _transforms = transforms.Compose([rscnn_d_utils.PointcloudToTensor()])
        rscnn_params = {
            'num_points': 1024,  # although it does not matter
            'root': data_path,
            'transforms': _transforms,
            'train': (split in ["train", "valid"]),
            'data_file': {
                'train': train_data_path,
                'valid': valid_data_path,
                'test':  test_data_path
            }[self.split]
        }
        self.rscnn_dataset = rscnn_ModelNet40Cls(**rscnn_params)
        self.PointcloudScaleAndTranslate = PointcloudScaleAndTranslate()

    def __len__(self):
        return self.rscnn_dataset.__len__()

    def __getitem__(self, idx):
        point, label = self.rscnn_dataset.__getitem__(idx)
        # for compatibility with the overall code
        point = np.array(point)
        label = label[0].item()

        return {'pc': point, 'label': label}

    def batch_proc(self, data_batch, device):
        point = data_batch['pc'].to(device)
        if self.split == "train":
            # (B, npoint)
            fps_idx = pointnet2_utils.furthest_point_sample(point, 1200)
            fps_idx = fps_idx[:, np.random.choice(1200, self.num_points,
                                                  False)]
            point = pointnet2_utils.gather_operation(
                point.transpose(1, 2).contiguous(),
                fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            point.data = self.PointcloudScaleAndTranslate(point.data)
        else:
            fps_idx = pointnet2_utils.furthest_point_sample(
                point, self.num_points)  # (B, npoint)
            point = pointnet2_utils.gather_operation(
                point.transpose(1, 2).contiguous(),
                fps_idx).transpose(1, 2).contiguous()
        # to maintain compatibility
        point = point.cpu()
        return {'pc': point, 'label': data_batch['label']}


# distilled from the following sources:
# https://github.com/charlesq34/pointnet2/blob/7961e26e31d0ba5a72020635cee03aac5d0e754a/modelnet_h5_dataset.py
# https://github.com/charlesq34/pointnet2/blob/7961e26e31d0ba5a72020635cee03aac5d0e754a/train.py
class ModelNet40PN2(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.dataset_name = 'modelnet40_pn2'
        data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]
        pointnet2_params = {
            'list_filename': data_path,
            # this has nothing to do with actual dataloader batch size
            'batch_size': 32,
            'npoints': num_points,
            'shuffle': False
        }

        # loading all the pointnet2data
        self._dataset = pointnet2_ModelNetH5Dataset(**pointnet2_params)
        all_pc = []
        all_label = []
        while self._dataset.has_next_batch():
            # augmentation here has nothing to do with actual data_augmentation
            pc, label = self._dataset.next_batch(augment=False)
            all_pc.append(pc)
            all_label.append(label)
        self.all_pc = np.concatenate(all_pc)
        self.all_label = np.concatenate(all_label)

    def __len__(self):
        return self.all_pc.shape[0]

    def __getitem__(self, idx):
        return {'pc': self.all_pc[idx], 'label': np.int64(self.all_label[idx])}

    def batch_proc(self, data_batch, device):
        if self.split == "train":
            point = np.array(data_batch['pc'])
            point = self._dataset._augment_batch_data(point)
            # converted to tensor to maintain compatibility with the other code
            data_batch['pc'] = torch.tensor(point)
        else:
            pass

        return data_batch


class ModelNet40Dgcnn(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]

        dgcnn_params = {
            'partition': 'train' if split in ['train', 'valid'] else 'test',
            'num_points': num_points,
            "data_path":  self.data_path
        }
        self.dataset = dgcnn_ModelNet40(**dgcnn_params)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        pc, label = self.dataset.__getitem__(idx)
        return {'pc': pc, 'label': label.item()}

def load_data(data_path,corruption,severity):

    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label

class ModelNet40C(Dataset):
    def __init__(self, split, test_data_path,corruption,severity):
        assert split == 'test'
        self.split = split
        self.data_path = {
            "test":  test_data_path
        }[self.split]
        self.corruption = corruption
        self.severity = severity

        self.data, self.label = load_data(self.data_path, self.corruption, self.severity)
        # self.num_points = num_points
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        return {'pc': pointcloud, 'label': label.item()}

    def __len__(self):
        return self.data.shape[0]


def create_dataloader(split, cfg):
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size
    dataset_args = {
        "split": split
    }

    if cfg.EXP.DATASET == "modelnet40_rscnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_RSCNN))
        # augmentation directly done in the code so that
        # it is as similar to the vanilla code as possible
        dataset = ModelNet40Rscnn(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_pn2":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_PN2))
        dataset = ModelNet40PN2(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_dgcnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_DGCNN))
        dataset = ModelNet40Dgcnn(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_c":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_C))
        dataset = ModelNet40C(**dataset_args)
    else:
        assert False

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = None

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        pin_memory=(torch.cuda.is_available()) and (not num_workers)
    )

