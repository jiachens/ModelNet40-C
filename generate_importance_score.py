import torch
import numpy as np
import torchvision
import os, sys
import argparse
import pickle
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader, get_dataset
from configs import get_cfg_defaults
import random

parser = argparse.ArgumentParser(description='PyTorch modelnec40 training')

######################### Data Setting #########################
parser.add_argument('--dataset', type=str, default='modelnet40_dgcnn', choices=['modelnet40_rscnn', 'modelnet40_pn2', 'modelnet40_dgcnn', 'modelnet40_c'])

######################### Path Setting #########################
parser.add_argument('--data-dir', default='./', type=str, help='The dir path of the data.')
parser.add_argument('--base-dir', default='./', type=str, help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='',
                    help='The name of the training task.')
parser.add_argument('--exp-config', type=str, default='./configs/pgd/dgcnn.yaml')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='3',
                    help='The ID of GPU.')


args = parser.parse_args()
######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'model_best_test.pth')
td_path = os.path.join(task_dir, f'td.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}.pickle')

dataset = args.dataset
if dataset == "modelnet40_dgcnn":
    num_classes = 40
# TODO: add other dataset


######################### Ftn definition #########################
"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))

    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss


"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance, targets):
    # targets = []
    indices = []
    # print(len(dataset), dataset[0][1].shape[0])
    data_size = len(targets) # batch size

    ## dataset should contain 9840 samples

    for i in range(len(dataset)):
        _, y, idxes = dataset[i]
        # targets.append(y.numpy())
        indices.append(idxes)
    # print(targets)
    targets = torch.flatten(torch.tensor(targets))
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

    def record_training_dynamics(td_log):
        output = td_log['output'].cpu()
        predicted = output.argmax(dim=1)

        index = indices[td_log['iterarion']]
        # print(index)
        label = targets[index]
        correctness = (predicted == label).type(torch.int)
        # print("correctness", correctness)
        # print("last correctness", data_importance['last_correctness'][index])

        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        # print(type(index), index, type(correctness))
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin
    
    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        record_training_dynamics(item)


"""Calculate td metrics"""
def EL2N(td_log, dataset, data_importance, targets, max_epoch=10):
    # targets = []
    indices = []
    data_size = len(targets)

    for i in range(len(dataset)):
        _, y, idxes = dataset[i]
        # targets.append(y.numpy())
        indices.append(idxes)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = td_log['output']
        output = output.to('cpu')
        # index = td_log['iterarion'].type(torch.long)
        index = indices[td_log['iterarion']]

        label = targets[index]
        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        label_onehot = np.squeeze(label_onehot)
        # print(label_onehot.shape, output.shape)
        el2n_score = torch.sqrt(l2_loss(label_onehot, output).sum(dim=1))
        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)


######################### Testing #########################

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get dataset
cfg = get_cfg_defaults()
cfg.merge_from_file(args.exp_config)
cfg.freeze()
random.seed(cfg.EXP.SEED)
np.random.seed(cfg.EXP.SEED)
torch.manual_seed(cfg.EXP.SEED)
loader_train = create_dataloader(split='train', cfg=cfg, coreset_method=None)

index_dataset = []
for i, data_batch in enumerate(loader_train):
    # print(data_batch.keys(), data_batch['idx'])
    index_dataset.append((data_batch['pc'], data_batch['label'], data_batch['idx']))


dataset_all = get_dataset(split='train', cfg=cfg, coreset_method=None)

data_importance = {}

# load model


with open(td_path, 'rb') as f:
    pickled_data = pickle.load(f)
    # print(pickled_data['training_dynamics'][0])


training_dynamics = pickled_data['training_dynamics']

training_dynamics_metrics(training_dynamics, index_dataset, data_importance, dataset_all.dataset.label)

# EL2N
EL2N(training_dynamics, index_dataset, data_importance, dataset_all.dataset.label, max_epoch=10)

print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)