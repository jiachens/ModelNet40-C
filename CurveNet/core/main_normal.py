"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: main_normal.py
@Time: 2021/01/21 3:10 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from data import ModelNetNormal
from models.curvenet_normal import CurveNet
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream


def _init_():
    # fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    # prepare file structures
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    if not os.path.exists('../checkpoints/'+args.exp_name):
        os.makedirs('../checkpoints/'+args.exp_name)
    if not os.path.exists('../checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('../checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_normal.py ../checkpoints/'+args.exp_name+'/main_normal.py.backup')
    os.system('cp models/curvenet_normal.py ../checkpoints/'+args.exp_name+'/curvenet_normal.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNetNormal(args.num_points, partition='train'), 
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNetNormal(args.num_points, partition='test'), 
                             num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    # create model
    model = CurveNet(args.multiplier).to(device)
    model = nn.DataParallel(model)
    io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")

    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [140, 180], gamma=0.1)

    criterion = torch.nn.CosineEmbeddingLoss()

    best_test_loss = 99
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        for data, seg in train_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            #print(seg_pred.shape, seg.shape)
            loss = criterion(seg_pred.view(-1, 3), seg.view(-1,3).squeeze(), torch.tensor(1).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss/count)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            
            loss = criterion(seg_pred.view(-1, 3), seg.view(-1,3).squeeze(), torch.tensor(1).cuda())
            count += batch_size
            test_loss += loss.item() * batch_size
        
        if test_loss*1.0/count <= best_test_loss:
            best_test_loss = test_loss*1.0/count
            torch.save(model.state_dict(), '../checkpoints/%s/models/model.t7' % args.exp_name)
        outstr = 'Test %d, loss: %.6f, best loss %.6f' % (epoch, test_loss/count, best_test_loss)
        io.cprint(outstr)

def test(args, io):
    test_loader = DataLoader(ModelNetNormal(args.num_points, partition='test'),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = CurveNet(args.multiplier).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    criterion = torch.nn.CosineEmbeddingLoss()
    
    model = model.eval()
    test_loss = 0.0
    count = 0
    for data, seg in test_loader:
        data, seg = data.to(device), seg.to(device)
        #print(data.shape, seg.shape)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        loss = criterion(seg_pred.view(-1, 3), seg.view(-1,3).squeeze(), torch.tensor(1).cuda())
        count += batch_size
        test_loss += loss.item() * batch_size
    outstr = 'Test :: test loss: %.6f' % (test_loss*1.0/count)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--multiplier', type=float, default=2.0, metavar='MP',
                        help='network expansion multiplier')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    seed = np.random.randint(1, 10000)

    _init_()

    io = IOStream('../checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    io.cprint('random seed is: ' + str(seed))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        with torch.no_grad():
            test(args, io)
