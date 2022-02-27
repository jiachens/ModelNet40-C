import numpy as np
import torch
import sys
from main import get_loss
sys.path.append("./emd/")
import emd_module as emd

def cutmix_r(data_batch,cfg):
    r = np.random.rand(1)
    if cfg.AUG.BETA > 0 and r < cfg.AUG.PROB:
        lam = np.random.beta(cfg.AUG.BETA, cfg.AUG.BETA)
        B = data_batch['pc'].size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['label']
        target_b = data_batch['label'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        point_a = data_batch['pc']
        point_b = data_batch['pc'][rand_index]
        point_c = data_batch['pc'][rand_index]
        # point_a, point_b, point_c = point_a.to(device), point_b.to(device), point_c.to(device)

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(cfg.DATALOADER.MODELNET40_DGCNN.num_points * lam)
        int_lam = max(1, int_lam)
        gamma = np.random.choice(cfg.DATALOADER.MODELNET40_DGCNN.num_points, int_lam, replace=False, p=None)
        for i2 in range(B):
            data_batch['pc'][i2, gamma, :] = point_c[i2, gamma, :]

        # adjust lambda to exactly match point ratio
        lam = int_lam * 1.0 / cfg.DATALOADER.MODELNET40_DGCNN.num_points
        # points = data_batch['pc'].transpose(2, 1)
        data_batch['label_2'] = target_b
        data_batch['lam'] = lam

    return data_batch
        # pred, trans_feat = model(points)
        # loss = criterion(pred, target_a.long()) * (1. - lam) + criterion(pred, target_b.long()) * lam
    


def cutmix_k(data_batch,cfg):
    r = np.random.rand(1)
    if cfg.AUG.BETA > 0 and r < cfg.AUG.PROB:
        lam = np.random.beta(cfg.AUG.BETA, cfg.AUG.BETA)
        B = data_batch['pc'].size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['label']
        target_b = data_batch['label'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        point_a = data_batch['pc']
        point_b = data_batch['pc'][rand_index]
        point_c = data_batch['pc'][rand_index]

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(cfg.DATALOADER.MODELNET40_DGCNN.num_points * lam)
        int_lam = max(1, int_lam)

        random_point = torch.from_numpy(np.random.choice(1024, B, replace=False, p=None))
        # kNN
        ind1 = torch.tensor(range(B))
        query = point_a[ind1, random_point].view(B, 1, 3)
        dist = torch.sqrt(torch.sum((point_a - query.repeat(1, cfg.DATALOADER.MODELNET40_DGCNN.num_points, 1)) ** 2, 2))
        idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
        for i2 in range(B):
            data_batch['pc'][i2, idxs[i2], :] = point_c[i2, idxs[i2], :]
        # adjust lambda to exactly match point ratio
        lam = int_lam * 1.0 / cfg.DATALOADER.MODELNET40_DGCNN.num_points
        # points = points.transpose(2, 1)
        # pred, trans_feat = model(points)
        # loss = criterion(pred, target_a.long()) * (1. - lam) + criterion(pred, target_b.long()) * lam
        data_batch['label_2'] = target_b
        data_batch['lam'] = lam
        
    return data_batch


def mixup(data_batch,cfg):

    batch_size = data_batch['pc'].size()[0]
    idx_minor = torch.randperm(batch_size)
    mixrates = (0.5 - np.abs(np.random.beta(cfg.AUG.MIXUPRATE, cfg.AUG.MIXUPRATE, batch_size) - 0.5))
    label_main = data_batch['label']
    label_minor = data_batch['label'][idx_minor]
    label_new = torch.zeros(batch_size, 40)
    for i in range(batch_size):
        if label_main[i] == label_minor[i]: # same label
            label_new[i][label_main[i]] = 1.0
        else:
            label_new[i][label_main[i]] = 1 - mixrates[i]
            label_new[i][label_minor[i]] = mixrates[i]
    label = label_new

    data_minor = data_batch['pc'][idx_minor]
    mix_rate = torch.tensor(mixrates).float()
    mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)

    mix_rate_expand_xyz = mix_rate.expand(data_batch['pc'].shape)

    remd = emd.emdModule()
    remd = remd.cuda()
    _, ass = remd(data_batch['pc'], data_minor, 0.005, 300)
    ass = ass.long()
    for i in range(batch_size):
        data_minor[i] = data_minor[i][ass[i]]
    data_batch['pc'] = data_batch['pc'] * (1 - mix_rate_expand_xyz) + data_minor * mix_rate_expand_xyz
    data_batch['label_2'] = label_minor
    data_batch['lam'] = torch.tensor(mix_rate).squeeze_()

    return data_batch


def knn_points(k, xyz, query, nsample=512):
    B, N, C = xyz.shape
    _, S, _ = query.shape # S=1
    
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    sqrdists = square_distance(query, xyz) # Bx1,N #제곱거리
    tmp = np.sort(sqrdists, axis=2)
    knn_dist = np.zeros((B,1))
    for i in range(B):
        knn_dist[i][0] = tmp[i][0][k]
        group_idx[i][sqrdists[i]>knn_dist[i][0]]=N
    # group_idx[sqrdists > radius ** 2] = N
    # print("group idx : \n",group_idx)
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
    
def cut_points_knn(data_batch, idx, radius, nsample=512, k=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = knn_points(k=k, xyz=data_batch[:,:,:3], query=query_points[:,:,:3], nsample=nsample)
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6

def cut_points(data_batch, idx, radius, nsample=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = query_ball_point_for_rsmix(radius, nsample, data_batch[:,:,:3], query_points[:,:,:3])
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6


def query_ball_point_for_rsmix(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample], S=1
    """
    # device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    dist = -2 * np.matmul(src, dst.transpose(0, 2, 1))
    dist += np.sum(src ** 2, -1).reshape(B, N, 1)
    dist += np.sum(dst ** 2, -1).reshape(B, 1, M)
    
    return dist


def pts_num_ctrl(pts_erase_idx, pts_add_idx):
    '''
        input : pts - to erase 
                pts - to add
        output :pts - to add (number controled)
    '''
    if len(pts_erase_idx)>=len(pts_add_idx):
        num_diff = len(pts_erase_idx)-len(pts_add_idx)
        if num_diff == 0:
            pts_add_idx_ctrled = pts_add_idx
        else:
            pts_add_idx_ctrled = np.append(pts_add_idx, pts_add_idx[np.random.randint(0, len(pts_add_idx), size=num_diff)])
    else:
        pts_add_idx_ctrled = np.sort(np.random.choice(pts_add_idx, size=len(pts_erase_idx), replace=False))
    return pts_add_idx_ctrled

def rsmix(data, cfg, n_sample=512, KNN=False):
    cut_rad = np.random.beta(cfg.AUG.BETA, cfg.AUG.BETA)
    data_batch = data['pc'].numpy()
    label_batch = data['label'].numpy()

    rand_index = np.random.choice(data_batch.shape[0],data_batch.shape[0], replace=False) # label dim : (16,) for model
    
    if len(label_batch.shape) is 1:
        label_batch = np.expand_dims(label_batch, axis=1)
        
    label_a = label_batch[:,0]
    label_b = label_batch[rand_index][:,0]
        
    data_batch_rand = data_batch[rand_index] # BxNx3(with normal=6)
    rand_idx_1 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    rand_idx_2 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    if KNN:
        knn_para = min(int(np.ceil(cut_rad*n_sample)),n_sample)
        pts_erase_idx, query_point_1 = cut_points_knn(data_batch, rand_idx_1, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points_knn(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_2 x 3(or 6)
    else:
        pts_erase_idx, query_point_1 = cut_points(data_batch, rand_idx_1, cut_rad, nsample=n_sample) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample) # B x num_points_in_radius_2 x 3(or 6)
    
    query_dist = query_point_1[:,:,:3] - query_point_2[:,:,:3]
    
    pts_replaced = np.zeros((1,data_batch.shape[1],data_batch.shape[2]))
    lam = np.zeros(data_batch.shape[0],dtype=float)

    for i in range(data_batch.shape[0]):
        if pts_erase_idx[i][0][0]==data_batch.shape[1]:
            tmp_pts_replaced = np.expand_dims(data_batch[i], axis=0)
            lam_tmp = 0
        elif pts_add_idx[i][0][0]==data_batch.shape[1]:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            dup_points_idx = np.random.randint(0,len(tmp_pts_erased), size=len(pts_erase_idx_tmp))
            tmp_pts_replaced = np.expand_dims(np.concatenate((tmp_pts_erased, data_batch[i][dup_points_idx]), axis=0), axis=0)
            lam_tmp = 0
        else:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_tmp = np.unique(pts_add_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_ctrled_tmp = pts_num_ctrl(pts_erase_idx_tmp,pts_add_idx_tmp)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            # input("INPUT : ")
            tmp_pts_to_add = np.take(data_batch_rand[i], pts_add_idx_ctrled_tmp, axis=0)
            tmp_pts_to_add[:,:3] = query_dist[i]+tmp_pts_to_add[:,:3]
            
            tmp_pts_replaced = np.expand_dims(np.vstack((tmp_pts_erased,tmp_pts_to_add)), axis=0)
            
            lam_tmp = len(pts_add_idx_ctrled_tmp)/(len(pts_add_idx_ctrled_tmp)+len(tmp_pts_erased))
        
        pts_replaced = np.concatenate((pts_replaced, tmp_pts_replaced),axis=0)
        lam[i] = lam_tmp
    
    data_batch_mixed = np.delete(pts_replaced, [0], axis=0)    
        
    data['pc'] = torch.FloatTensor(data_batch_mixed)
    data['label'] = torch.tensor(label_a)
    data['label_2'] = torch.tensor(label_b)
    data['lam'] = torch.tensor(lam)

    return data


def pgd(data_batch,model, task, loss_name, dataset_name, step= 7, eps=0.05, alpha=0.01):
    model.eval()
    data = data_batch['pc']
    adv_data=data.clone()
    adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
    adv_data.detach()
    adv_data_batch = {}

    for _ in range(step):
        adv_data.requires_grad=True
        out = model(**{'pc':adv_data})
        adv_data_batch['pc'] = adv_data
        adv_data_batch['label'] = data_batch['label']
        model.zero_grad()
        loss = get_loss(task, loss_name, adv_data_batch, out, dataset_name)
        loss.backward()
        with torch.no_grad():
            adv_data = adv_data + alpha * adv_data.grad.sign()
            delta = adv_data-data
            # print(delta)
            delta = torch.clamp(delta,-eps,eps)
            adv_data = (data+delta).detach_()
    
    return adv_data_batch
