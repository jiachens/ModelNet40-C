from collections import OrderedDict

# Path of the dataset containing npy files.
data_dir = "modelnet40_c"
# Path of logs of experiments.
result_dir = "output"

# For each OrderedDict item, the first element is the name in log files while the
# second item is the name to display on visualized figures.
def_ranges = {
    'model': OrderedDict([
        ('pointnet', 'PointNet'), 
        ('pointnet2', 'PointNet++'), 
        ('dgcnn', 'DGCNN'), 
        ('rscnn', 'RSCNN'), 
        ('pct', 'PCT'), 
        ('simpleview', 'SimpleView'),
        ('curvenet', 'CurveNet'),
        ("gdanet", "GDANet"),
        ('pointMLP', "PointMLP"),
        ("pointMLP2", "PointMLP-Elite")
    ]),
    'train_mode': OrderedDict([
        ('cutmix_r', 'PointCutMix-R'),
        ('cutmix_k', 'PointCutMix-K'), 
        ('mixup', 'PointMixup'),
        ('rsmix', 'RSMix'),
        ('bn', 'BN'),
        ('tent', 'TENT'),
        ('pgd', 'PGD'),
        ('none', "Standard"),
        ('megamerger', "megamerger")
    ]),
    'corruption': OrderedDict([
        ("occlusion", "Occlusion"),
        ("lidar", "LiDAR"),
        ("density_inc", "Local_Density_Inc"),
        ("density", "Local_Density_Dec"),
        ("cutout", "Cutout"),
        ("uniform", "Uniform"),
        ("gaussian", "Gaussian"),
        ("impulse", "Impulse"),
        ("upsampling", "Upsampling"),
        ("background", "Background"),
        ("rotation", "Rotation"),
        ("shear", "Shear"),
        ("distortion", "FFD"),
        ("distortion_rbf", "RBF"),
        ("distortion_rbf_inv", "Inv_RBF"),
        ("none", "none"),
    ]),
    'severity': [1, 2, 3, 4, 5],
    'metric': ['acc', "class_acc", "err", "class_err"]
}
