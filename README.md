# Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarking-robustness-of-3d-point-cloud/3d-point-cloud-classification-on-modelnet40-c)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40-c?p=benchmarking-robustness-of-3d-point-cloud)

This repo contains the dataset and code for the paper [Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions](https://arxiv.org/abs/2201.12296) by Jiachen Sun et al. This codebase is based on [SimpleView](https://github.com/princeton-vl/SimpleView), and we thank the authors for their great contributions.

## ModelNet40-C
![image](https://github.com/jiachens/ModelNet40-C/blob/master/img/1.png)

![image](https://github.com/jiachens/ModelNet40-C/blob/master/img/54.png)


More visualizations can be found [here](https://github.com/jiachens/ModelNet40-C/blob/master/img).

[Download ModelNet40-C from Google Drive.](https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX?usp=sharing)

[Download ModelNet40-C using our provided script.](#download-datasets-including-modelnet40-c-and-pre-trained-models)

[Download ModelNet40-C from Zenodo.](https://zenodo.org/record/6017834#.YgNeKu7MK3J)

## ModelNet40-C Leaderboard 

**Architecture+Data Augmentation Leaderboard** </br>

| **Architecture** | **Data Augmentation**    | **Corruption Error Rate (%)** | **Clean Error Rate (%)** | **Checkpoint**                                                                                   |
|------------------|---------------|:-------------------------:|:--------------------:|--------------------------------------------------------------------------------------------------|
| PCT              | PointCutMix-R |            16.3           |          7.2         | [checkpoint](https://drive.google.com/file/d/1OcH0o7V_RhAOj9pSuS39G43VrBWS-1v3/view?usp=sharing) |
| PCT              | PointCutMix-K |            16.5           |          6.9         | [checkpoint](https://drive.google.com/file/d/1T4KwMkgAqAItHZc-Q96H1qGMPNoObJkJ/view?usp=sharing) |
| DGCNN            | PointCutMix-R |            17.3           |          6.8         | [checkpoint](https://drive.google.com/file/d/1Z_6D_MmjecDHhY2q-I-aok9nlD9RkAS1/view?usp=sharing) |
| PCT              | RSMix         |            17.3           |          6.9         | [checkpoint](https://drive.google.com/file/d/18BqbMCpdbEGdyQVdMwYPDrff5bmgeF9B/view?usp=sharing) |
| DGCNN            | PointCutMix-K |            17.3           |          7.4         | [checkpoint](https://drive.google.com/file/d/1rUQApmyEJUpv7JzkhJuwEeZOmmZ9vEDU/view?usp=sharing) |
| RSCNN            | PointCutMix-R |            17.9           |          7.6         | [checkpoint](https://drive.google.com/file/d/1EggUiFcCgpHOwjgQKRgBxST1utfOAryc/view?usp=sharing) |
| DGCNN            | RSMix         |            18.1           |          7.1         | [checkpoint](https://drive.google.com/file/d/11tNaF-YsJ6hZNm2pY6LX6Ny-ceGkI0Cr/view?usp=sharing) |
| PCT              | PGD Adv Train |            18.4           |          8.9         | [checkpoint](https://drive.google.com/file/d/1Y7JaW-CLPCcqQQGiuL9BfVKfAkm6MxEA/view?usp=sharing) |
| PointNet++       | PointCutMix-R |            19.1           |          7.1         | [checkpoint](https://drive.google.com/file/d/1un_H1oq18MrN604mbR3htBNqdOgnXqwQ/view?usp=sharing) |
| PointNet++       | PointMixup    |            19.3           |          7.1         | [checkpoint](https://drive.google.com/file/d/1fzFOeJcenn7a4glsfs7IEcSTjovZoAkB/view?usp=sharing) |
| PCT              | PointMixup    |            19.5           |          7.4         | [checkpoint](https://drive.google.com/file/d/1OcBm-PCImcW8h1mb9ZY4CcX2nDN_rB8b/view?usp=sharing) |
| SimpleView       | PointCutMix-R |            19.7           |          7.9         | [checkpoint](https://drive.google.com/file/d/178LQKtmCeNIbdPXYZXZHRmAQt-YCY_eL/view?usp=sharing) |
| RSCNN            | PointMixup    |            19.8           |          7.2         | [checkpoint](https://drive.google.com/file/d/1FRPU_QTR3vda1CqPWKkREprIZshv4cYk/view?usp=sharing) |
| PointNet++       | PointCutMix-K |            20.2           |          6.7         | [checkpoint](https://drive.google.com/file/d/1JLL7ym-fMUS4VFisf-AENB5trYJb_0-J/view?usp=sharing) |

We allow users to directly download all pre-trained models with every data augmentation method [here](#download-datasets-including-modelnet40-c-and-pre-trained-models).

**Architecture Leaderboard** </br>

| **Architecture** | **Corruption Error Rate (%)** | **Clean Error Rate (%)** | **Checkpoint**                                                                                   | 
|------------------|:-------------------------:|:--------------------:|--------------------------------------------------------------------------------------------------|
| CurveNet       |            22.7           |          6.6         | checkpoint |
| PointNet++       |            23.6           |          7.0         | [checkpoint](https://drive.google.com/file/d/18_297KJ8slsJq1rGDsvuQ29VICs-EJTa/view?usp=sharing) |
| PCT              |            25.5           |          7.1         | [checkpoint](https://drive.google.com/file/d/1NFAhupQKn-sBLYW1YpUAf4jdqMpFcV7Z/view?usp=sharing) |
| GDANet       |            25.6           |          7.5         | checkpoint |
| DGCNN            |            25.9           |          7.4         | [checkpoint](https://drive.google.com/file/d/1JMCmujJM4J_OyxuZuDN4befFmtG1_p49/view?usp=sharing) |
| RSCNN            |            26.2           |          7.7         | [checkpoint](https://drive.google.com/file/d/1RKhXKjZvKvZM2the2qqFhnytAX2H634U/view?usp=sharing) |
| SimpleView       |            27.2           |          6.1         | [checkpoint](https://drive.google.com/file/d/1jscF5p3Q7DHWl-FgGGemQP3CeXITsTyY/view?usp=sharing) |
| PointNet         |            28.3           |          9.3         | [checkpoint](https://drive.google.com/file/d/1eW26u0nm6HETwDSiCyCEoLLY3WnOVt73/view?usp=sharing) |
| PointMLP         |            31.9           |          6.3         | checkpoint |
| PointMLP-Elite         |            32.4           |          7.2         | checkpoint |

More models' results coming soon ......

We allow users to directly download all pre-trained models with standard training [here](#download-datasets-including-modelnet40-c-and-pre-trained-models).

## Getting Started

First clone the repository. We would refer to the directory containing the code as `ModelNet40-C`.

```
git clone --recurse-submodules git@github.com:jiachens/ModelNet40-C.git
```

#### Requirements
The code is tested on Linux OS with Python version **3.7.5**, CUDA version **10.0**, CuDNN version **7.6** and GCC version **5.4**. We recommend using these versions especially for installing [pointnet++ custom CUDA modules](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/22e8cf527b696b63b66f3873d80ae5f93744bdef).

[02-23-2022] The updated codes are tested on Python version **3.7.5**, CUDA version **11.4**, CuDNN version **8.2** and GCC version **7.5** with the latest ```torch``` and ```torchvision``` libs, but we still suggest the original setup in case of any unstable bugs.

#### Install Libraries
We recommend you first install [Anaconda](https://anaconda.org/) and create a virtual environment.
```
conda create --name modelnetc python=3.7.5
```

Activate the virtual environment and install the libraries. Make sure you are in `ModelNet40-C`.
```
conda activate modelnetc
pip install -r requirements.txt
conda install sed  # for downloading data and pretrained models
```

For PointNet++, we need to install custom CUDA modules. Make sure you have access to a GPU during this step. You might need to set the appropriate `TORCH_CUDA_ARCH_LIST` environment variable depending on your GPU model. The following command should work for most cases `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"`. However, if the install fails, check if `TORCH_CUDA_ARCH_LIST` is correctly set. More details could be found [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

Third-party modules `pointnet2_pyt`, `PCT_Pytorch`, `emd`, and `PyGeM` can be installed by the following script.

```
./setup.sh
```

#### Download Datasets Including ModelNet40-C and Pre-trained Models
Make sure you are in `ModelNet40-C`. `download.sh` script can be used for downloading all the data and the pretrained models. It also places them at the correct locations.

To download ModelNet40 execute the following command. This will download the ModelNet40 point cloud dataset released with pointnet++ as well as the validation splits used in our work.
```
./download.sh modelnet40
```
To generate the ModelNet40-C dataset, please run:
```
python data/process.py
python data/generate_c.py
```
NOTE that the generation needs a monitor connected since Open3D library does not support background rendering. 

We also allow users to download ModelNet40-C directly. Please fill this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSdrzt8EtQdjGMlwIwWAzb39KzzVzijpK6-sPEaps07MjQwGGQ/viewform?usp=sf_link) while downloading our dataset.
```
./download.sh modelnet40_c
```
To download the pretrained models with standard training recipe, execute the following command.
```
./download.sh cor_exp
```
To download the pretrained models using different data augmentation strategies, execute the following command.
```
./download.sh runs
```

#### New Features
\[02-23-2022\]
- We include PointMLP-Elite and GDANet in our benchmark

\[02-18-2022\]
- We include CurveNet and PointMLP in our benchmark

\[01-28-2022\]
- We include Point Cloud Transformer (PCT) in our benchmark
- `ModelNet40-C/configs` contains config files to enable different data augmentations and test-time adaptation methods
- `ModelNet40-C/aug_utils.py` contains the data augmentation codes in our paper
- `ModelNet40-C/third_party` contains the test-time adaptation used in our paper

#### Code Organization In Originial SimpleView
- `ModelNet40-C/models`: Code for various models in PyTorch.
- `ModelNet40-C/configs`: Configuration files for various models.
- `ModelNet40-C/main.py`: Training and testing any models.
- `ModelNet40-C/configs.py`: Hyperparameters for different models and dataloader.
- `ModelNet40-C/dataloader.py`: Code for different variants of the dataloader.
- `ModelNet40-C/*_utils.py`: Code for various utility functions.

 
## Running Experiments

#### Training and Config files
To train or test any model, we use the `main.py` script. The format for running this script is as follows. 
```
python main.py --exp-config <path to the config>
```

The config files are named as `<protocol>_<model_name><_extra>_run_<seed>.yaml` (`<protocol> ∈ [dgcnn, pointnet2, rscnn]`; `<model_name> ∈ [dgcnn, pointnet2, rscnn, pointnet, simpleview]` ). For example, the config file to run an experiment for PointNet++ in DGCNN protocol with seed 1 `dgcnn_pointnet2_run_1.yaml`. To run a new experiment with a different seed, you need to change the `SEED` parameter in the config file. All of our experiments are done based on seed 1.

We additionally leverage PointCutMix: `configs/cutmix`, PointMixup: `configs/mixup`, RSMix: `configs/rsmix`, and PGD-based adversarial training `configs/pgd` as the training-time config files.

For example, to train PCT with PointCutMix-R, please use the following command:
```
python main.py --exp-config configs/cutmix/pct_r.yaml
```

#### Evaluate a pretrained model
We provide pretrained models. They can be downloaded using the `./download.sh cor_exp` and `./download.sh runs` commands and are stored in the `ModelNet40-C/runs` (for data augmentation recipes) and `ModelNet40-C/cor_exp` (for standard trained models) folders. To test a pretrained model, the command is of the following format.

Additionally, we provide test-time config files in `configs/bn` and `configs/tent` for BN and TENT in our paper with the following commands:
```
python main.py --entry test --model-path <cor_exp/runs>/<cfg_name>/<model_name>.pth --exp-config configs/<cfg_name>.yaml
```

We list all the evaluation commands in the `eval_cor.sh`, `eval_og.sh`, `eval_tent_cutmix.sh` scripts. Note that in `eval_cor.sh` it is expected that pgd with PointNet++, RSCNN, and SimpleView do not have outputs since they do not fit the adversarial training framework. We have mentioned this in our paper.

## Citation
Please cite our paper and SimpleView if you use our benchmark and analysis results. Thank you!
```
@article{sun2022benchmarking,
      title={Benchmarking Robustness of 3D Point Cloud Recognition Against Common Corruptions}, 
      author={Jiachen Sun and Qingzhao Zhang and Bhavya Kailkhura and Zhiding Yu and Chaowei Xiao and Z. Morley Mao},
      journal={arXiv preprint arXiv:2201.12296},
      year={2022}
}
```
```
@article{goyal2021revisiting,
  title={Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline},
  author={Goyal, Ankit and Law, Hei and Liu, Bowei and Newell, Alejandro and Deng, Jia},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## References

[1] [Zhang, Jinlai, et al. "PointCutMix: Regularization Strategy for Point Cloud Classification." arXiv preprint arXiv:2101.01461 (2021).](https://arxiv.org/pdf/2101.01461.pdf)

[2] [Chen, Yunlu, et al. "Pointmixup: Augmentation for point clouds." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer International Publishing, 2020.](https://link.springer.com/content/pdf/10.1007/978-3-030-58580-8_20.pdf)

[3] [Lee, Dogyoon, et al. "Regularization Strategy for Point Cloud via Rigidly Mixed Sample." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Regularization_Strategy_for_Point_Cloud_via_Rigidly_Mixed_Sample_CVPR_2021_paper.pdf)

[4] [Sun, Jiachen, et al. "Adversarially Robust 3D Point Cloud Recognition Using Self-Supervisions." Advances in Neural Information Processing Systems 34 (2021).](https://proceedings.neurips.cc/paper/2021/file/82cadb0649a3af4968404c9f6031b233-Paper.pdf)

[5] [Schneider, Steffen, et al. "Improving robustness against common corruptions by covariate shift adaptation." arXiv preprint arXiv:2006.16971 (2020).](https://arxiv.org/pdf/2006.16971.pdf)

[6] [Wang, Dequan, et al. "Tent: Fully test-time adaptation by entropy minimization." arXiv preprint arXiv:2006.10726 (2020).](https://arxiv.org/pdf/2006.10726.pdf)

[7] [Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)

[8] [Qi, Charles R., et al. "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." arXiv preprint arXiv:1706.02413 (2017).](https://arxiv.org/pdf/1706.02413.pdf)

[9] [Liu, Yongcheng, et al. "Relation-shape convolutional neural network for point cloud analysis." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Relation-Shape_Convolutional_Neural_Network_for_Point_Cloud_Analysis_CVPR_2019_paper.pdf)

[10] [Wang, Yue, et al. "Dynamic graph cnn for learning on point clouds." Acm Transactions On Graphics (tog) 38.5 (2019): 1-12.](https://dl.acm.org/doi/pdf/10.1145/3326362)

[11] [Goyal, Ankit, et al. "Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline." arXiv preprint arXiv:2106.05304 (2021).](https://arxiv.org/pdf/2106.05304.pdf)

[12] [Guo, Meng-Hao, et al. "Pct: Point cloud transformer." Computational Visual Media 7.2 (2021): 187-199.](https://link.springer.com/content/pdf/10.1007/s41095-021-0229-5.pdf)


[13] [Xiang, Tiange, et al. "Walk in the cloud: Learning curves for point clouds shape analysis." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.](https://arxiv.org/pdf/2105.01288.pdf)

[14] [Ma, Xu, et al. "Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework." arXiv preprint arXiv:2202.07123 (2022).](https://arxiv.org/pdf/2202.07123.pdf)

[15] [Learning Geometry-Disentangled Representation for Complementary Understanding of 3D Object Point Cloud.](https://arxiv.org/abs/2012.10921)
