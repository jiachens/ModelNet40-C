# Learning Geometry-Disentangled Representation for Complementary Understanding of 3D Object Point Cloud. 
This repository is built for the paper:

__Learning Geometry-Disentangled Representation for Complementary Understanding of 3D Object Point Cloud (_AAAI2021_)__ [[arXiv](https://arxiv.org/abs/2012.10921)]
<br>
by [Mutian Xu*](https://mutianxu.github.io/), [Junhao Zhang*](https://junhaozhang98.github.io/), Zhipeng Zhou, Mingye Xu, Xiaojuan Qi and Yu Qiao.


## Overview
Geometry-Disentangled Attention Network for 3D object point cloud classification and segmentation (GDANet):
<img src = './imgs/GDANet.jpg' width = 800>

## Citation
If you find the code or trained models useful, please consider citing:

    @misc{xu2021learning,
      title={Learning Geometry-Disentangled Representation for Complementary Understanding of 3D Object Point Cloud}, 
      author={Mutian Xu and Junhao Zhang and Zhipeng Zhou and Mingye Xu and Xiaojuan Qi and Yu Qiao},
      year={2021},
      eprint={2012.10921},
      archivePrefix={arXiv},
      primaryClass={cs.CV}


## Installation


### Requirements
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.5+
* PyTorch 1.0+

### Dataset
* Create the folder to symlink the data later:
    
    `mkdir -p data`
    
* __Object Classification__: 

    Download and unzip [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) (415M), then symlink the path to it as follows (you can alternatively modify the path [here](https://github.com/mutianxu/GDANet/blob/main/util/data_util.py#L12)) :
    
    `ln -s /path to modelnet40/modelnet40_ply_hdf5_2048 data`
    
* __Shape Part Segmentation__:
    
    Download and unzip [ShapeNet Part](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) (674M), then symlink the path to it as follows (you can alternatively modify the path [here](https://github.com/mutianxu/GDANet/blob/main/util/data_util.py#L70)) :
    
    `ln -s /path to shapenet part/shapenetcore_partanno_segmentation_benchmark_v0_normal data`

## Usage

### Object Classification on ModelNet40
* Train:
 
    `python main_cls.py`

* Test:
    * Run the voting evaluation script, after this voting you will get an accuracy of 93.8% if all things go right:
    
        `python voting_eval_modelnet.py --model_path 'pretrained/GDANet_ModelNet40_93.4.t7'`
    
    * You can also directly evaluate our pretrained model without voting to get an accuracy of 93.4%:
    
        `python main.py --eval True --model_path 'pretrained/GDANet_ModelNet40_93.4.t7'`
    
### Shape Part Segmentation on ShapeNet Part
* Train:
    * Training from scratch:

        `python main_ptseg.py`
   
    * If you want resume training from checkpoints, specify `resume` in the args:

        `python main_ptseg.py --resume True`

* Test:

    You can choose to test the model with the best instance mIoU, class mIoU or accuracy, by specifying `model_type` in the args:
    
    * `python main_ptseg.py --model_type 'ins_iou'` (best instance mIoU, default)
    
    * `python main_ptseg.py --model_type 'cls_iou'` (best class mIoU)
    
    * `python main_ptseg.py --model_type 'acc'` (best accuracy)


## Other information

Please contact Mutian Xu (mino1018@outlook.com) or Junhao Zhang (junhaozhang98@gmail.com) for further discussion.

## Acknowledgement
This code is is partially borrowed from [DGCNN](https://github.com/WangYueFt/dgcnn) and [PointNet++](https://github.com/charlesq34/pointnet2).  