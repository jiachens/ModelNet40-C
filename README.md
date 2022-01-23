**Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions** <br>

NOTE: All the identifiable information in this codebase is not from the authors, but from the existing published papers or open-sourced projects.

This Codebase is based on [SimpleView](https://github.com/princeton-vl/SimpleView), and we thank the authors for their great contributions:
```
@article{goyal2021revisiting,
  title={Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline},
  author={Goyal, Ankit and Law, Hei and Liu, Bowei and Newell, Alejandro and Deng, Jia},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## Getting Started

First clone the repository. We would refer to the directory containing the code as `ModelNet40-C`.

```
git clone git@github.com:jiachens/ModelNet40-C.git
```

#### Requirements
The code is tested on Linux OS with Python version **3.7.5**, CUDA version **10.0**, CuDNN version **7.6** and GCC version **5.4**. We recommend using these versions especially for installing [pointnet++ custom CUDA modules](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/22e8cf527b696b63b66f3873d80ae5f93744bdef).

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
```
cd pointnet2_pyt && pip install -e . && cd ..
```
We also nned to install the earth mover distance module.
```
cd emd && python setup.py install && cd ..
```
We also need to install PyGem, since we it for the ModelNet40-C dataset generation.
```
cd PyGem && python setup.py install && cd ..
```

To generate the ModelNet40-C dataset, please run:
```
python data/generate_c.py
```
NOTE that the generation needs a monitor connected since Open3D library does not support background rendering. 

#### Download Datasets Including ModelNet40-C and Pre-trained Models
Make sure you are in `ModelNet40-C`. `download.sh` script can be used for downloading all the data and the pretrained models. It also places them at the correct locations. First, use the following command to provide execute permission to the `download.sh` script. 
```
chmod +x download.sh
```
We also allow users to download ModelNet40-C directly:
```
./download.sh modelnet40_c
```
To download ModelNet40 execute the following command. This will download the ModelNet40 point cloud dataset released with pointnet++ as well as the validation splits used in our work.
```
./download.sh modelnet40
```
To download the pretrained models with standard training recipe, execute the following command.
```
./download.sh cor_exp
```
To download the pretrained models using different data augmentation strategies, execute the following command.
```
./download.sh runs
```

## New Features
- We include Point Cloud Transformer (PCT) in our benchmark
- `ModelNet40-C/configs` contains config files to enable different data augmentations and test-time adaptation methods
- `ModelNet40-C/aug_utils.py` contains the data augmentation codes in our paper
- `ModelNet40-C/third_party` contains the test-time adaptation used in our paper

## Code Organization In Originial SimpleView
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

We additionally leverage PointCutMix: `configs/cutmix`, PointMixup: `configs/mixup`, and RSMix: `configs/rsmix` as the training time config files.


#### Evaluate a pretrained model
We provide pretrained models. They can be downloaded using the `./download cor_exp` and `./download runs` command and are stored in the `SimpleView/pretrained` folder. To test a pretrained model, the command is of the following format.

```
python main.py --entry test --model-path <cor_exp/runs>/<cfg_name>/<model_name>.pth --exp-config configs/<cfg_name>.yaml
```

We list all the evaluation commands in the `eval_*.sh` script. 

