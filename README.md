#  LKPM:Large Kernel Point Mamba for 3D Point Clouds

This repository contains the implementation of **LKPM**. 


![teaser](Figure1-Model%20Overview.png)



## 1. Installation

The code has been tested on Ubuntu 22.04 .


1. Install [Conda](https://www.anaconda.com/) and create a `Conda` environment.

    ```bash
    conda create --name lkpm python=3.10
    conda activate lkpm
    ```

2. Install PyTorch-2.2.2 with conda according to the official documentation.

    ```bash
    conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

3. Clone this repository and install the requirements.

    ```bash
    pip install -r requirements.txt
    ```

4. Install the library for octree-based depthwise convolution.

    ```bash
    git clone https://github.com/octree-nn/dwconv.git
    pip install ./dwconv
    ```
   
5. Install ``causal-conv1d`` and ``mamba``, which you can download in this [link](https://sjtueducn-my.sharepoint.com/:u:/g/personal/yj1938_sjtu_edu_cn/EfvXT20i7IBPsw_KR47ok3wB0l531kf7DMQwJWjdnPxlkQ?e=iDhOe9).
    ```bash
    pip install -e causal-conv1d
    pip install -e mamba
    ```

6. To run the detection experiments,
   [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) is required.
   And the code is tested with `mmdet3d==1.0.0rc5`. Run the following commands
   to install it. For detailed instructions, please refer to the
   [official documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html#installation).
   Make sure the results of
   [FCAF3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcaf3d/README.md)
   can be reproduced before running the experiments.

    ```bash
    pip install openmim==0.3.3
    mim install mmcv-full==1.6.2
    mim install mmdet==2.26.0
    mim install mmsegmentation==0.29.1
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout v1.0.0rc5
    pip install -e .
    ```

## 2. ScanNet Segmentation

1. **Data**: Download the data from the
   [ScanNet benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/).
   Unzip the data and place it to the folder <scannet_folder>. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet.py --run process_scannet --path_in <scannet_folder>
    ```

2. **Train**: Run the following command to train the network with 4 GPUs and
   port 10001. The mIoU on the validation set without voting is 76.4.And the training
   weights can be downloaded
   [here](...).

    ```bash
    python scripts/run_seg_scannet.py --gpu 0,1,2,3 --alias scannet --port 10001
    ```

3. **Evaluate**: Run the following command to get the per-point predictions for
   the validation dataset with a voting strategy. And after voting, the mIoU is
   77 on the validation dataset.

    ```bash
    python scripts/run_seg_scannet.py --gpu 0 --alias scannet --run validate
    ```


## 3. ScanNet200 Segmentation


1. **Data**: Download the data from the
   [ScanNet benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/).
   Unzip the data and place it to the folder <scannet_folder>. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet.py --run process_scannet --path_in <scannet_folder>  \
           --path_out data/scanet200.npz  --align_axis  --scannet200
    ```

2. **Train**: Run the following command to train the network with 4 GPUs. The
    training log and weights can be downloaded
   [here](....).
   
    ```bash
    python scripts/run_seg_scannet200.py --gpu 0,1,2,3 --alias scannet --port 10001
    ```

3. **Evaluate**: Run the following command to get the per-point predictions for
   the validation dataset with a voting strategy. And after voting, the mIoU is
   32.8 on the validation dataset.

    ```bash
    python scripts/run_seg_scannet200.py --gpu 0 --alias scannet200 --run validate
    ```
## 4. ModelNet40 Classification

1. **Data**: Run the following command to prepare the dataset.

    ```bash
    python tools/cls_modelnet.py
    ```

2. **Train**: Run the following command to train the network with 1 GPU. The
   classification accuracy on the testing set without voting is 94.8%.
    ```bash
    python classification.py --config configs/cls_m40.yaml SOLVER.gpu 0,
    ```

## 5. SUN RGB-D Detection

1. **Data**: Prepare the data according to the
   [official documentation](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/sunrgbd.html)
   of mmdetection3d. Denote the path to the data as <sunrgbd_folder>. Run the
   following command to build a symbolic link to the data.

    ```bash
    ln -s <sunrgbd_folder> data/sunrgbd
    ```

2. **Training**: Run the following command to train the network with 4 GPUs. The
    maximum mAP@0.25 and mAP@05 on the validation set are 69.3 and 48.5,
    respectively. The training log and weights can be downloaded
    [here](...).

    ```bash
    CUDA_VISIBLE_DEVICES=0 python detection.py configs/det_sunrgbd.py
        --work-dir=logs/sunrgbd/lkpm
    ```
## 6. Acknowledgement 
Our project is based on 
- Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba))
- Octformer([paper](https://arxiv.org/abs/2305.03045), [code](https://github.com/octree-nn/octformer))

Thanks for their wonderful works!
