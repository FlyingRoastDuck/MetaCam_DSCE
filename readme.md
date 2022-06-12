## [Joint Noise-Tolerant Learning and Meta Camera Shift Adaptation for Unsupervised Person Re-Identification](https://arxiv.org/abs/2103.04618) (CVPR'21)

### Introduction

This is the official repo for the CVPR 2021 paper "MetaCam+DSCE".

[2021.5.24] 
We recorded a video on [Zhidongxi](https://course.zhidx.com/c/OWYyZTMxNzJhYTA2YzEyYjZhYjM=).

![](figures/metacam.png)

### Prerequisites

- CUDA>=10.0
- At least two 1080-Ti GPUs 
- Other necessary packages listed in [requirements.txt](requirements.txt)
- Training Data
  
  (Market-1501, DukeMTMC-reID and MSMT-17. You can download these datasets from [Zhong's repo](https://github.com/zhunzhong07/ECN))

   Unzip all datasets and ensure the file structure is as follow:
   
   ```
   MetaCam_DSCE/data    
   │
   └───market1501 OR dukemtmc OR msmt17
        │   
        └───DukeMTMC-reID OR Market-1501-v15.09.15 OR MSMT17_V1
            │   
            └───bounding_box_train
            │   
            └───bounding_box_test
            | 
            └───query
            │   
            └───list_train.txt (only for MSMT-17)
            | 
            └───list_query.txt (only for MSMT-17)
            | 
            └───list_gallery.txt (only for MSMT-17)
            | 
            └───list_val.txt (only for MSMT-17)
   ```

### Usage

See [run.sh](run.sh) for details.


### Acknowledgments
This repo borrows partially from [MWNet (meta-learning)](https://github.com/xjtushujun/meta-weight-net), 
[ECN (exemplar memory)](https://github.com/zhunzhong07/ECN) and 
[SpCL (faiss-based acceleration)](https://github.com/yxgeee/SpCL). 
If you find our code useful, please cite their papers.

### Resources

1. Pre-trained MMT-500 models to reproduce Tab. 3 of our paper. 
   [BaiduNetDisk](https://pan.baidu.com/s/1GDMFDrOpd3H7FA3t35Sv6A), Passwd: jr1l.
   [Google Drive](https://drive.google.com/drive/folders/1sEi9fOeNQmrjQ4ZEQ3sbsKg26Dc6Boe1?usp=sharing).
   
2. Pedestrian images used to plot Fig.3 in our paper. 
   [BaiduNetDisk](https://pan.baidu.com/s/1ahoj3fk-6OwCM4yWeDJbiQ), Passwd: f248.
   [Google Drive](https://drive.google.com/file/d/1lk4DbkJR9BWpVFb_AnFUlFJPmQVrj-MX/view?usp=sharing).
   
   Please download 'marCam' and 'dukeCam', 
   put them under 'MetaCam_DSCE/data', 
   uncomment L#87-89 and L#163-168 of train_usl_knn_merge.py 
   to visualize pedestrian features.
   
3. Training logs.
   [BaiduNetDisk](https://pan.baidu.com/s/1Dq2PjJXDLfjM8gQIQcbN6A), Passwd: mecq.
   [Google Drive](https://drive.google.com/drive/folders/15jxAP0E1K6rE9Z4jH64Qy5bO8O6jA96A?usp=sharing).
   


### How to Cite
```yang
@inproceedings{yang2021joint,
  title={Joint Noise-Tolerant Learning and Meta Camera Shift Adaptation for Unsupervised Person Re-Identification},
  author={Yang Fengxiang and Zhong Zhun and Luo Zhiming and Cai Yuanzheng and Lin Yaojin and Li Shaozi and Nicu Sebe},
  booktitle={CVPR},
  pages={4855--4864},
  year={2021}
}
```

### Contact Us

Email: yangfx@stu.xmu.edu.cn
